/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.linalg.distributed

import org.apache.spark.ml.linalg.{VMatrix, Vector}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.HashMap
import org.apache.spark.Partitioner
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag

class VBlockMatrix(
    val rowsPerBlock: Int,
    val colsPerBlock: Int,
    val blocks: RDD[((Int, Int), VMatrix)],
    val gridPartitioner: VGridPartitioner) {

  final val mapJoinPartitionsShuffleRdd2 =
    System.getProperty("vflbfgs.mapJoinPartitions.shuffleRdd2", "true").toBoolean

  def horizontalZipVector[T: ClassTag](vector: DistributedVector)(
      f: (((Int, Int), VMatrix, Vector) => T)
    ): RDD[((Int, Int), T)] = {

    import org.apache.spark.rdd.VRDDFunctions._

    val gridPartitionerParam = gridPartitioner
    require(gridPartitionerParam.colBlocks == vector.numPartitions)
    blocks.mapJoinPartition(vector.values, mapJoinPartitionsShuffleRdd2)(
      (pid: Int) => {
        // pid is the partition ID of blockMatrix RDD
        val colPartId = gridPartitionerParam.colPartId(pid)
        val startIdx = colPartId * gridPartitionerParam.colBlocksPerPart
        var endIdx = startIdx + gridPartitionerParam.colBlocksPerPart
        if (endIdx > gridPartitionerParam.colBlocks) endIdx = gridPartitionerParam.colBlocks
        (startIdx until endIdx).toArray // The corresponding partition ID of `vector`
      },
      (pid: Int,
       mIter: Iterator[((Int, Int), VMatrix)],
       vIter: Array[(Int, Iterator[Vector])]) => {
        val vMap = new HashMap[Int, Vector]
        util.Random.shuffle(vIter.toList).foreach { case (colId: Int, iter: Iterator[Vector]) =>
            val v = iter.next()
            assert(!iter.hasNext)
            vMap += (colId -> v)
        }
        mIter.map { case ((rowBlockIdx: Int, colBlockIdx: Int), sm: VMatrix) =>
          val partVector = vMap(colBlockIdx)
          ((rowBlockIdx, colBlockIdx), f((rowBlockIdx, colBlockIdx), sm, partVector))
        }
      }
    )
  }

  def verticalZipVector[T: ClassTag](vec: DistributedVector)(
      f: (((Int, Int), VMatrix, Vector) => T)
    ): RDD[((Int, Int), T)] = {

    import org.apache.spark.rdd.VRDDFunctions._

    val gridPartitionerParam = gridPartitioner
    require(gridPartitionerParam.rowBlocks == vec.numPartitions)
    blocks.mapJoinPartition(vec.values, mapJoinPartitionsShuffleRdd2)(
      (pid: Int) => {
        val rowPartId = gridPartitionerParam.rowPartId(pid)
        val startIdx = rowPartId * gridPartitionerParam.rowBlocksPerPart
        var endIdx = startIdx + gridPartitionerParam.rowBlocksPerPart
        if (endIdx > gridPartitionerParam.rowBlocks) endIdx = gridPartitionerParam.rowBlocks
        (startIdx until endIdx).toArray
      },
      (pid: Int, mIter: Iterator[((Int, Int), VMatrix)],
       vIter: Array[(Int, Iterator[Vector])]) => {
        val vMap = new HashMap[Int, Vector]
        util.Random.shuffle(vIter.toList).foreach {
          case (rowId: Int, iter: Iterator[Vector]) =>
            val v = iter.next()
            assert(!iter.hasNext)
            vMap += (rowId -> v)
        }
        mIter.map { case ((rowBlockIdx: Int, colBlockIdx: Int), sm: VMatrix) =>
          val partVector = vMap(rowBlockIdx)
          ((rowBlockIdx, colBlockIdx), f((rowBlockIdx, colBlockIdx), sm, partVector))
        }
      }
    )
  }

  def horizontalZipVector2(vector: DistributedVector)(
    f: (((Int, Int), VMatrix, Vector) => VMatrix)
  ): VBlockMatrix = {
    val newBlocks = horizontalZipVector(vector)(f)
    new VBlockMatrix(rowsPerBlock, colsPerBlock, newBlocks, gridPartitioner)
  }

  def verticalZipVector2(vector: DistributedVector)(
    f: (((Int, Int), VMatrix, Vector) => VMatrix)
  ): VBlockMatrix = {
    val newBlocks = verticalZipVector(vector)(f)
    new VBlockMatrix(rowsPerBlock, colsPerBlock, newBlocks, gridPartitioner)
  }

  def persist(storageLevel: StorageLevel) = {
    blocks.persist(storageLevel)
    this
  }
}

private[spark] class VGridPartitioner(
    val rowBlocks: Int,
    val colBlocks: Int,
    val rowBlocksPerPart: Int,
    val colBlocksPerPart: Int) extends Partitioner {

  require(rowBlocks > 0)
  require(colBlocks > 0)
  require(rowBlocksPerPart > 0)
  require(colBlocksPerPart > 0)

  val rowPartitions = math.ceil(rowBlocks * 1.0 / rowBlocksPerPart).toInt
  val colPartitions = math.ceil(colBlocks * 1.0 / colBlocksPerPart).toInt

  override def numPartitions: Int = rowPartitions * colPartitions

  /**
   * Returns the index of the partition the input coordinate belongs to.
   *
   * @param key The partition id i (calculated through this method for coordinate (i, j) in
   *            `simulateMultiply`, the coordinate (i, j) or a tuple (i, j, k), where k is
   *            the inner index used in multiplication. k is ignored in computing partitions.
   * @return The index of the partition, which the coordinate belongs to.
   */
  override def getPartition(key: Any): Int = {
    key match {
      case i: Int => i
      case (i: Int, j: Int) =>
        getPartitionId(i, j)
      case (i: Int, j: Int, _: Int) =>
        getPartitionId(i, j)
      case _ =>
        throw new IllegalArgumentException(s"Unrecognized key: $key.")
    }
  }

  /** Partitions sub-matrices as blocks with neighboring sub-matrices. */
  def getPartitionId(i: Int, j: Int): Int = {
    require(0 <= i && i < rowBlocks, s"Row index $i out of range [0, $rowBlocks).")
    require(0 <= j && j < colBlocks, s"Column index $j out of range [0, $colBlocks).")
    i / rowBlocksPerPart + j / colBlocksPerPart * rowPartitions
  }

  def rowPartId(partId: Int) = partId % rowPartitions
  def colPartId(partId: Int) = partId / rowPartitions

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: VGridPartitioner =>
        (this.rowBlocks == r.rowBlocks) && (this.colBlocks == r.colBlocks) &&
          (this.rowBlocksPerPart == r.rowBlocksPerPart) && (this.colBlocksPerPart == r.colBlocksPerPart)
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(
      rowBlocks: java.lang.Integer,
      colBlocks: java.lang.Integer,
      rowBlocksPerPart: java.lang.Integer,
      colBlocksPerPart: java.lang.Integer)
  }

  def getRowSplitedGridPartitioner(numSplits: Int) = {
    new VRowSplitedGridPartitioner(
      rowBlocks, colBlocks, rowBlocksPerPart, colBlocksPerPart, numSplits)
  }

}

private[spark] class VRowSplitedGridPartitioner(
    override val rowBlocks: Int,
    override val colBlocks: Int,
    override val rowBlocksPerPart: Int,
    override val colBlocksPerPart: Int,
    val numSplits: Int)
  extends VGridPartitioner(rowBlocks, colBlocks, rowBlocksPerPart, colBlocksPerPart)
{
  override def getPartition(key: Any): Int = {
    key match {
      case (i: Int, j: Int) =>
        val gridId = super.getPartitionId(i, j)
        (i % numSplits) * super.numPartitions + gridId
      case _ =>
        throw new IllegalArgumentException(s"Unrecognized key: $key.")
    }
  }

  override def numPartitions: Int = super.numPartitions * numSplits

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: VRowSplitedGridPartitioner =>
        super.equals(obj) && this.numSplits == r.numSplits
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(
      super.hashCode: java.lang.Integer,
      numSplits: java.lang.Integer)
  }
}

private[spark] object VGridPartitioner {

  /** Creates a new [[VGridPartitioner]] instance. */
  def apply(rowBlocks: Int, colBlocks: Int, rowBlocksPerPart: Int, colBlocksPerPart: Int): VGridPartitioner = {
    new VGridPartitioner(rowBlocks, colBlocks, rowBlocksPerPart, colBlocksPerPart)
  }

  /** Creates a new [[VGridPartitioner]] instance with the input suggested number of partitions. */
  def apply(rowBlocks: Int, colBlocks: Int, suggestedNumPartitions: Int): VGridPartitioner = {
    require(suggestedNumPartitions > 0)
    val scale = 1.0 / math.sqrt(suggestedNumPartitions)
    val rowBlocksPerPart = math.round(math.max(scale * rowBlocks, 1.0)).toInt
    val colBlocksPerPart = math.round(math.max(scale * colBlocks, 1.0)).toInt
    new VGridPartitioner(rowBlocks, colBlocks, rowBlocksPerPart, colBlocksPerPart)
  }
}
