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

package org.apache.spark.ml.util

import java.util.concurrent.{Callable, ExecutorService, Future}

import scala.collection.mutable.HashMap
import org.apache.spark.Partitioner

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils

import scala.reflect.ClassTag


private[ml] object VUtils {

  def kvRDDToDV(
      rdd: RDD[(Int, Vector)],
      sizePerPart: Int,
      nParts: Int,
      nSize: Long): DistributedVector = {
    new DistributedVector(
      rdd.partitionBy(new DistributedVectorPartitioner(nParts)).map(_._2),
      sizePerPart, nParts, nSize)
  }

  // Get number of blocks in row or column direction
  def getNumBlocks(elementsPerBlock: Int, numElements: Long): Int = {
    val numBlocks = (numElements - 1) / elementsPerBlock + 1
    require(numBlocks < Int.MaxValue)
    numBlocks.toInt
  }

  def splitArrIntoDV(
      sc: SparkContext,
      arr: Array[Double],
      partSize: Int,
      partNum: Int): DistributedVector = {
    var i = 0
    val splitArr = new Array[Array[Double]](partNum)
    val lastSplitSize = arr.length - partSize * (partNum - 1)
    while (i < partNum) {
      if (i < partNum - 1) splitArr(i) = arr.slice(partSize * i, partSize * i + partSize)
      else splitArr(i) = arr.slice(partSize * i, arr.length)
      i += 1
    }
    val rdd = sc.parallelize(splitArr.zipWithIndex.map(x => (x._2, Vectors.dense(x._1))), partNum)
    kvRDDToDV(rdd, partSize, partNum, arr.length)
  }

  def splitSparseVector(sv: SparseVector, colsPerBlock: Int): Array[SparseVector] = {
    val totalSize = sv.size
    val colBlocks = getNumBlocks(colsPerBlock, totalSize)

    val indicesArr = Array.fill(colBlocks)(new ArrayBuffer[Int])
    val valuesArr = Array.fill(colBlocks)(new ArrayBuffer[Double])

    sv.foreachActive { case (index: Int, value: Double) =>
      indicesArr(index / colsPerBlock) += (index % colsPerBlock)
      valuesArr(index / colsPerBlock) += value
    }

    val result = new Array[SparseVector](colBlocks)
    var i = 0
    while (i < colBlocks) {
      val size = if (i == colBlocks - 1) {
        (totalSize - 1) % colsPerBlock + 1
      } else {
        colsPerBlock
      }
      result(i) = new SparseVector(size, indicesArr(i).toArray, valuesArr(i).toArray)
      i = i + 1
    }
    result
  }

  def computePartitionSize(rdd: RDD[_]): Array[Long] = {
    rdd.mapPartitionsWithIndex { case (index, iter) =>
      List((index, Utils.getIteratorSize(iter))).toIterator
    }.collect().sortWith(_._1 < _._1).map(_._2)
  }

  // Helper method for concurrent execute tasks
  def concurrentExecuteTasks[T](
      tasks: Seq[T],
      executorPool: ExecutorService,
      taskFn: T => Unit) = {
    val futureArr = new Array[Future[Object]](tasks.size)
    var i = 0
    while (i < tasks.size) {
      val task = tasks(i)
      val callable = new Callable[Object] {
        override def call(): Object = {
          taskFn(task)
          null
        }
      }
      futureArr(i) = executorPool.submit(callable)
      i += 1
    }
    // wait all tasks finish
    futureArr.map(_.get())
  }

  def zipRDDWithIndex[T: ClassTag](
      partitionSizes: Array[Long],
      rdd: RDD[T]): RDD[(Long, T)] = {
    val startIndices = partitionSizes.scanLeft(0L)(_ + _)
    val bcStartIndices = rdd.sparkContext.broadcast(startIndices)

    rdd.mapPartitionsWithIndex { case (partIndex, iter) =>
      val startIndex = bcStartIndices.value(partIndex)
      iter.zipWithIndex.map { x =>
        (startIndex + x._2, x._1)
      }
    }
  }

  def vertcatSparseVectorIntoMatrix(vecs: Array[SparseVector]): SparseMatrix = {
    val numCols = vecs(0).size
    val numRows = vecs.length
    var i = 0
    val entries = new ArrayBuffer[(Int, Int, Double)]()

    if (numRows < numCols) {
      // generating sparse matrix with CSR format
      while (i < vecs.length) {
        val rowIdx = i
        vecs(i).foreachActive { case (colIdx: Int, v: Double) =>
          entries.append((colIdx, rowIdx, v))
        }
        i = i + 1
      }
      SparseMatrix.fromCOO(numCols, numRows, entries).transpose
    } else {
      // generating sparse matrix with CSC format
      while (i < vecs.length) {
        val rowIdx = i
        vecs(i).foreachActive { case (colIdx: Int, v: Double) =>
          entries.append((rowIdx, colIdx, v))
        }
        i = i + 1
      }
      SparseMatrix.fromCOO(numRows, numCols, entries)
    }
  }

  def zipRDDWithPartitionIDAndCollect[T: ClassTag](rdd: RDD[T]): Array[(Int, T)] = {
    rdd.mapPartitionsWithIndex{
      case (pid: Int, iter: Iterator[T]) =>
        iter.map((pid, _))
    }.collect()
  }


  def blockMatrixHorzZipVec[T: ClassTag](
      blockMatrixRDD: RDD[((Int, Int), SparseMatrix)],
      dvec: DistributedVector,
      gridPartitioner: GridPartitionerV2,
      f: (((Int, Int), SparseMatrix, Vector) => T)
  ): RDD[((Int, Int), T)] = {
    import org.apache.spark.rdd.VRDDFunctions._
    require(gridPartitioner.cols == dvec.numBlocks)
    blockMatrixRDD.mapJoinPartition(dvec.blocks)(
      (pid: Int) => {  // pid is the partition ID of blockMatrix RDD
        val colPartId = gridPartitioner.colPartId(pid)
        val startIdx = colPartId * gridPartitioner.colsPerPart
        var endIdx = startIdx + gridPartitioner.colsPerPart
        if (endIdx > gridPartitioner.cols) endIdx = gridPartitioner.cols
        (startIdx until endIdx).toArray // The corresponding partition ID of dvec
      },
      (pid: Int, mIter: Iterator[((Int, Int), SparseMatrix)], vIters: Array[(Int, Iterator[Vector])]) => {
        val vMap = new HashMap[Int, Vector]
        vIters.foreach {
          case(colId: Int, iter: Iterator[Vector]) =>
            val v = iter.next()
            assert(!iter.hasNext)
            vMap += (colId -> v)
        }
        mIter.map { case ((rowBlockIdx: Int, colBlockIdx: Int), sm: SparseMatrix) =>
          val vecPart = vMap(colBlockIdx)
          ((rowBlockIdx, colBlockIdx), f((rowBlockIdx, colBlockIdx), sm, vecPart))
        }
      }
    )
  }

  def blockMatrixVertZipVec[T: ClassTag](
      blockMatrixRDD: RDD[((Int, Int), SparseMatrix)],
      dvec: DistributedVector,
      gridPartitioner: GridPartitionerV2,
      f: (((Int, Int), SparseMatrix, Vector) => T)
  ): RDD[((Int, Int), T)] = {
    import org.apache.spark.rdd.VRDDFunctions._
    require(gridPartitioner.rows == dvec.numBlocks)
    blockMatrixRDD.mapJoinPartition(dvec.blocks)(
      (pid: Int) => {
        val rowPartId = gridPartitioner.rowPartId(pid)
        val startIdx = rowPartId * gridPartitioner.rowsPerPart
        var endIdx = startIdx + gridPartitioner.rowsPerPart
        if (endIdx > gridPartitioner.rows) endIdx = gridPartitioner.rows
        (startIdx until endIdx).toArray
      },
      (pid: Int, mIter: Iterator[((Int, Int), SparseMatrix)], vIters: Array[(Int, Iterator[Vector])]) => {
        val vMap = new HashMap[Int, Vector]
        vIters.foreach {
          case(rowId: Int, iter: Iterator[Vector]) =>
            val v = iter.next()
            assert(!iter.hasNext)
            vMap += (rowId -> v)
        }
        mIter.map { case ((rowBlockIdx: Int, colBlockIdx: Int), sm: SparseMatrix) =>
          val horzPart = vMap(rowBlockIdx)
          ((rowBlockIdx, colBlockIdx), f((rowBlockIdx, colBlockIdx), sm, horzPart))
        }
      }
    )
  }
}

class OneDimGridPartitioner(val total: Long, val partSize: Int) extends Partitioner {

  require(total > partSize && partSize > 0)

  val partNum = {
    val _partNum = (total - 1) / partSize + 1
    require(_partNum > 0 && _partNum <= Int.MaxValue)
    _partNum.toInt
  }

  override def getPartition(key: Any): Int = (key.asInstanceOf[Long] / partSize).toInt

  override def numPartitions: Int = partNum
}

private[spark] class GridPartitionerV2(
    val rows: Int,
    val cols: Int,
    val rowsPerPart: Int,
    val colsPerPart: Int) extends Partitioner {

  require(rows > 0)
  require(cols > 0)
  require(rowsPerPart > 0)
  require(colsPerPart > 0)

  val rowPartitions = math.ceil(rows * 1.0 / rowsPerPart).toInt
  val colPartitions = math.ceil(cols * 1.0 / colsPerPart).toInt

  override val numPartitions: Int = rowPartitions * colPartitions

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
  private def getPartitionId(i: Int, j: Int): Int = {
    require(0 <= i && i < rows, s"Row index $i out of range [0, $rows).")
    require(0 <= j && j < cols, s"Column index $j out of range [0, $cols).")
    i / rowsPerPart + j / colsPerPart * rowPartitions
  }

  def rowPartId(partId: Int) = partId % rowPartitions
  def colPartId(partId: Int) = partId / rowPartitions

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: GridPartitionerV2 =>
        (this.rows == r.rows) && (this.cols == r.cols) &&
          (this.rowsPerPart == r.rowsPerPart) && (this.colsPerPart == r.colsPerPart)
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(
      rows: java.lang.Integer,
      cols: java.lang.Integer,
      rowsPerPart: java.lang.Integer,
      colsPerPart: java.lang.Integer)
  }
}

private[spark] object GridPartitionerV2 {

  /** Creates a new [[GridPartitionerV2]] instance. */
  def apply(rows: Int, cols: Int, rowsPerPart: Int, colsPerPart: Int): GridPartitionerV2 = {
    new GridPartitionerV2(rows, cols, rowsPerPart, colsPerPart)
  }

  /** Creates a new [[GridPartitionerV2]] instance with the input suggested number of partitions. */
  def apply(rows: Int, cols: Int, suggestedNumPartitions: Int): GridPartitionerV2 = {
    require(suggestedNumPartitions > 0)
    val scale = 1.0 / math.sqrt(suggestedNumPartitions)
    val rowsPerPart = math.round(math.max(scale * rows, 1.0)).toInt
    val colsPerPart = math.round(math.max(scale * cols, 1.0)).toInt
    new GridPartitionerV2(rows, cols, rowsPerPart, colsPerPart)
  }
}

private[spark] class VectorSummarizer extends Serializable {

  private var sum: Array[Double] = null

  def add(v: Vector): VectorSummarizer = {
    if (sum == null) {
      sum = v.toDense.toArray
    } else {
      val localSum = sum
      v.foreachActive { (index: Int, value: Double) =>
        localSum(index) += value
      }
    }
    this
  }

  def merge(s: VectorSummarizer): VectorSummarizer = {
    val sum2 = s.sum
    if (sum == null) {
      sum = sum2
    } else {
      require(sum.length == sum2.length)
      var i = 0
      while (i < sum.length) {
        sum(i) += sum2(i)
        i += 1
      }
    }
    this
  }

  def toDenseVector: DenseVector = {
    Vectors.dense(sum).toDense
  }

  def toArray: Array[Double] = sum

}


