package org.apache.spark.ml.optim

import scala.collection.mutable.HashMap

import org.apache.spark.Partitioner

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD}
import org.apache.spark.util.Utils

import scala.reflect.ClassTag


object VFUtils {

  def kvRDDToDV(rdd: RDD[(Int, Vector)], sizePerPart: Int, nParts: Int, nSize: Long)
    : DistributedVector = {
    new DistributedVector(
      rdd.partitionBy(new DistributedVectorPartitioner(nParts)).map(_._2),
      sizePerPart, nParts, nSize
    )
  }

  def getSplitPartNum(partSize: Int, totalSize: Long): Int = {
    val num = (totalSize - 1) / partSize + 1
    require(num < Int.MaxValue)
    num.toInt
  }

  def splitArrIntoDV(sc: SparkContext, arr: Array[Double], partSize: Int, partNum: Int) = {
    var i = 0;
    val splitArr = new Array[Array[Double]](partNum)
    val lastSplitSize = arr.length - partSize * (partNum - 1)
    while (i < partNum) {
      if (i < partNum - 1) splitArr(i) = arr.slice(partSize * i, partSize * i + partSize)
      else splitArr(i) = arr.slice(partSize * i, arr.length)
      i += 1
    }
    val rdd = sc.parallelize(splitArr.zipWithIndex.map(x => (x._2, Vectors.dense(x._1))))
    kvRDDToDV(rdd, partSize, partNum, arr.length)
  }

  def splitSparseVector(sv: SparseVector, vecPartSize: Int): Array[SparseVector] = {
    val totalSize = sv.size
    val partNum = getSplitPartNum(vecPartSize, totalSize)

    val indicesArr = Array.fill(partNum)(new ArrayBuffer[Int])
    val valuesArr = Array.fill(partNum)(new ArrayBuffer[Double])

    sv.foreachActive((index: Int, value: Double) => {
      indicesArr(index / vecPartSize) += (index % vecPartSize)
      valuesArr(index / vecPartSize) += value
    })

    val result = new Array[SparseVector](partNum)
    var i = 0
    while (i < partNum) {
      val size = if (i == partNum - 1) {
        ((totalSize - 1) % vecPartSize) + 1
      } else vecPartSize
      result(i) = new SparseVector(size, indicesArr(i).toArray, valuesArr(i).toArray)
      i = i + 1
    }
    result
  }

  def computePartitionStartIndices(rdd: RDD[_]): Array[Long] = {
    rdd.mapPartitionsWithIndex((index, iter) =>
      List((index, Utils.getIteratorSize(iter))).toIterator)
      .collect().sortWith(_._1 < _._1).map(_._2)
  }

  def zipRDDWithIndex[T: ClassTag](partitionSizeArray: Array[Long], rdd: RDD[T]): RDD[(Long, T)] = {
    val startIndices = partitionSizeArray.scanLeft(0L)(_ + _)
    val bcStartIndices = rdd.sparkContext.broadcast(startIndices)

    rdd.mapPartitionsWithIndex((partIndex, iter) => {
      val startIndex = bcStartIndices.value(partIndex)
      iter.zipWithIndex.map { x =>
        (startIndex + x._2, x._1)
      }
    })
  }

  def vertcatSparseVectorIntoCSRMatrix(vecs: Array[SparseVector]): SparseMatrix = {
    var i = 0;
    val entries = new ArrayBuffer[(Int, Int, Double)]()
    while (i < vecs.length) {
      val rowIdx = i
      vecs(i).foreachActive { case (colIdx: Int, v: Double) =>
        entries.append((colIdx, rowIdx, v))
      }
      i = i + 1
    }
    SparseMatrix.fromCOO(vecs(0).size, vecs.length, entries).transpose
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
      f: ((SparseMatrix, Vector) => T)
  ) = {
    import org.apache.spark.ml.optim.VFRDDFunctions._
    require(gridPartitioner.cols == dvec.nParts)
    blockMatrixRDD.mapJoinPartition(dvec.vecs)(
      (pid: Int) => {
        val colPartId = gridPartitioner.colPartId(pid)
        val startIdx = colPartId * gridPartitioner.colsPerPart
        var endIdx = startIdx + gridPartitioner.colsPerPart
        if (endIdx > gridPartitioner.cols) endIdx = gridPartitioner.cols
        (startIdx until endIdx).toArray
      },
      (pid: Int, mIter: Iterator[((Int, Int), SparseMatrix)], vIters: Array[(Int, Iterator[Vector])]) => {
        val vMap = new HashMap[Int, Vector]
        vIters.foreach {
          case(colId: Int, iter: Iterator[Vector]) =>
            val v = iter.next()
            assert(!iter.hasNext)
            vMap += (colId -> v)
        }
        mIter.map {
          block =>
            val vecPart = vMap(block._1._2)
            (block._1._1, f(block._2, vecPart))
        }
      }
    )
  }
  def blockMatrixVertZipVec[T: ClassTag](
      blockMatrixRDD: RDD[((Int, Int), SparseMatrix)],
      dvec: DistributedVector,
      gridPartitioner: GridPartitionerV2,
      f: (((SparseMatrix, Vector) => T))
  ) = {
    import org.apache.spark.ml.optim.VFRDDFunctions._
    require(gridPartitioner.rows == dvec.nParts)
    blockMatrixRDD.mapJoinPartition(dvec.vecs)(
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
        mIter.map {
          block =>
            val horzPart = vMap(block._1._1)
            (block._1._2, f(block._2, horzPart))
        }
      }
    )
  }


}

class OneDimGridPartitioner(val total: Long, val partSize: Int) extends Partitioner{
  require(total > partSize && partSize > 0)
  val partNum = {
    val _partNum = (total - 1) / partSize + 1
    require(_partNum > 0 && _partNum <= Int.MaxValue)
    _partNum.toInt
  }
  override def getPartition(key: Any): Int = (key.asInstanceOf[Long] / partSize).toInt
  override def numPartitions: Int = partNum
}

class GridPartitioner(val rows: Int, val cols: Int) extends Partitioner{
  override val numPartitions: Int = rows * cols
  override def getPartition(key: Any): Int = {
    key match {
      case (i: Int, j: Int) =>
        rows * j + i
      case _ =>
        throw new IllegalArgumentException(s"Unrecognized key: $key")
    }
  }
}

class GridPartitionerV2(
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
object GridPartitionerV2 {

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


