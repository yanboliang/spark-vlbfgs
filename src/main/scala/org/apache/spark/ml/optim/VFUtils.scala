package org.apache.spark.ml.optim

import org.apache.spark.Partitioner

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg._
import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD}
import org.apache.spark.util.Utils

import scala.reflect.ClassTag

class CustomPartitionCoalescer(partitionNum: Int, mapFun: (Int) => Int)
  extends PartitionCoalescer with Serializable{

  override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
    val parentRDDPartNum = parent.partitions.length
    val partitionGroups = Array.tabulate[PartitionGroup](partitionNum){idx => new PartitionGroup()}
    var i = 0
    while (i < parentRDDPartNum) {
      partitionGroups(mapFun(i)).partitions.append(parent.partitions(i))
      i += 1
    }
    partitionGroups
  }
}

object VFUtils {

  def customCoalesceRDD[T: ClassTag](rdd: RDD[T], partitionNum: Int)(f: (Int) => Int): RDD[T] = {
    val partitionCoalescer = new CustomPartitionCoalescer(partitionNum, f)
    rdd.coalesce(partitionNum, false, Some(partitionCoalescer))
  }

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
