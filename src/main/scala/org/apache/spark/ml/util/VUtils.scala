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
import org.apache.spark.ml.linalg.distributed.{DistributedVector, DistributedVectorPartitioner}
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

  /**
   * Generate a zipWithIndex iterator, avoid index value overflowing problem
   * in scala's zipWithIndex
   */
  def getIteratorZipWithIndex[T](iterator: Iterator[T], startIndex: Long): Iterator[(Long, T)] = {
    new Iterator[(Long, T)] {
      require(startIndex >= 0, "startIndex should be >= 0.")
      var index: Long = startIndex - 1L
      def hasNext: Boolean = iterator.hasNext
      def next(): (Long, T) = {
        index += 1L
        (index, iterator.next())
      }
    }
  }

  def zipRDDWithIndex[T: ClassTag](
      partitionSizes: Array[Long],
      rdd: RDD[T]): RDD[(Long, T)] = {
    val startIndices = partitionSizes.scanLeft(0L)(_ + _)
    val bcStartIndices = rdd.sparkContext.broadcast(startIndices)

    rdd.mapPartitionsWithIndex { case (partIndex, iter) =>
      val startIndex = bcStartIndices.value(partIndex)
      VUtils.getIteratorZipWithIndex(iter, startIndex)
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
