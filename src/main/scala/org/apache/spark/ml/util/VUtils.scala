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

import org.apache.spark.ml.feature.Instance
import org.apache.spark.SparkContext

import scala.collection.mutable.{ArrayBuffer, ArrayBuilder}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.linalg.distributed.{DistributedVector, DistributedVectorPartitioner, VBlockMatrix, VGridPartitioner}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.OptimMultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD
import org.apache.spark.util.Utils

import scala.reflect.ClassTag


private[spark] object VUtils {

  def KVRDDToDV(
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
      sizePerPart: Int,
      numPartitions: Int): DistributedVector = {
    var i = 0
    val splitArr = new Array[Array[Double]](numPartitions)
    while (i < numPartitions) {
      if (i < numPartitions - 1) splitArr(i) = arr.slice(sizePerPart * i, sizePerPart * i + sizePerPart)
      else splitArr(i) = arr.slice(sizePerPart * i, arr.length)
      i += 1
    }
    val rdd = sc.parallelize(splitArr.zipWithIndex.map(x => (x._2, Vectors.dense(x._1))), numPartitions)
    KVRDDToDV(rdd, sizePerPart, numPartitions, arr.length)
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

  def vertcatSparseVectorIntoMatrix(values: Array[SparseVector]): SparseMatrix = {
    val numCols = values(0).size
    val numRows = values.length
    var i = 0
    val entries = new ArrayBuffer[(Int, Int, Double)]()

    if (numRows < numCols) {
      // generating sparse matrix with CSR format
      while (i < values.length) {
        val rowIdx = i
        values(i).foreachActive { case (colIdx: Int, v: Double) =>
          entries.append((colIdx, rowIdx, v))
        }
        i = i + 1
      }
      SparseMatrix.fromCOO(numCols, numRows, entries).transpose
    } else {
      // generating sparse matrix with CSC format
      while (i < values.length) {
        val rowIdx = i
        values(i).foreachActive { case (colIdx: Int, v: Double) =>
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

  // This method is only used for testing
  def printUsedMemory(tag: String): Unit = {
    if (System.getProperty("GCTest.doGCWhenPrintMem", "false").toBoolean) {
      System.gc()
    }
    System.err.println(s"thread: ${Thread.currentThread().getId}, tag: ${tag}, mem: ${
      Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()
    }")
  }

  // TODO: add test
  // Don't modify it before test added.
  def packLabelsAndWeights(
      partitionSizes: Array[Long],
      labelsAndWeights: RDD[(Double, Double)],
      rowsPerBlock: Int,
      rowBlocks: Int): RDD[(Array[Double], Array[Double])] = {
    VUtils.zipRDDWithIndex(partitionSizes, labelsAndWeights)
      .map { case (rowIdx: Long, (label: Double, weight: Double)) =>
        val rowBlockIdx = (rowIdx / rowsPerBlock).toInt
        val inBlockIdx = (rowIdx % rowsPerBlock).toInt
        (rowBlockIdx, (inBlockIdx, label, weight))
      }
      .groupByKey(new DistributedVectorPartitioner(rowBlocks))
      .map { case (rowBlockIdx: Int, iter: Iterable[(Int, Double, Double)]) =>
        val tupleArr = iter.toArray.sortWith(_._1 < _._1)
        val labelArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._2)
        val weightArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._3)
        (labelArr, weightArr)
      }
  }

  // TODO: add test
  // Don't modify it before test added.
  def genRawFeatureBlocks(
      partitionSizes: Array[Long],
      instances: RDD[Instance],
      numFeatures: Long,
      numInstances: Long,
      colBlocks: Int,
      rowBlocks: Int,
      colsPerBlock: Int,
      rowsPerBlock: Int,
      colPartitions: Int,
      rowPartitions: Int,
      compressFeatureMatrix: Boolean,
      generatingFeatureMatrixBuffer: Int,
      rowPartitionSplitNumOnGeneratingFeatureMatrix: Int): VBlockMatrix = {

    var localRowPartitions = rowPartitions
    var localColPartitions = colPartitions
    if (localRowPartitions > rowBlocks) localRowPartitions = rowBlocks
    if (localColPartitions > colBlocks) localColPartitions = colBlocks

    val gridPartitioner = new VGridPartitioner(
      rowBlocks,
      colBlocks,
      rowBlocks / localRowPartitions,
      colBlocks / localColPartitions
    )
    val rowSplitedGridPartitioner = gridPartitioner.getRowSplitedGridPartitioner(
      rowPartitionSplitNumOnGeneratingFeatureMatrix
    )

    val lastBlockColSize = (numFeatures - (colBlocks - 1) * colsPerBlock).toInt
    val lastBlockRowSize = (numInstances - (rowBlocks - 1) * rowsPerBlock).toInt

    // Here we first count the exact number of active features in each block. Then
    // in the following step, we can ask for determinate size of row indices array,
    // column indices array and value array. This can help to reduce some heap memory
    // issues, like GC and OOM. The idea is trading time for space in at an acceptable
    // level.
    val activeInBlock = VUtils.zipRDDWithIndex(partitionSizes, instances)
      .mapPartitions { iter: Iterator[(Long, Instance)] =>
        new Iterator[Array[((Int, Int), Int)]] {

          override def hasNext: Boolean = iter.hasNext

          override def next(): Array[((Int, Int), Int)] = {
            val buffArr = Array.fill(colBlocks)(0)

            var shouldBreak = false
            var blockRowIndex = -1
            while (iter.hasNext && !shouldBreak) {
              val (rowIndex: Long, instance: Instance) = iter.next()
              if (blockRowIndex == -1) {
                blockRowIndex = (rowIndex / rowsPerBlock).toInt
              }
              val inBlockRowIndex = (rowIndex % rowsPerBlock).toInt
              instance.features.foreachActive { (colIndex: Int, value: Double) =>
                val blockColIndex = colIndex / colsPerBlock
                buffArr(blockColIndex) += 1
              }
              if (inBlockRowIndex == rowsPerBlock - 1 ||
                inBlockRowIndex % generatingFeatureMatrixBuffer ==
                  generatingFeatureMatrixBuffer - 1
              ) {
                shouldBreak = true
              }
            }
            buffArr.zipWithIndex.map { case (numActive: Int, blockColIndex: Int) =>
              ((blockRowIndex, blockColIndex), numActive)
            }
          }
        }
      }

    var rawFeatureBlocks: RDD[((Int, Int), VMatrix)] =
      VUtils.zipRDDWithIndex(partitionSizes, instances).zipPartitions(activeInBlock) { (it1, it2) =>
        // Zip the raw instance rdd and activeInBlock rdd. We place the active feature information
        // in front of instance data, and mark boundary with (-1, null) tuple.
        //
        //  [ active feature information in each block  ---> boundary ---> instances data ]
        //  [ ((1, 3), 10),  ((1, 4), 8)  ... ((3, 9), 17),   boundary, instance, ... instance] partition 0
        //  [ ((5, 6), 3),   ((5, 7), 12) ... ((7, 12), 9),   boundary, instance, ... instance] partition 1
        //  [ ((9, 3), 1),   ((9, 4), 18) ... ((11, 9), 7),   boundary, instance, ... instance] partition 2
        //                                    ...
        //  [ ((11, 7), 2),  ((11, 8), 5) ... ((11, 13), 14), boundary, instance, ... instance] partition N
        new Iterator[(Int, Any)] {
          var shouldSkip = false
          override def hasNext: Boolean = it2.hasNext || it1.hasNext

          override def next(): (Int, Any) = {
            // mark data item with different types:
            // 1: active feature information in each block
            // -1: boundary
            // 2: raw instance data
            if (it2.hasNext) {
              (1, it2.next())
            } else if (!shouldSkip) {
              shouldSkip = true
              (-1, null)
            } else {
              (2, it1.next())
            }
          }
        }}.mapPartitions { iter: Iterator[(Int, Any)] =>
          // Pre-fetch the front active feature information in each partition
          val activeInBlock = new ArrayBuffer[Array[((Int, Int), Int)]]
          var shouldBreak = false
          while (iter.hasNext && !shouldBreak) {
            val a = iter.next()
            if (a._1 == 1) {
              activeInBlock += a._2.asInstanceOf[Array[((Int, Int), Int)]]
            } else if (a._1 == -1) {
              shouldBreak = true
            }
          }

          new Iterator[Array[((Int, Int), (Array[Int], Array[Int], Array[Double]))]] {

            override def hasNext: Boolean = iter.hasNext

            val idx = Array.fill(colBlocks)(0)
            var inBlockRowIndex = 0
            var blockColIndex = 0
            var inBlockColIndex = 0
            var n = 0
            var m = 0

            override def next(): Array[((Int, Int), (Array[Int], Array[Int], Array[Double]))] = {
              for(i <- 0 until colBlocks) {
                idx(i) = 0
              }
              val buffArr = Array.tabulate(colBlocks) { i =>
                // Give each array with a determinate size
                Tuple3(
                  new Array[Int](activeInBlock.apply(n)(i)._2),
                  new Array[Int](activeInBlock.apply(n)(i)._2),
                  new Array[Double](activeInBlock.apply(n)(i)._2))
              }
              n += 1

              shouldBreak = false
              var blockRowIndex = -1
              while (iter.hasNext && !shouldBreak) {
                val (rowIndex: Long, instance: Instance) = iter.next()._2.asInstanceOf[(Long, Instance)]
                if (blockRowIndex == -1) {
                  blockRowIndex = (rowIndex / rowsPerBlock).toInt
                }
                inBlockRowIndex = (rowIndex % rowsPerBlock).toInt
                instance.features.foreachActive { (colIndex: Int, value: Double) =>
                  blockColIndex = colIndex / colsPerBlock
                  inBlockColIndex = colIndex % colsPerBlock
                  m = idx(blockColIndex)
                  val COOBuffTuple = buffArr(blockColIndex)
                  COOBuffTuple._1(m) = inBlockRowIndex
                  COOBuffTuple._2(m) = inBlockColIndex
                  COOBuffTuple._3(m) = value
                  idx(blockColIndex) += 1
                }
                if (inBlockRowIndex == rowsPerBlock - 1 ||
                  inBlockRowIndex % generatingFeatureMatrixBuffer ==
                    generatingFeatureMatrixBuffer - 1
                ) {
                  shouldBreak = true
                }
              }
              var _blockColIndex = -1
              buffArr.map { case (tuple: (Array[Int], Array[Int],
                Array[Double])) =>
                _blockColIndex += 1
                ((blockRowIndex, _blockColIndex), (tuple._1, tuple._2, tuple._3))
              }
            }
          }.flatMap(_.toIterator)}
        .groupByKey(rowSplitedGridPartitioner)
        .map { case (coodinate: (Int, Int), cooList: Iterable[(Array[Int], Array[Int], Array[Double])]) =>
          val cooBuff = new ArrayBuffer[(Int, Int, Double)]
          cooList.foreach { case (rowIndices: Array[Int], colIndices: Array[Int], values: Array[Double]) =>
            var i = 0
            while (i < rowIndices.length) {
              cooBuff += Tuple3(rowIndices(i), colIndices(i), values(i))
              i += 1
            }
          }
          val numRows = if (coodinate._1 == rowBlocks - 1) lastBlockRowSize else rowsPerBlock
          val numCols = if (coodinate._2 == colBlocks - 1) lastBlockColSize else colsPerBlock
          (coodinate, VMatrices.COOEntries(numRows, numCols, cooBuff, compressFeatureMatrix))
        }
    if (rowPartitionSplitNumOnGeneratingFeatureMatrix > 1) {
      rawFeatureBlocks = rawFeatureBlocks.partitionBy(gridPartitioner)
    }

    new VBlockMatrix(
      rowsPerBlock, colsPerBlock, rawFeatureBlocks, gridPartitioner)
  }

  // TODO: add test
  // Don't modify it before test added.
  def genFeatureStd(
      rawFeatures: VBlockMatrix,
      weightDV: DistributedVector,
      colsPerBlock: Int,
      colBlocks: Int,
      numFeatures: Long) = {
    val summarizerSeqOp = (summarizer: OptimMultivariateOnlineSummarizer,
                           tuple: (VMatrix, Vector)) => {
      val (block, weight) = tuple
      block.rowIter.zip(weight.toArray.toIterator).foreach {
        case (rowVector: Vector, weight: Double) =>
          summarizer.add(rowVector, weight)
      }
      summarizer
    }
    val summarizerCombineOp = (s1: OptimMultivariateOnlineSummarizer,
                               s2: OptimMultivariateOnlineSummarizer) => s1.merge(s2)

    val featureStdSummarizerRDD = rawFeatures.verticalZipVector(weightDV) {
      (blockCoordinate: (Int, Int), block: VMatrix, weight: Vector) => (block, weight)
    }.map { case ((rowBlockIdx: Int, colBlockIdx: Int), tuple: (VMatrix, Vector)) =>
      (colBlockIdx, tuple)
    }.aggregateByKey(
      new OptimMultivariateOnlineSummarizer(
        OptimMultivariateOnlineSummarizer.varianceMask),
      new DistributedVectorPartitioner(colBlocks)
    )(summarizerSeqOp, summarizerCombineOp)

    val featureStdRDD: RDD[Vector] = featureStdSummarizerRDD.values.map { summarizer =>
      Vectors.dense(summarizer.variance.toArray.map(math.sqrt))
    }

    new DistributedVector(
      featureStdRDD, colsPerBlock, colBlocks, numFeatures)
  }

  // TODO: add test
  // Don't modify it before test added.
  def genFeatureBlocks(
      rawFeatures: VBlockMatrix,
      compressFeatureMatrix: Boolean,
      featuresStd: DistributedVector): VBlockMatrix = {
    rawFeatures.horizontalZipVector2(featuresStd) {
      (blockCoordinate: (Int, Int), block: VMatrix, partFeaturesStd: Vector) =>
        val partFeatureStdArr = partFeaturesStd.asInstanceOf[DenseVector].values
        var numActive = 0
        block.foreachActive { case (i: Int, j: Int, value: Double) =>
          if (partFeatureStdArr(j) != 0 && value != 0) {
            numActive += 1
          }
        }
        val rowIndices: Array[Int] = new Array[Int](numActive)
        val colIndices: Array[Int] = new Array[Int](numActive)
        val values: Array[Double] = new Array[Double](numActive)
        var n = 0
        block.foreachActive { case (i: Int, j: Int, value: Double) =>
          if (partFeatureStdArr(j) != 0 && value != 0) {
            rowIndices(n) = i
            colIndices(n) = j
            values(n) = value / partFeatureStdArr(j)
            n += 1
          }
        }
        VMatrices.COOArrays(block.numRows, block.numCols, rowIndices, colIndices,
          values, compressFeatureMatrix)
    }
  }

  // TODO: add test
  // Don't modify it before test added.
  def sparseVectorToColumnMajorSparseMatrix(
      numRows: Int, numCols: Int, sv: SparseVector): SparseMatrix = {
    val numEntries = sv.size

    val colPtrs = new Array[Int](numCols + 1)
    val rowIndices = ArrayBuilder.make[Int]
    rowIndices.sizeHint(numEntries)
    val values = ArrayBuilder.make[Double]
    values.sizeHint(numEntries)
    var nnz = 0
    var prevCol = 0
    var prevRow = -1
    var prevVal = 0.0

    val loopFun = (i: Int, j: Int, v: Double) => {
      if (v != 0) {
        if (i == prevRow && j == prevCol) {
          prevVal += v
        } else {
          if (prevVal != 0) {
            require(prevRow >= 0 && prevRow < numRows,
              s"Row index out of range [0, $numRows): $prevRow.")
            nnz += 1
            rowIndices += prevRow
            values += prevVal
          }
          prevRow = i
          prevVal = v
          while (prevCol < j) {
            colPtrs(prevCol + 1) = nnz
            prevCol += 1
          }
        }
      }
    }

    sv.foreachActive { (index: Int, v: Double) =>
      val i = index % numRows
      val j = index / numRows
      loopFun(i, j, v)
    }

    loopFun(numRows, numCols, 1.0)

    new SparseMatrix(numRows, numCols, colPtrs, rowIndices.result(), values.result())
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
      if (sum2 != null)
        sum = sum2.clone()
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
