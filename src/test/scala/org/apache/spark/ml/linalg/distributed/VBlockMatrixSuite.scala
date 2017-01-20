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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.{SparseMatrix, Vector}
import org.apache.spark.ml.util.VUtils
import org.apache.spark.ml.util.VUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext

class VBlockMatrixSuite extends SparkFunSuite with MLlibTestSparkContext {

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  def testHorizontalZipVector(
      rowBlocks: Int,
      colBlocks: Int,
      rowBlocksPerPart: Int,
      colBlocksPerPart: Int,
      shuffleRdd2: Boolean) = {
    val blocks = Array.tabulate(rowBlocks * colBlocks) { idx =>
      val rowBlockIdx = idx % rowBlocks
      val colBlockIdx = idx / rowBlocks
      ((rowBlockIdx, colBlockIdx),
        SparseMatrix.fromCOO(1, 2, Array((0, 0, rowBlockIdx.toDouble), (0, 1, colBlockIdx.toDouble))))
    }
    val gridPartitioner = VGridPartitioner(rowBlocks, colBlocks, rowBlocksPerPart, colBlocksPerPart)
    val blockMatrix = new VBlockMatrix(
      1, 2, sc.parallelize(blocks).partitionBy(gridPartitioner), gridPartitioner)
    val arrayData = Array.tabulate(colBlocks)(idx => idx.toDouble)
    val dVector = splitArrIntoDV(sc, arrayData, 1, colBlocks)
    val f = (blockCoordinate: (Int, Int), block: SparseMatrix, v: Vector) => {
      // This is the triple used to be checked below.
      // (rowBlockIdx, colBlockIdx, partVectorIdx)
      (block(0, 0), block(0, 1), v(0))
    }
    val res = blockMatrix.horizontalZipVector(dVector)(f).map {
      case (blockCoordinate: (Int, Int), triple: (Double, Double, Double)) =>
        (blockCoordinate._1, triple._1.toInt, triple._2.toInt, triple._3.toInt)
    }.collect().map { v =>
      assert(v._1 == v._2 && v._3 == v._4)
      (v._2, v._3)
    }.toSet

    assert(res === Array.tabulate(rowBlocks * colBlocks) { idx =>
      val rowIdx = idx % rowBlocks
      val colIdx = idx / rowBlocks
      (rowIdx, colIdx)
    }.toSet)
  }

  def testVerticalZipVector(
      rowBlocks: Int,
      colBlocks: Int,
      rowBlocksPerPart: Int,
      colBlocksPerPart: Int,
      shuffleRdd2: Boolean) = {
    val blocks = Array.tabulate(rowBlocks * colBlocks) { idx =>
      val rowBlockIdx = idx % rowBlocks
      val colBlockIdx = idx / rowBlocks
      ((rowBlockIdx, colBlockIdx),
        SparseMatrix.fromCOO(1, 2, Array((0, 0, rowBlockIdx.toDouble), (0, 1, colBlockIdx.toDouble))))
    }
    val gridPartitioner = VGridPartitioner(rowBlocks, colBlocks, rowBlocksPerPart, colBlocksPerPart)
    val blockMatrix = new VBlockMatrix(
      1, 2, sc.parallelize(blocks).partitionBy(gridPartitioner), gridPartitioner)
    val arrayData = Array.tabulate(rowBlocks)(idx => idx.toDouble)
    val dVector = VUtils.splitArrIntoDV(sc, arrayData, 1, rowBlocks)
    val f = (blockCoordinate: (Int, Int), block: SparseMatrix, v: Vector) => {
      (block(0, 0), block(0, 1), v(0))
    }
    val res = blockMatrix.verticalZipVector(dVector)(f).map {
      case (blockCoordinate: (Int, Int), triple: (Double, Double, Double)) =>
        (blockCoordinate._2, triple._1.toInt, triple._2.toInt, triple._3.toInt)
    }.collect().map { v =>
      assert(v._1 == v._3 && v._2 == v._4)
      (v._2, v._3)
    }.toSet

    assert(res === Array.tabulate(rowBlocks * colBlocks) { idx =>
      val rowIdx = idx % rowBlocks
      val colIdx = idx / rowBlocks
      (rowIdx, colIdx)
    }.toSet)
  }

  test("horizontalZipVector") {
    testHorizontalZipVector(5, 4, 2, 3, false)
    testHorizontalZipVector(8, 6, 2, 3, false)
    testHorizontalZipVector(3, 5, 3, 5, false)
    testHorizontalZipVector(15, 4, 6, 1, false)
    testHorizontalZipVector(15, 3, 6, 2, false)

    testHorizontalZipVector(4, 5, 3, 2, false)
    testHorizontalZipVector(6, 8, 3, 2, false)
    testHorizontalZipVector(5, 3, 5, 3, false)
    testHorizontalZipVector(4, 15, 1, 6, false)
    testHorizontalZipVector(3, 15, 2, 6, false)

    testHorizontalZipVector(5, 4, 2, 3, true)
    testHorizontalZipVector(8, 6, 2, 3, true)
    testHorizontalZipVector(3, 5, 3, 5, true)
    testHorizontalZipVector(15, 4, 6, 1, true)
    testHorizontalZipVector(15, 3, 6, 2, true)

    testHorizontalZipVector(4, 5, 3, 2, true)
    testHorizontalZipVector(6, 8, 3, 2, true)
    testHorizontalZipVector(5, 3, 5, 3, true)
    testHorizontalZipVector(4, 15, 1, 6, true)
    testHorizontalZipVector(3, 15, 2, 6, true)
  }

  test("verticalZipVector") {
    testVerticalZipVector(5, 4, 2, 3, false)
    testVerticalZipVector(8, 6, 2, 3, false)
    testVerticalZipVector(3, 5, 3, 5, false)
    testVerticalZipVector(15, 4, 6, 1, false)
    testVerticalZipVector(15, 3, 6, 2, false)

    testVerticalZipVector(4, 5, 3, 2, false)
    testVerticalZipVector(6, 8, 3, 2, false)
    testVerticalZipVector(5, 3, 5, 3, false)
    testVerticalZipVector(4, 15, 1, 6, false)
    testVerticalZipVector(3, 15, 2, 6, false)

    testVerticalZipVector(5, 4, 2, 3, true)
    testVerticalZipVector(8, 6, 2, 3, true)
    testVerticalZipVector(3, 5, 3, 5, true)
    testVerticalZipVector(15, 4, 6, 1, true)
    testVerticalZipVector(15, 3, 6, 2, true)

    testVerticalZipVector(4, 5, 3, 2, true)
    testVerticalZipVector(6, 8, 3, 2, true)
    testVerticalZipVector(5, 3, 5, 3, true)
    testVerticalZipVector(4, 15, 1, 6, true)
    testVerticalZipVector(3, 15, 2, 6, true)
  }
}
