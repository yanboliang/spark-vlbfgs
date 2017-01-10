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

  def testBlockMatrixHorzZipVecFunc(
      rowBlocks: Int,
      colBlocks: Int,
      rowsPerBlock: Int,
      colsPerBlock: Int,
      shuffleRdd2: Boolean) = {
    val arrMatrix = Array.tabulate(rowBlocks * colBlocks) { idx =>
      val rowIdx = idx % rowBlocks
      val colIdx = idx / rowBlocks
      ((rowIdx, colIdx),
        SparseMatrix.fromCOO(1, 2, Array((0, 0, rowIdx.toDouble), (0, 1, colIdx.toDouble)))
        )
    }
    val gridPartitioner = VGridPartitioner(rowBlocks, colBlocks, rowsPerBlock, colsPerBlock)
    val blockMatrix = new VBlockMatrix(rowsPerBlock, colsPerBlock,
      sc.parallelize(arrMatrix).partitionBy(gridPartitioner), gridPartitioner)
    val arrVec = Array.tabulate(colBlocks)(idx => idx.toDouble)
    val dvec = splitArrIntoDV(sc, arrVec, 1, colBlocks)
    val f = (blockCoords: (Int, Int), sv: SparseMatrix, v: Vector) => {
      (sv(0, 0), sv(0, 1), v(0))
    }
    val res0 = blockMatrix.horizontalZipVec(dvec)(f)
      .map(x => (x._1._1, x._2))
    val res = res0.map { v =>
      (v._1, v._2._1.toInt, v._2._2.toInt, v._2._3.toInt)
    }.collect().map { v =>
      assert(v._1 == v._2 && v._3 == v._4)
      (v._2, v._3)
    }.sortBy(v => v._1 + v._2 * 1000)

    assert(res === Array.tabulate(rowBlocks * colBlocks) { idx =>
      val rowIdx = idx % rowBlocks
      val colIdx = idx / rowBlocks
      (rowIdx, colIdx)
    })
  }

  def testBlockMatrixVertZipVecFunc(
      rowBlocks: Int,
      colBlocks: Int,
      rowsPerBlock: Int,
      colsPerBlock: Int,
      shuffleRdd2: Boolean) = {
    val arrMatrix = Array.tabulate(rowBlocks * colBlocks) { idx =>
      val rowIdx = idx % rowBlocks
      val colIdx = idx / rowBlocks
      ((rowIdx, colIdx),
        SparseMatrix.fromCOO(1, 2, Array((0, 0, rowIdx.toDouble), (0, 1, colIdx.toDouble)))
        )
    }
    val gridPartitioner = VGridPartitioner(rowBlocks, colBlocks, rowsPerBlock, colsPerBlock)
    val blockMatrix = new VBlockMatrix(rowsPerBlock, colsPerBlock,
      sc.parallelize(arrMatrix).partitionBy(gridPartitioner), gridPartitioner)
    val arrVec = Array.tabulate(rowBlocks)(idx => idx.toDouble)
    val dvec = VUtils.splitArrIntoDV(sc, arrVec, 1, rowBlocks)
    val f = (blockCoords: (Int, Int), sv: SparseMatrix, v: Vector) => {
      (sv(0, 0), sv(0, 1), v(0))
    }
    val res0 = blockMatrix.verticalZipVec(dvec)(f)
      .map(x => (x._1._2, x._2))
    val res = res0.map { v =>
      (v._1, v._2._1.toInt, v._2._2.toInt, v._2._3.toInt)
    }.collect().map { v =>
      assert(v._1 == v._3 && v._2 == v._4)
      (v._2, v._3)
    }.sortBy(v => v._1 + v._2 * 1000)

    assert(res === Array.tabulate(rowBlocks * colBlocks) { idx =>
      val rowIdx = idx % rowBlocks
      val colIdx = idx / rowBlocks
      (rowIdx, colIdx)
    })
  }

  test("blockMatrixHorzZipVec") {
    testBlockMatrixHorzZipVecFunc(5, 4, 2, 3, false)
    testBlockMatrixHorzZipVecFunc(8, 6, 2, 3, false)
    testBlockMatrixHorzZipVecFunc(3, 5, 3, 5, false)
    testBlockMatrixHorzZipVecFunc(15, 4, 6, 1, false)
    testBlockMatrixHorzZipVecFunc(15, 3, 6, 2, false)

    testBlockMatrixHorzZipVecFunc(4, 5, 3, 2, false)
    testBlockMatrixHorzZipVecFunc(6, 8, 3, 2, false)
    testBlockMatrixHorzZipVecFunc(5, 3, 5, 3, false)
    testBlockMatrixHorzZipVecFunc(4, 15, 1, 6, false)
    testBlockMatrixHorzZipVecFunc(3, 15, 2, 6, false)

    testBlockMatrixHorzZipVecFunc(5, 4, 2, 3, true)
    testBlockMatrixHorzZipVecFunc(8, 6, 2, 3, true)
    testBlockMatrixHorzZipVecFunc(3, 5, 3, 5, true)
    testBlockMatrixHorzZipVecFunc(15, 4, 6, 1, true)
    testBlockMatrixHorzZipVecFunc(15, 3, 6, 2, true)

    testBlockMatrixHorzZipVecFunc(4, 5, 3, 2, true)
    testBlockMatrixHorzZipVecFunc(6, 8, 3, 2, true)
    testBlockMatrixHorzZipVecFunc(5, 3, 5, 3, true)
    testBlockMatrixHorzZipVecFunc(4, 15, 1, 6, true)
    testBlockMatrixHorzZipVecFunc(3, 15, 2, 6, true)
  }

  test("blockMatrixVertZipVec") {
    testBlockMatrixVertZipVecFunc(5, 4, 2, 3, false)
    testBlockMatrixVertZipVecFunc(8, 6, 2, 3, false)
    testBlockMatrixVertZipVecFunc(3, 5, 3, 5, false)
    testBlockMatrixVertZipVecFunc(15, 4, 6, 1, false)
    testBlockMatrixVertZipVecFunc(15, 3, 6, 2, false)

    testBlockMatrixVertZipVecFunc(4, 5, 3, 2, false)
    testBlockMatrixVertZipVecFunc(6, 8, 3, 2, false)
    testBlockMatrixVertZipVecFunc(5, 3, 5, 3, false)
    testBlockMatrixVertZipVecFunc(4, 15, 1, 6, false)
    testBlockMatrixVertZipVecFunc(3, 15, 2, 6, false)

    testBlockMatrixVertZipVecFunc(5, 4, 2, 3, true)
    testBlockMatrixVertZipVecFunc(8, 6, 2, 3, true)
    testBlockMatrixVertZipVecFunc(3, 5, 3, 5, true)
    testBlockMatrixVertZipVecFunc(15, 4, 6, 1, true)
    testBlockMatrixVertZipVecFunc(15, 3, 6, 2, true)

    testBlockMatrixVertZipVecFunc(4, 5, 3, 2, true)
    testBlockMatrixVertZipVecFunc(6, 8, 3, 2, true)
    testBlockMatrixVertZipVecFunc(5, 3, 5, 3, true)
    testBlockMatrixVertZipVecFunc(4, 15, 1, 6, true)
    testBlockMatrixVertZipVecFunc(3, 15, 2, 6, true)
  }
}
