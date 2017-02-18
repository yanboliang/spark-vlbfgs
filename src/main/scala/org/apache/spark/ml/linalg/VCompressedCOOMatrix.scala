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

package org.apache.spark.ml.linalg

/**
 * Compressed version for `VCOOMatrix`
 * Each entry of COO list use 24 bit integer pair as the coodinate,
 * and value use `float` instead of `double`
 */

class VCompressedCOOMatrix(
    override val numRows: Int,
    override val numCols: Int,
    private val rowIndices: Array[Byte],
    private val colIndices: Array[Byte],
    private val values: Array[Float]) extends VMatrix {

  assert(numRows >= 0 && numRows < (1 << 24))
  assert(numCols >= 0 && numCols < (1 << 24))

  private def get24BitInt(arr: Array[Byte], index: Int): Int = {
    val b1 = arr(index * 3) & 255
    val b2 = arr(index * 3 + 1) & 255
    val b3 = arr(index * 3 + 2) & 255

    val res = (b1 | (b2 << 8) | (b3 << 16))
    res
  }

  override def rowIter: Iterator[Vector] = {
    val zeroRow = new SparseVector(numCols, new Array[Int](0), new Array[Double](0))

    if (values.length == 0) {
      Iterator.continually(zeroRow).take(numRows)
    } else {
      var prevRowIndex = get24BitInt(rowIndices, 0)
      var prevRowIndexPtr = 0
      val iter = (rowIndices.toIterator.grouped(3).map(x => get24BitInt(x.toArray, 0))
        ++ Iterator.single(numRows)).zipWithIndex.flatMap {
        case (rowIndex: Int, rowIndexPtr: Int) =>
          if (rowIndex == prevRowIndex) {
            Iterator.empty
          } else {
            val headElem = new SparseVector(numCols,
              colIndices.slice(prevRowIndexPtr * 3, rowIndexPtr * 3)
                .grouped(3).map(x => get24BitInt(x, 0)).toArray,
              values.slice(prevRowIndexPtr, rowIndexPtr).map(_.toDouble)
            )
            assert(rowIndex > prevRowIndex)

            val emptyRowIter = Iterator.continually(zeroRow).take(rowIndex - prevRowIndex - 1)

            prevRowIndex = rowIndex
            prevRowIndexPtr = rowIndexPtr

            Iterator.single(headElem) ++ emptyRowIter
          }
      }
      Iterator.continually(zeroRow).take(prevRowIndex) ++ iter
    }
  }

  override def numActives: Int = values.length

  override private[spark] def foreachActive(f: (Int, Int, Double) => Unit): Unit = {
    var i = 0
    while (i < values.length) {
      f(get24BitInt(rowIndices, i), get24BitInt(colIndices, i), values(i))
      i += 1
    }
  }
}
