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

import java.util.Arrays

/**
 * Row major COO format matrix.
 * It will use less storage space than `ml.linalg.SparseMatrix`,
 * when the matrix is extremely sparse (for example, each row contains less than one
 * nonzero element in average)
 */
class VCOOMatrix(
    override val numRows: Int,
    override val numCols: Int,
    private val rowIndices: Array[Int],
    private val colIndices: Array[Int],
    private val values: Array[Double]) extends VMatrix {

  /**
   * Returns an iterator of row vectors.
   */
  override def rowIter: Iterator[Vector] = {
    val zeroRow = new SparseVector(numCols, new Array[Int](0), new Array[Double](0))

    if (values.length == 0) {
      Iterator.continually(zeroRow).take(numRows)
    } else {
      var prevRowIndex = rowIndices(0)
      var prevRowIndexPtr = 0
      val iter = (rowIndices.toIterator ++ Iterator.single(numRows)).zipWithIndex.flatMap {
        case (rowIndex: Int, rowIndexPtr: Int) =>
          if (rowIndex == prevRowIndex) {
            Iterator.empty
          } else {
            val headElem = new SparseVector(numCols,
              colIndices.slice(prevRowIndexPtr, rowIndexPtr),
              values.slice(prevRowIndexPtr, rowIndexPtr)
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

  /**
   * Applies a function `f` to all the active elements of dense and sparse matrix. The ordering
   * of the elements are not defined.
   *
   * @param f the function takes three parameters where the first two parameters are the row
   *          and column indices respectively with the type `Int`, and the final parameter is the
   *          corresponding value in the matrix with type `Double`.
   */
  private[spark] override def foreachActive(f: (Int, Int, Double) => Unit) = {
    var i = 0
    while (i < values.length) {
      f(rowIndices(i), colIndices(i), values(i))
      i += 1
    }
  }

  override def numActives: Int = values.length

}
