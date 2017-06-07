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
 * Row major COO format matrix.
 * It will use less storage space than `ml.linalg.SparseMatrix`,
 * when the matrix is extremely sparse (for example, each row contains less than one
 * nonzero element in average)
 */

trait VMatrix extends Serializable {

  def numRows: Int

  def numCols: Int

  def rowIter: Iterator[Vector]

  def numActives: Int

  // A basic implementation, not use binary search.
  def apply(i: Int, j: Int): Double = {
    assert(i >= 0 && i < numRows && j >= 0 && j <= numCols)
    var elemVal = 0.0
    foreachActive { case (vi: Int, vj: Int, value: Double) =>
      if (vi == i && vj == j) {
        elemVal = value
      }
    }
    elemVal
  }

  private[spark] def foreachActive(f: (Int, Int, Double) => Unit)

}

object VMatrices {

  /**
   * Create COO format matrix from (rowIdx, colIdx, value) entries.
   */
  def COOEntries(
      numRows: Int,
      numCols: Int,
      entries: Iterable[(Int, Int, Double)],
      compressed: Boolean = false): VMatrix = {
    val sortedEntries = entries.toSeq.sortBy(v => (v._1, v._2)) // sort by row major order.
    val rowIndices = new Array[Int](sortedEntries.length)
    val colIndices = new Array[Int](sortedEntries.length)
    val values = new Array[Double](sortedEntries.length)

    var ptr = 0
    sortedEntries.foreach { case (rowIndex: Int, colIndex: Int, value: Double) =>
      rowIndices(ptr) = rowIndex
      colIndices(ptr) = colIndex
      values(ptr) = value
      ptr += 1
    }

    COOArrays(numRows, numCols, rowIndices, colIndices, values, compressed)
  }

  /**
   * Create COO format matrix from three arrays: row indices, column indices and values.
   */
  def COOArrays(
      numRows: Int,
      numCols: Int,
      rowIndices: Array[Int],
      colIndices: Array[Int],
      values: Array[Double],
      compressed: Boolean = false): VMatrix = {
    if (!compressed) {
      new VCOOMatrix(numRows, numCols, rowIndices, colIndices, values)
    } else {
      def intTo24Bit = (v: Int) =>
        Array((v & 255).toByte, ((v >> 8) & 255).toByte, ((v >> 16) & 255).toByte)

      new VCompressedCOOMatrix(numRows, numCols,
        rowIndices.flatMap(x => intTo24Bit(x)),
        colIndices.flatMap(x => intTo24Bit(x)), values.map(_.toFloat))
    }
  }

}
