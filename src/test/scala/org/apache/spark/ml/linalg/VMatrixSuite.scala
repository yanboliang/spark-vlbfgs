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

import org.apache.spark.SparkFunSuite

import scala.collection.mutable.ArrayBuffer

class VMatrixSuite extends SparkFunSuite {

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  test ("rowIter") {

    for (compressed <- List(false, true)) {
      val emptyRow = Vectors.sparse(4, new Array[(Int, Double)](0))

      assert(VMatrices.COOEntries(3, 4, new Array[(Int, Int, Double)](0), compressed).rowIter.toArray ===
        Array(emptyRow, emptyRow, emptyRow)
      )

      assert(VMatrices.COOEntries(3, 4, Array((2, 2, 4.0), (2, 3, 5.0), (0, 1, 2.0), (0, 3, 3.0)), compressed)
        .rowIter.toArray ===
        Array(
          Vectors.sparse(4, Array((1, 2.0), (3, 3.0))),
          emptyRow,
          Vectors.sparse(4, Array((2, 4.0), (3, 5.0)))
        ))

      assert(VMatrices.COOEntries(6, 4, Array((1, 1, 2.0), (1, 3, 3.0), (4, 1, 7.0)), compressed)
        .rowIter.toArray ===
        Array(
          emptyRow,
          Vectors.sparse(4, Array((1, 2.0), (3, 3.0))),
          emptyRow,
          emptyRow,
          Vectors.sparse(4, Array((1, 7.0))),
          emptyRow
        )
      )
    }
    assert(VMatrices.COOEntries(8401234, 7105678, Array((70000, 7000000, 2.0),
      (8400000, 6999999, 7.0), (8400000, 125, 3.0), (255, 2, 4.0), (254, 65050, 5.0)), true)
      .rowIter.zipWithIndex.filter(_._1.numActives > 0).toArray ===
      Array(
        (Vectors.sparse(7105678, Array((65050, 5.0))), 254),
        (Vectors.sparse(7105678, Array((2, 4.0))), 255),
        (Vectors.sparse(7105678, Array((7000000, 2.0))), 70000),
        (Vectors.sparse(7105678, Array((125, 3.0), (6999999, 7.0))), 8400000)
      )
    )
  }

  test ("fromCOO && apply && foreach") {

    for (compressed <- List(false, true)) {
      val mat = VMatrices.COOEntries(8401234, 7105678, Array((70000, 7000000, 2.0),
        (8400000, 6999999, 7.0), (8400000, 125, 3.0), (255, 2, 4.0), (254, 65050, 5.0)), true)
      assert(mat(255, 666666) == 0.0 && mat(70000, 7000000) == 2.0 && mat(8400000, 125) == 3.0
        && mat(255, 2) == 4.0 && mat(254, 65050) == 5.0)

      val buff = new ArrayBuffer[(Int, Int, Double)]()
      mat.foreachActive { case (i: Int, j: Int, value: Double) =>
        buff += Tuple3(i, j, value)
      }
      assert(buff.toArray === Array((254, 65050, 5.0), (255, 2, 4.0), (70000, 7000000, 2.0),
        (8400000, 125, 3.0), (8400000, 6999999, 7.0)))
    }
  }

}
