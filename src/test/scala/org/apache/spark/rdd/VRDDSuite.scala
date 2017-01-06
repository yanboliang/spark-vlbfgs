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

package org.apache.spark.rdd

import org.apache.spark.{SharedSparkContext, SparkFunSuite}

class VRDDSuite extends SparkFunSuite with SharedSparkContext {

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  test("test multiZipRDDs") {
    val rdd1 = sc.makeRDD(Array(1, 2, 3, 4), 2)
    val rddList = List(rdd1, rdd1.map(_ + 10), rdd1.map(_ + 200))
    val zipped = VRDDFunctions.zipMultiRDDs(rddList) {
      iterList: List[Iterator[Int]] => new Iterator[Int]{
        override def hasNext: Boolean = iterList.map(_.hasNext).reduce(_ && _)
        override def next(): Int = iterList.map(_.next()).sum
      }
    }
    assert(zipped.glom().map(_.toList).collect().toList ===
      List(List(213, 216), List(219, 222)))
  }
}
