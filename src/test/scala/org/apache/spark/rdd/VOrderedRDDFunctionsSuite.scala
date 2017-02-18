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

import org.apache.spark.rdd.VOrderedRDDFunctions._
import org.apache.spark.{Partitioner, SparkFunSuite}
import org.apache.spark.mllib.util.MLlibTestSparkContext

class VOrderedRDDFunctionsSuite extends SparkFunSuite with MLlibTestSparkContext {

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  test("testGroupByKeyUsingSort") {
    val rdd: RDD[(Int, Int)] =
      sc.parallelize(Seq((1, 4), (1, 5), (1, 8), (0, 3), (0, 6), (2, 3), (3, 2)), 3)
    val res = rdd.groupByKeyUsingSort(new Partitioner {
      override def numPartitions: Int = 3
      override def getPartition(key: Any): Int = key.asInstanceOf[Int] % 3
    }).mapValues(_.toList).collect()

    assert(res === Array(
      (0, List(3, 6)), (3, List(2)), (1, List(4, 5, 8)), (2, List(3))
    ))
  }

}
