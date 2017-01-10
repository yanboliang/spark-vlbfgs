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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.distributed.{DistributedVectorPartitioner, VGridPartitioner}
import org.apache.spark.mllib.util.MLlibTestSparkContext

class VRDDFunctionsSuite extends SparkFunSuite with MLlibTestSparkContext {

  import org.apache.spark.rdd.VRDDFunctions._

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  def testMapJoinPartitions(shuffleRdd2: Boolean): Unit = {
    val sc = spark.sparkContext
    val rdd1 = sc.parallelize(Array.tabulate(81) {
      idx => {
        val rowIdx = idx % 9
        val colIdx = idx / 9
        ((rowIdx, colIdx), (rowIdx, colIdx))
      }
    }).partitionBy(VGridPartitioner(9, 9, 3, 3)).cache()
    rdd1.count()
    val rdd2 = sc.parallelize(Array.tabulate(9)(idx => (idx, idx)))
      .partitionBy(new DistributedVectorPartitioner(9)).cache()
    rdd2.count()

    val rddr = rdd1.mapJoinPartition(rdd2, shuffleRdd2)(
      (x: Int) => {
        val blockColIdx = x / 3
        val pos = blockColIdx * 3
        Array(pos, pos + 1, pos + 2)
      },
      (p1: Int, iter1, list: Array[(Int, Iterator[(Int, Int)])]) => {
        Iterator((p1, list.map(tuple => (tuple._1, tuple._2.next())).mkString(",")))
      }
    )

    assert(rddr.collect() === Array(
      (0, "(0,(0,0)),(1,(1,1)),(2,(2,2))"),
      (1, "(0,(0,0)),(1,(1,1)),(2,(2,2))"),
      (2, "(0,(0,0)),(1,(1,1)),(2,(2,2))"),
      (3, "(3,(3,3)),(4,(4,4)),(5,(5,5))"),
      (4, "(3,(3,3)),(4,(4,4)),(5,(5,5))"),
      (5, "(3,(3,3)),(4,(4,4)),(5,(5,5))"),
      (6, "(6,(6,6)),(7,(7,7)),(8,(8,8))"),
      (7, "(6,(6,6)),(7,(7,7)),(8,(8,8))"),
      (8, "(6,(6,6)),(7,(7,7)),(8,(8,8))")
    ))
  }

  test("mapJoinPartitions V1") {
    testMapJoinPartitions(false)
  }

  test("mapJoinPartitions V2") {
    testMapJoinPartitions(true)
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
