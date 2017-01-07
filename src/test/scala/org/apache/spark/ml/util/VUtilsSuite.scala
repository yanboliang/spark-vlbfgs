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

import java.util.concurrent.Executors

import org.apache.spark.SparkFunSuite
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.linalg.distributed.DistributedVectorPartitioner
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext

class VUtilsSuite extends SparkFunSuite with MLlibTestSparkContext {

  import VUtils._

  @transient var testZipIdxRdd: RDD[Int] = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    testZipIdxRdd = sc.parallelize(Seq(
      (0, 1), (0, 2), (1, 1), (1, 2),
      (1, 3), (2, 1), (2, 2), (2, 3),
      (2, 4), (2,5)
    )).partitionBy(new DistributedVectorPartitioner(3)).map(_._2)
  }

  test("getNumBlocks") {
    assert(getNumBlocks(2, 3) == 2)
    assert(getNumBlocks(2, 4) == 2)
    assert(getNumBlocks(2, 5) == 3)
    assert(getNumBlocks(2, 6) == 3)
    assert(getNumBlocks(3, 4) == 2)
    assert(getNumBlocks(3, 5) == 2)
    assert(getNumBlocks(3, 6) == 2)
    assert(getNumBlocks(3, 7) == 3)
  }

  test("splitArrIntoDV") {
    val arrs = splitArrIntoDV(
      sc, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), 3, 3)
      .blocks.collect()

    assert(arrs(0).toArray === Array(1.0, 2.0, 3.0))
    assert(arrs(1).toArray === Array(4.0, 5.0, 6.0))
    assert(arrs(2).toArray === Array(7.0))
  }

  test("splitSparseVector") {
    val sv1 = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).toSparse
    val splist1 = splitSparseVector(sv1, 2)
    assert(
        splist1(0) == Vectors.dense(1.0, 2.0) &&
        splist1(1) == Vectors.dense(3.0, 4.0) &&
        splist1(2) == Vectors.dense(5.0, 6.0)
    )

    val sv2 = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0).toSparse
    val splist2 = splitSparseVector(sv2, 3)
    assert(
        splist2(0) == Vectors.dense(1.0, 2.0, 3.0) &&
        splist2(1) == Vectors.dense(4.0, 5.0)
    )
  }

  test("computePartitionSize") {
    val sizes = computePartitionSize(testZipIdxRdd).map(_.toInt)
    assert(sizes === Array(2, 3, 5))
  }

  test("zipRDDWithIndex") {
    val res = zipRDDWithIndex(Array(2L, 3L, 5L), testZipIdxRdd)
      .collect()
      .map(x => (x._1.toInt, x._2))
    assert(res ===
      Array(
        (0, 1), (1, 2), (2, 1), (3, 2),
        (4, 3), (5, 1), (6, 2), (7, 3),
        (8, 4), (9, 5)
      ))
  }

  test("vertcatSparseVectorIntoMatrix") {
    assert(vertcatSparseVectorIntoMatrix(Array(
      Vectors.dense(1.0, 2.0, 3.0).toSparse,
      Vectors.dense(8.0, 7.0, 9.0).toSparse
    )).toDense == Matrices.dense(2, 3, Array(1.0, 8.0, 2.0, 7.0, 3.0, 9.0)))
    assert(vertcatSparseVectorIntoMatrix(Array(
      Vectors.dense(1.0, 2.0).toSparse,
      Vectors.dense(3.0, 8.0).toSparse,
      Vectors.dense(7.0, 9.0).toSparse
    )).toDense == Matrices.dense(3, 2, Array(1.0, 3.0, 7.0, 2.0, 8.0, 9.0)))
  }

  test("vector summarizer") {
    val testRDD: RDD[Vector] = sc.parallelize(Seq(
      Vectors.dense(1.0, 2.0, 3.0),
      Vectors.dense(2.0, 0.0, 0.0),
      Vectors.dense(5.0, 2.0, 1.0),
      Vectors.dense(-4.0, 2.0, 1.0)
    ), 3)
    val result = testRDD.aggregate(new VectorSummarizer)(
      (s: VectorSummarizer, v: Vector) => s.add(v),
      (s1: VectorSummarizer, s2: VectorSummarizer) => s1.merge(s2)
    ).toDenseVector

    assert(result ~== Vectors.dense(4.0, 6.0, 5.0) relTol 1e-3)
  }

  test("concurrent execute tasks") {
    val res = new Array[Int](10)
    val pool = Executors.newCachedThreadPool()
    VUtils.concurrentExecuteTasks(0 until 10, pool, (taskId: Int) => {
      Thread.sleep(500 + 200 * (taskId % 3))
      res(taskId) = taskId * 10
    })
    assert(res === (0 until 100 by 10).toArray)
  }

  test("getIteratorZipWithIndex") {
    val iterator = VUtils.getIteratorZipWithIndex(Iterator(0, 1, 2), -1L + Int.MaxValue)
    assert(iterator.toArray === Array(
      (-1L + Int.MaxValue, 0), (0L + Int.MaxValue, 1), (1L + Int.MaxValue, 2)
    ))
    intercept[IllegalArgumentException] {
      VUtils.getIteratorZipWithIndex(Iterator(0, 1, 2), -1L)
    }
  }
}
