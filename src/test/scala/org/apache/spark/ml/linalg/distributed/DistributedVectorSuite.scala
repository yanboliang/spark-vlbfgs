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

import breeze.linalg.{DenseVector => BDV, norm => Bnorm}
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.ml.util.VUtils
import org.apache.spark.mllib.util.MLlibTestSparkContext

class DistributedVectorSuite extends SparkFunSuite with MLlibTestSparkContext {

  var BV1: BDV[Double] = null
  var BV2: BDV[Double] = null
  var BV3: BDV[Double] = null
  var BV4: BDV[Double] = null
  var DV1: DistributedVector = null
  var DV2: DistributedVector = null
  var DV3: DistributedVector = null
  var DV4: DistributedVector = null

  override def beforeAll(): Unit = {
    super.beforeAll()

    val v1: Array[Double] = Seq(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).toArray
    val v2: Array[Double] = Seq(-1.0, -2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0).toArray
    val v3: Array[Double] = Seq(-1.0, -2.0, -3.0, 5.0, -6.0, 7.0, 8.0, 9.0).toArray
    val v4: Array[Double] = Seq(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 8.0, 0.0).toArray

    BV1 = new BDV(v1)
    BV2 = new BDV(v2)
    BV3 = new BDV(v3)
    BV4 = new BDV(v4)

    val sizePerPart = 3
    val numPartitions = VUtils.getNumBlocks(sizePerPart, v1.length)

    DV1 = VUtils.splitArrIntoDV(sc, v1, sizePerPart, numPartitions).persist()
    DV2 = VUtils.splitArrIntoDV(sc, v2, sizePerPart, numPartitions).persist()
    DV3 = VUtils.splitArrIntoDV(sc, v3, sizePerPart, numPartitions).persist()
    DV4 = VUtils.splitArrIntoDV(sc, v4, sizePerPart, numPartitions).persist()
  }

  test("toLocal") {
    val localDV1 = DV1.toLocal
    assert(localDV1 ~== Vectors.fromBreeze(BV1) relTol 1e-8)
    val localDV4 = DV4.compressed.toLocal
    assert(localDV4 ~== Vectors.fromBreeze(BV4) relTol 1e-8)
  }

  test("add") {
    val local1 = DV1.add(2.0).persist().toLocal
    val local2 = DV1.add(DV2).persist().toLocal
    assert(local1 ~== Vectors.fromBreeze(BV1 + 2.0) relTol 1e-8)
    assert(local2 ~== Vectors.fromBreeze(BV1 + BV2) relTol 1e-8)
  }

  test("scale") {
    val local1 = DV1.scale(2.0).persist().toLocal
    assert(local1 ~== Vectors.fromBreeze(BV1 * 2.0) relTol 1e-8)
  }

  test("addScaledVector") {
    val res = DV1.addScaledVector(3.0, DV2).persist().toLocal
    assert(res ~== Vectors.fromBreeze(BV1 + (BV2 * 3.0)) relTol 1e-8)
  }

  test("dot") {
    val dotVal = DV1.dot(DV2)
    val bDotVal = BV1.dot(BV2)
    assert(dotVal ~== bDotVal relTol 1e-8)
  }

  test("norm") {
    assert(DV1.norm ~== Bnorm(BV1) relTol 1e-8)
  }

  test("combine") {
    val combined = DistributedVectors.combine(
      (10.0, DV1), (100.0, DV2), (18.0, DV3)
    ).persist().toLocal
    val bCombined = (BV1 * 10.0) + (BV2 * 100.0) + (BV3 * 18.0)
    assert(combined ~== Vectors.fromBreeze(bCombined) relTol 1e-8)
  }

  test("zeros") {
    var res1 = VUtils.zipRDDWithPartitionIDAndCollect(
      DistributedVectors.zeros(sc, 3, 2, 5).values)
    var res2 = Array((0, Vectors.dense(0.0, 0.0, 0.0)), (1, Vectors.dense(0.0, 0.0)))
    assert(res1 === res2)
    res1 = VUtils.zipRDDWithPartitionIDAndCollect(
      DistributedVectors.zeros(sc, 3, 2, 7, 1.5).values)
    res2 = Array((0, Vectors.dense(0.0, 0.0, 0.0)), (1, Vectors.dense(0.0, 0.0, 0.0, 1.5)))
    assert(res1 === res2)
  }
}
