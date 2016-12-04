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

package org.apache.spark.ml.classification

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.Dataset

class RealDataSuite extends SparkFunSuite with MLlibTestSparkContext {

  // https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a
  @transient var dataset1: Dataset[_] = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    dataset1 = spark.read.format("libsvm").load("data/a9a")
  }

  ignore("a9a") {
    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setRegParam(0.5)
    val model = trainer.fit(dataset1)

    val vtrainer = new VLogisticRegression()
      .setColsPerBlock(100)
      .setRowsPerBlock(10)
      .setColPartitions(3)
      .setRowPartitions(3)
      .setRegParam(0.5)
    val vmodel = vtrainer.fit(dataset1)

    assert(vmodel.coefficients.toLocal ~== model.coefficients relTol 1E-3)
  }

}