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
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{SparseMatrix, Vector, Vectors}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.language.existentials


class VSoftmaxRegressionSuite extends SparkFunSuite with MLlibTestSparkContext {

  import testImplicits._

  private val seed = 42
  @transient var multinomialDataset: Dataset[_] = _
  private val eps: Double = 1e-5

  override def beforeAll(): Unit = {
    super.beforeAll()

    multinomialDataset = {
      val nPoints = 50
      val coefficients = Array(
        -0.57997, 0.912083, -0.371077, -0.819866, 2.688191,
        -0.16624, -0.84355, -0.048509, -0.301789, 4.170682)

      val xMean = Array(5.843, 3.057, 3.758, 1.199)
      val xVariance = Array(0.6856, 0.1899, 3.116, 0.581)

      val testData = LogisticRegressionSuite.generateMultinomialLogisticInput(
        coefficients, xMean, xVariance, addIntercept = true, nPoints, seed)

      val df = sc.parallelize(testData, 4).toDF().withColumn("weight", rand(seed))
      df.cache()
      println("softmax test data:")
      df.show(10, false)
      df
    }
  }

  test("test on multinomialDataset") {

    def b2s(b: Boolean): String = {
      if (b) "w/" else "w/o"
    }

    for (standardization <- Seq(false, true)) {
      for ((reg, elasticNet) <- Seq((0.0, 0.0), (2.3, 0.0), (0.3, 0.05), (0.01, 1.0))) {
        println()
        println(s"# test ${b2s(standardization)} standardization, reg=${reg}, elasticNet=${elasticNet}")

        val trainer = new LogisticRegression()
          .setFamily("multinomial")
          .setStandardization(standardization)
          .setWeightCol("weight")
          .setRegParam(reg)
          .setFitIntercept(false)
          .setElasticNetParam(elasticNet)

        val model = trainer.fit(multinomialDataset)

        val vtrainer = new VSoftmaxRegression()
          .setColsPerBlock(2)
          .setRowsPerBlock(5)
          .setColPartitions(2)
          .setRowPartitions(3)
          .setWeightCol("weight")
          .setGeneratingFeatureMatrixBuffer(2)
          .setStandardization(standardization)
          .setRegParam(reg)
          .setElasticNetParam(elasticNet)
        val vmodel = vtrainer.fit(multinomialDataset)

        println(s"VSoftmaxRegression coefficientMatrix:\n" +
          s"${vmodel.coefficientMatrix.asInstanceOf[SparseMatrix].toDense},\n" +
          s"ml.SoftmaxRegression coefficientMatrix:\n" +
          s"${model.coefficientMatrix}\n")

        assert(vmodel.coefficientMatrix ~== model.coefficientMatrix relTol eps)
      }
    }
  }
}
