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

package org.apache.spark.ml.regression

import scala.language.existentials
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.DataFrame


class VLinearRegressionSuite extends SparkFunSuite with MLlibTestSparkContext {

  import testImplicits._
  var datasetWithWeight: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    datasetWithWeight = sc.parallelize(Seq(
      Instance(17.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
      Instance(19.0, 2.0, Vectors.dense(1.0, 7.0)),
      Instance(23.0, 3.0, Vectors.dense(2.0, 11.0)),
      Instance(29.0, 4.0, Vectors.dense(3.0, 13.0))
    ), 2).toDF()
  }

  test("test on datasetWithWeight") {

    def b2s(b: Boolean): String = {
      if (b) "w/" else "w/o"
    }

    for (fitIntercept <- Seq(false, true)) {
      for (standardization <- Seq(false, true)) {
        for ((reg, elasticNet)<- Seq((0.0, 0.0), (2.3, 0.0), (2.3, 0.5))) {

          println()
          println(s"# test ${b2s(fitIntercept)} intercept, ${b2s(standardization)} standardization, reg=${reg}, elasticNet=${elasticNet}")

          val vtrainer = new VLinearRegression()
            .setColsPerBlock(1)
            .setRowsPerBlock(1)
            .setGeneratingFeatureMatrixBuffer(2)
            .setFitIntercept(fitIntercept)
            .setStandardization(standardization)
            .setRegParam(reg)
            .setWeightCol("weight")
            .setElasticNetParam(elasticNet)
          val vmodel = vtrainer.fit(datasetWithWeight)

          // Note that in ml.LinearRegression, when datasets numInstanse is small
          // solver l-bfgs and solver normal will generate slightly different result when reg not zero
          // because there std calculation result have multiple difference numInstance/(numInstance - 1)
          // here test keep consistent with l-bfgs solver
          val trainer = new LinearRegression()
            .setSolver("l-bfgs") // by default it may use noraml solver so here force set it.
            .setFitIntercept(fitIntercept)
            .setStandardization(standardization)
            .setRegParam(reg)
            .setWeightCol("weight")
            .setElasticNetParam(elasticNet)

          val model = trainer.fit(datasetWithWeight)
          logInfo(s"LinearRegression total iterations: ${model.summary.totalIterations}")

          println(s"VLinearRegression coefficients: ${vmodel.coefficients.toDense}, intercept: ${vmodel.intercept}\n" +
            s"LinearRegression coefficients: ${model.coefficients.toDense}, intercept: ${model.intercept}")

          def filterSmallValue(v: Vector) = {
            Vectors.dense(v.toArray.map(x => if (math.abs(x) < 1e-6) 0.0 else x))
          }
          assert(filterSmallValue(vmodel.coefficients) ~== filterSmallValue(model.coefficients) relTol 1e-3)
          assert(vmodel.intercept ~== model.intercept relTol 1e-3)
        }
      }
    }
  }
}
