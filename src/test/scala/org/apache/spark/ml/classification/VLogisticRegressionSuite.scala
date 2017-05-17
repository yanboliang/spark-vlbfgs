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

import scala.language.existentials
import scala.util.Random
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{DataFrame, Row}

import scala.util.control.Breaks._

class VLogisticRegressionSuite extends SparkFunSuite with MLlibTestSparkContext {

  override def beforeAll(): Unit = {
    super.beforeAll()

    testData1 = genLORTestData(
      nPoints = 20,
      coefficients = Array(-0.57997, 0.912083, -0.371077, -0.819866),
      xMean = Array(5.843, 3.057, 3.758, 1.199),
      xVariance = Array(0.6856, 0.1899, 3.116, 0.581),
      addIntecept = false
    )._2

    testData1WithIntecept = genLORTestData(
      nPoints = 20,
      coefficients = Array(-0.57997, 0.912083, -0.371077, -0.819866, 2.688191),
      xMean = Array(5.843, 3.057, 3.758, 1.199),
      xVariance = Array(0.6856, 0.1899, 3.116, 0.581),
      addIntecept = true
    )._2
  }

  var testData1: DataFrame = null
  var testData1WithIntecept: DataFrame = null

  def genLORTestData(
      nPoints: Int,
      coefficients: Array[Double],
      xMean: Array[Double],
      xVariance: Array[Double],
      addIntecept: Boolean): (DataFrame, DataFrame) = {

    val testData = LogisticRegressionSuite.generateMultinomialLogisticInput(
        coefficients, xMean, xVariance, addIntecept, nPoints, 42)

    // Let's over-sample the positive samples twice.
    val data1 = testData.flatMap { case labeledPoint: LabeledPoint =>
      if (labeledPoint.label == 1.0) {
        Iterator(labeledPoint, labeledPoint)
      } else {
        Iterator(labeledPoint)
      }
    }

    val rnd = new Random(8392)
    val data2 = testData.flatMap { case LabeledPoint(label: Double, features: Vector) =>
      if (rnd.nextGaussian() > 0.0) {
        if (label == 1.0) {
          Iterator(
            Instance(label, 1.2, features),
            Instance(label, 0.8, features),
            Instance(0.0, 0.0, features))
        } else {
          Iterator(
            Instance(label, 0.3, features),
            Instance(1.0, 0.0, features),
            Instance(label, 0.1, features),
            Instance(label, 0.6, features))
        }
      } else {
        if (label == 1.0) {
          Iterator(Instance(label, 2.0, features))
        } else {
          Iterator(Instance(label, 1.0, features))
        }
      }
    }

    (spark.createDataFrame(sc.parallelize(data1, 4)),
      spark.createDataFrame(sc.parallelize(data2, 4)))
  }

  test("test on testData1, w/ weight, L2 reg = 0, w/o intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(3)
      .setRowsPerBlock(2)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)
    val model = trainer.fit(testData1)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficients: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
  }

  test("test on testData1, w/o weight, L2 reg = 0, w/o intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(2)
      .setRowsPerBlock(3)
      .setGeneratingFeatureMatrixBuffer(2)
      .setRegParam(0.0)
      .setStandardization(true)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setRegParam(0.0)
      .setStandardization(true)
    val model = trainer.fit(testData1)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficients: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
  }

  test("test on testData1, w/ weight, L2 reg = 0.8, w/ standardize, w/o intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(1)
      .setRowsPerBlock(1)
      .setColPartitions(3)
      .setRowPartitions(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(true)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(true)
    val model = trainer.fit(testData1)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
  }

  test("test on testData1, w/ weight, L2 reg = 0.8, w/o standardize, w/o intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(4)
      .setRowsPerBlock(5)
      .setGeneratingFeatureMatrixBuffer(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(false)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(false)
    val model = trainer.fit(testData1)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
  }

  test("test on testData1, w/ weight, L2 reg = 0.8, w/o standardize, w/o intercept, compress features") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(4)
      .setRowsPerBlock(5)
      .setGeneratingFeatureMatrixBuffer(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(false)
      .setEagerPersist(false)
      .setCompressFeatureMatrix(true)
    val vmodel = vtrainer.fit(testData1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(false)
    val model = trainer.fit(testData1)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
  }

  test("test on testData1, w/ weight, reg = 0.8, elasticNet = 1.0, w/ standardize, w/o intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(1)
      .setRowsPerBlock(1)
      .setColPartitions(3)
      .setRowPartitions(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(1.0)
      .setStandardization(true)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(1.0)
      .setStandardization(true)
    val model = trainer.fit(testData1)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
  }

  test("test on testData1, w/ weight, reg = 0.8, elasticNet = 1.0, w/o standardize, w/o intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(1)
      .setRowsPerBlock(1)
      .setColPartitions(3)
      .setRowPartitions(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(1.0)
      .setStandardization(false)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(1.0)
      .setStandardization(false)
    val model = trainer.fit(testData1)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
  }

  test("test on testData1, w/ weight, reg = 0.8, elasticNet = 0.6, w/o standardize, w/o intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(4)
      .setRowsPerBlock(5)
      .setGeneratingFeatureMatrixBuffer(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(0.6)
      .setStandardization(false)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(0.6)
      .setStandardization(false)
    val model = trainer.fit(testData1)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
  }

  test("test on testData1WithIntecept, w/ weight, L2 reg = 0, w/ intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(true)
      .setColsPerBlock(3)
      .setRowsPerBlock(2)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1WithIntecept)

    val trainer = new LogisticRegression()
      .setFitIntercept(true)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)
    val model = trainer.fit(testData1WithIntecept)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficients: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
    println(s"VLogisticRegression intercept: ${vmodel.intercept}\n" +
      s"LogisticRegression intercept: ${model.intercept}")
    assert(vmodel.intercept ~== model.intercept relTol 1e-3)
  }

  test("test on testData1WithIntecept, w/o weight, L2 reg = 0, w/ intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(true)
      .setColsPerBlock(2)
      .setRowsPerBlock(3)
      .setGeneratingFeatureMatrixBuffer(2)
      .setRegParam(0.0)
      .setStandardization(true)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1WithIntecept)

    val trainer = new LogisticRegression()
      .setFitIntercept(true)
      .setRegParam(0.0)
      .setStandardization(true)
    val model = trainer.fit(testData1WithIntecept)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficients: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
    println(s"VLogisticRegression intercept: ${vmodel.intercept}\n" +
      s"LogisticRegression intercept: ${model.intercept}")
    assert(vmodel.intercept ~== model.intercept relTol 1e-3)
  }

  test("test on testData1WithIntecept, w/ weight, L2 reg = 0.8, w/ standardize, w/ intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(true)
      .setColsPerBlock(1)
      .setRowsPerBlock(1)
      .setColPartitions(3)
      .setRowPartitions(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(true)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1WithIntecept)

    val trainer = new LogisticRegression()
      .setFitIntercept(true)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(true)
    val model = trainer.fit(testData1WithIntecept)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
    println(s"VLogisticRegression intercept: ${vmodel.intercept}\n" +
      s"LogisticRegression intercept: ${model.intercept}")
    assert(vmodel.intercept ~== model.intercept relTol 1e-3)
  }

  test("test on testData1WithIntecept, w/ weight, L2 reg = 0.8, w/o standardize, w/ intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(true)
      .setColsPerBlock(4)
      .setRowsPerBlock(5)
      .setGeneratingFeatureMatrixBuffer(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(false)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1WithIntecept)

    val trainer = new LogisticRegression()
      .setFitIntercept(true)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(false)
    val model = trainer.fit(testData1WithIntecept)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
    println(s"VLogisticRegression intercept: ${vmodel.intercept}\n" +
      s"LogisticRegression intercept: ${model.intercept}")
    assert(vmodel.intercept ~== model.intercept relTol 1e-3)
  }

  test("test on testData1WithIntecept, w/ weight, reg = 0.9, elasticNet = 0.1, w/ standardize, w/ intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(true)
      .setColsPerBlock(1)
      .setRowsPerBlock(1)
      .setColPartitions(3)
      .setRowPartitions(2)
      .setWeightCol("weight")
      .setRegParam(0.9)
      .setElasticNetParam(0.1)
      .setStandardization(true)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1WithIntecept)

    val trainer = new LogisticRegression()
      .setFitIntercept(true)
      .setWeightCol("weight")
      .setRegParam(0.9)
      .setElasticNetParam(0.1)
      .setStandardization(true)
    val model = trainer.fit(testData1WithIntecept)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
    println(s"VLogisticRegression intercept: ${vmodel.intercept}\n" +
      s"LogisticRegression intercept: ${model.intercept}")
    assert(vmodel.intercept ~== model.intercept relTol 1e-3)
  }

  test("test on testData1WithIntecept, w/ weight, reg = 0.8, elasticNet = 1.0, w/o standardize, w/ intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(true)
      .setColsPerBlock(1)
      .setRowsPerBlock(1)
      .setColPartitions(3)
      .setRowPartitions(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(1.0)
      .setStandardization(false)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1WithIntecept)

    val trainer = new LogisticRegression()
      .setFitIntercept(true)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(1.0)
      .setStandardization(false)
    val model = trainer.fit(testData1WithIntecept)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
    println(s"VLogisticRegression intercept: ${vmodel.intercept}\n" +
      s"LogisticRegression intercept: ${model.intercept}")
    assert(vmodel.intercept ~== model.intercept relTol 1e-3)
  }

  test("test on testData1WithIntecept, w/ weight, reg = 0.8, elasticNet = 0.6, w/o standardize, w/ intercept") {

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(true)
      .setColsPerBlock(4)
      .setRowsPerBlock(5)
      .setGeneratingFeatureMatrixBuffer(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(0.6)
      .setStandardization(false)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1WithIntecept)

    val trainer = new LogisticRegression()
      .setFitIntercept(true)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setElasticNetParam(0.6)
      .setStandardization(false)
    val model = trainer.fit(testData1WithIntecept)
    logInfo(s"LogisticRegression total iterations: ${model.summary.totalIterations}")

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients ~== model.coefficients relTol 1e-3)
    println(s"VLogisticRegression intercept: ${vmodel.intercept}\n" +
      s"LogisticRegression intercept: ${model.intercept}")
    assert(vmodel.intercept ~== model.intercept relTol 1e-3)
  }

  test("VLogisticRegression (binary) w/ weighted samples") {

    val (dataset, weightedDataset) = genLORTestData(
      nPoints = 20,
      coefficients = Array(-0.57997, 0.912083),
      xMean = Array(0.1, -0.1),
      xVariance = Array(0.6856, 0.1899),
      addIntecept = false
    )

    val vtrainer1 = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(2)
      .setRowsPerBlock(2)
      .setColPartitions(2)
      .setRowPartitions(3)
      .setRegParam(0.0)
      .setStandardization(true)
    val vtrainer2 = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(2)
      .setRowsPerBlock(2)
      .setColPartitions(3)
      .setRowPartitions(4)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)

    val vmodel1 = vtrainer1.fit(dataset)
    val vmodel2 = vtrainer1.fit(weightedDataset)
    val vmodel3 = vtrainer2.fit(weightedDataset)
    println(s"vmodel1: ${vmodel1.coefficients}\n" +
      s"vmodel2: ${vmodel2.coefficients}\nvmodel3: ${vmodel3.coefficients}")
    assert(vmodel1.coefficients !~= vmodel2.coefficients absTol 1E-3)
    assert(vmodel1.coefficients ~== vmodel3.coefficients absTol 1E-3)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)
    val model = trainer.fit(weightedDataset)
    println(s"model: ${model.coefficients}")
    assert(model.coefficients ~== vmodel1.coefficients absTol 1e-3)
  }

  test("VLogisticRegression: Predictor, Classifier methods") {
    val sqlContext = testData1.sqlContext
    import sqlContext.implicits._

    val vtrainer = new VLogisticRegression()
      .setFitIntercept(false)
      .setColsPerBlock(3)
      .setRowsPerBlock(2)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)
      .setEagerPersist(false)
    val vmodel = vtrainer.fit(testData1)
    assert(vmodel.numClasses === 2)
    val numFeatures = testData1.select("features").first().getAs[Vector](0).size
    assert(vmodel.numFeatures === numFeatures)

    val results = vmodel.transform(testData1)

    // Compare rawPrediction with probability
    results.select("rawPrediction", "probability").collect().foreach {
      case Row(raw: Vector, prob: Vector) =>
        assert(raw.size === 2)
        assert(prob.size === 2)
        val probFromRaw1 = 1.0 / (1.0 + math.exp(-raw(1)))
        assert(prob(1) ~== probFromRaw1 relTol 1e-5)
        assert(prob(0) ~== 1.0 - probFromRaw1 relTol 1e-5)
    }

    // Compare prediction with probability
    results.select("prediction", "probability").collect().foreach {
      case Row(pred: Double, prob: Vector) =>
        val predFromProb = prob.toArray.zipWithIndex.maxBy(_._1)._2
        assert(pred === predFromProb)
    }

    // force it to use raw2prediction
    vmodel.setProbabilityCol("")
    val resultsUsingRaw2Predict =
      vmodel.transform(testData1).select("prediction").as[Double].collect()
    resultsUsingRaw2Predict.zip(results.select("prediction").as[Double].collect()).foreach {
      case (pred1, pred2) => assert(pred1 === pred2)
    }

    // force it to use probability2prediction
    vmodel.setRawPredictionCol("")
    val resultsUsingProb2Predict =
      vmodel.transform(testData1).select("prediction").as[Double].collect()
    resultsUsingProb2Predict.zip(results.select("prediction").as[Double].collect()).foreach {
      case (pred1, pred2) => assert(pred1 === pred2)
    }

    // force it to use predict
    vmodel.setRawPredictionCol("").setProbabilityCol("")
    val resultsUsingPredict =
      vmodel.transform(testData1).select("prediction").as[Double].collect()
    resultsUsingPredict.zip(results.select("prediction").as[Double].collect()).foreach {
      case (pred1, pred2) => assert(pred1 === pred2)
    }
  }

}
