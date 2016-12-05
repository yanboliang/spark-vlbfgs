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
import org.apache.spark.ml.classification.LogisticRegressionSuite._
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext

class VLogisticRegressionSuite extends SparkFunSuite with MLlibTestSparkContext {

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  val data1 = Seq(
    Instance(1.0, 1.0, Vectors.dense(1.0, 0.0, -2.1, -1.5).toSparse),
    Instance(0.0, 2.0, Vectors.dense(0.9, 0.0, -2.1, -1.1).toSparse),
    Instance(1.0, 3.5, Vectors.dense(1.0, 0.0, 0.0, 1.2).toSparse),
    Instance(0.0, 0.0, Vectors.dense(-1.5, 0.0, -0.5, 0.0).toSparse),
    Instance(1.0, 1.0, Vectors.dense(-1.9, 0.0, -0.3, -1.5).toSparse)
  )

  test("test on data1, w/ weight, L2 reg = 0, w/o intercept") {
    val df1 = spark.createDataFrame(sc.parallelize(data1, 2))

    val vtrainer = new VLogisticRegression()
      .setColsPerBlock(3)
      .setRowsPerBlock(2)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)
    val vmodel = vtrainer.fit(df1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)
    val model = trainer.fit(df1)

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients.toLocal}\n" +
      s"LogisticRegression coefficients: ${model.coefficients}")
    assert(vmodel.coefficients.toLocal ~== model.coefficients relTol 1E-5)
  }

  test("test on data1, w/o weight, L2 reg = 0, w/o intercept") {
    val df1 = spark.createDataFrame(sc.parallelize(data1, 2))

    val vtrainer = new VLogisticRegression()
      .setColsPerBlock(2)
      .setRowsPerBlock(3)
      .setRegParam(0.0)
      .setStandardization(true)
    val vmodel = vtrainer.fit(df1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setRegParam(0.0)
      .setStandardization(true)
    val model = trainer.fit(df1)

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients.toLocal}\n" +
      s"LogisticRegression coefficients: ${model.coefficients}")
    assert(vmodel.coefficients.toLocal ~== model.coefficients relTol 1E-5)
  }

  test("test on data1, w/ weight, L2 reg = 0.8, w/ standardize, w/o intercept") {
    val df1 = spark.createDataFrame(sc.parallelize(data1, 2))

    val vtrainer = new VLogisticRegression()
      .setColsPerBlock(1)
      .setRowsPerBlock(1)
      .setColPartitions(3)
      .setRowPartitions(2)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(true)
    val vmodel = vtrainer.fit(df1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(true)
    val model = trainer.fit(df1)

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients.toLocal}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients.toLocal ~== model.coefficients relTol 1E-5)
  }

  test("test on data1, w/ weight, L2 reg = 0.8, w/o standardize, w/o intercept") {
    val df1 = spark.createDataFrame(sc.parallelize(data1, 2))

    val vtrainer = new VLogisticRegression()
      .setColsPerBlock(4)
      .setRowsPerBlock(5)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(false)
    val vmodel = vtrainer.fit(df1)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.8)
      .setStandardization(false)
    val model = trainer.fit(df1)

    println(s"VLogisticRegression coefficients: ${vmodel.coefficients.toLocal}\n" +
      s"LogisticRegression coefficient: ${model.coefficients}")
    assert(vmodel.coefficients.toLocal ~== model.coefficients relTol 1E-5)
  }

  test("VLogisticRegression (binary) w/ weighted samples") {
    val (dataset, weightedDataset) = {
      val nPoints = 20
      val coefficients = Array(-0.57997, 0.912083)
      val xMean = Array(0.1, -0.1)
      val xVariance = Array(0.6856, 0.1899)
      val testData =
        generateMultinomialLogisticInput(coefficients, xMean, xVariance, false, nPoints, 42)

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

    val vtrainer1 = new VLogisticRegression()
      .setColsPerBlock(2)
      .setRowsPerBlock(2)
      .setColPartitions(2)
      .setRowPartitions(3)
      .setRegParam(0.0)
      .setStandardization(true)
    val vtrainer2 = new VLogisticRegression()
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
    println(s"vmodel1: ${vmodel1.coefficients.toLocal}\n" +
      s"vmodel2: ${vmodel2.coefficients.toLocal}\nvmodel3: ${vmodel3.coefficients.toLocal}")
    assert(vmodel1.coefficients.toLocal !~= vmodel2.coefficients.toLocal absTol 1E-3)
    assert(vmodel1.coefficients.toLocal ~== vmodel3.coefficients.toLocal absTol 1E-3)

    val trainer = new LogisticRegression()
      .setFitIntercept(false)
      .setWeightCol("weight")
      .setRegParam(0.0)
      .setStandardization(true)
    val model = trainer.fit(weightedDataset)
    println(s"model: ${model.coefficients}")
    assert(model.coefficients ~== vmodel1.coefficients.toLocal absTol 1E-5)
  }
}
