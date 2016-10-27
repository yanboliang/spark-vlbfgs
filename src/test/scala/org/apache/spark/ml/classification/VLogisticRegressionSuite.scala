package org.apache.spark.ml.classification

import scala.collection.JavaConverters._
import scala.language.existentials
import scala.util.Random
import scala.util.control.Breaks._

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.classification.LogisticRegressionSuite._
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.lit

/**
  * Created by ThinkPad on 2016/10/8.
  */
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

  test("test on data1, use weight, L2 reg = 0, without intercept") {
    val dataset1 = spark.createDataFrame(sc.parallelize(data1, 2))
    val vftrainer1 = (new VLogisticRegression).setFeaturesPartSize(3).setInstanceStackSize(2)
      .setWeightCol("weight").setRegParam(0.0).setStandardization(true)
    val vfmodel1 = vftrainer1.fit(dataset1)
    val singletrainer1 = (new LogisticRegression).setFitIntercept(false)
      .setWeightCol("weight").setRegParam(0.0).setStandardization(true)
    val singlemodel1 = singletrainer1.fit(dataset1)

    println(s"with weight: vf coeffs: ${vfmodel1.coefficients.toLocal()}\nsingle coeffs: ${singlemodel1.coefficients}")
    assert(vfmodel1.coefficients.toLocal() ~== singlemodel1.coefficients relTol 1E-5)

  }
  test("test on data1, ignore weight, L2 reg = 0, without intercept") {
    val dataset1 = spark.createDataFrame(sc.parallelize(data1, 2))
    val vftrainer2 = (new VLogisticRegression).setFeaturesPartSize(2).setInstanceStackSize(3)
      .setRegParam(0.0).setStandardization(true)
    val vfmodel2 = vftrainer2.fit(dataset1)
    val singletrainer2 = (new LogisticRegression).setFitIntercept(false)
      .setRegParam(0.0).setStandardization(true)
    val singlemodel2 = singletrainer2.fit(dataset1)

    println(s"without weight: vf coeffs: ${vfmodel2.coefficients.toLocal()}\nsingle coeffs: ${singlemodel2.coefficients}")
    assert(vfmodel2.coefficients.toLocal() ~== singlemodel2.coefficients relTol 1E-5)
  }

  test("test on data1, use weight, L2 reg = 0.8, standardize = true, without intercept") {
    val dataset1 = spark.createDataFrame(sc.parallelize(data1, 2))
    val vftrainer2 = (new VLogisticRegression).setFeaturesPartSize(1).setInstanceStackSize(1)
        .setBlockMatrixColPartNum(3).setBlockMatrixRowPartNum(2)
      .setWeightCol("weight").setRegParam(0.8).setStandardization(true)
    val vfmodel2 = vftrainer2.fit(dataset1)
    val singletrainer2 = (new LogisticRegression).setFitIntercept(false)
      .setWeightCol("weight").setRegParam(0.8).setStandardization(true)
    val singlemodel2 = singletrainer2.fit(dataset1)

    println(s"without weight: vf coeffs: ${vfmodel2.coefficients.toLocal()}\nsingle coeffs: ${singlemodel2.coefficients}")
    assert(vfmodel2.coefficients.toLocal() ~== singlemodel2.coefficients relTol 1E-5)
  }

  test("test on data1, use weight, L2 reg = 0.8, standardize = false, without intercept") {
    val dataset1 = spark.createDataFrame(sc.parallelize(data1, 2))
    val vftrainer2 = (new VLogisticRegression).setFeaturesPartSize(4).setInstanceStackSize(5)
      .setWeightCol("weight").setRegParam(0.8).setStandardization(false)
    val vfmodel2 = vftrainer2.fit(dataset1)
    val singletrainer2 = (new LogisticRegression).setFitIntercept(false)
      .setWeightCol("weight").setRegParam(0.8).setStandardization(false)
    val singlemodel2 = singletrainer2.fit(dataset1)

    println(s"without weight: vf coeffs: ${vfmodel2.coefficients.toLocal()}\nsingle coeffs: ${singlemodel2.coefficients}")
    assert(vfmodel2.coefficients.toLocal() ~== singlemodel2.coefficients relTol 1E-5)
  }

  test("VF binary logistic regression with weighted samples") {
    val (dataset, weightedDataset) = {
      val nPoints = 20
      val coefficients = Array(-0.57997, 0.912083, -0.371077, -0.819866)
      val xMean = Array(0.1, -0.1, 0.0, 0.1)
      val xVariance = Array(0.6856, 0.1899, 3.116, 0.581)
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
    /*
    val btrainer1a = (new LogisticRegression).setFitIntercept(false)
      .setRegParam(0.0).setStandardization(true)
    val btrainer1b = (new LogisticRegression).setFitIntercept(false)
      .setWeightCol("weight").setRegParam(0.0).setStandardization(true)
    val bmodel1a0 = btrainer1a.fit(dataset)
    val bmodel1a1 = btrainer1a.fit(weightedDataset)
    val bmodel1b = btrainer1b.fit(weightedDataset)
    println(s"bmodel1a0: ${bmodel1a0.coefficients}\nbmodel1a1: ${bmodel1a1.coefficients}\nbmodel1b: ${bmodel1b.coefficients}")
    */
    val trainer1a = (new VLogisticRegression).setFeaturesPartSize(2).setInstanceStackSize(2)
      .setBlockMatrixColPartNum(2).setBlockMatrixRowPartNum(3)
      .setRegParam(0.0).setStandardization(true)
    val trainer1b = (new VLogisticRegression).setFeaturesPartSize(2).setInstanceStackSize(2)
      .setBlockMatrixColPartNum(3).setBlockMatrixRowPartNum(4)
      .setWeightCol("weight").setRegParam(0.0).setStandardization(true)
    val model1a0 = trainer1a.fit(dataset)
    val model1a1 = trainer1a.fit(weightedDataset)
    val model1b = trainer1b.fit(weightedDataset)
    println(s"model1a0: ${model1a0.coefficients.toLocal()}\nmodel1a1: ${model1a1.coefficients.toLocal()}\nmodel1b: ${model1b.coefficients.toLocal()}")
    assert(model1a0.coefficients.toLocal() !~= model1a1.coefficients.toLocal() absTol 1E-3)
    assert(model1a0.coefficients.toLocal() ~== model1b.coefficients.toLocal() absTol 1E-3)

    val strainer1b = (new LogisticRegression).setFitIntercept(false)
      .setWeightCol("weight").setRegParam(0.0).setStandardization(true)
    val smodel1b = strainer1b.fit(weightedDataset)
    println(s"s-model1b: ${smodel1b.coefficients}")
    assert(smodel1b.coefficients ~== model1b.coefficients.toLocal() absTol 1E-5)
  }
}
