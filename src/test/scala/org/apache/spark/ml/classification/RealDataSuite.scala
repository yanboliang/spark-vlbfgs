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