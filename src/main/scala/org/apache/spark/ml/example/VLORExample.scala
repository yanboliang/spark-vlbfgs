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

package org.apache.spark.ml.example

import org.apache.spark.ml.classification.{LogisticRegression, VLogisticRegression}
import org.apache.spark.sql.{Dataset, SparkSession}

object VLORExample {

  def main(args: Array[String]) = {

    var maxIter: Int = 100

    var dimension: Int = 780
    var colsPerBlock: Int = 100
    var rowsPerBlock: Int = 10
    var colPartitions: Int = 3
    var rowPartitions: Int = 3
    var regParam: Double = 0.5
    var fitIntercept: Boolean = true

    var eagerPersist = true
    var openUI = true

    var dataPath: String = null

    try {
      maxIter = args(0).toInt

      dimension = args(1).toInt

      colsPerBlock = args(2).toInt
      rowsPerBlock = args(3).toInt
      colPartitions = args(4).toInt
      rowPartitions = args(5).toInt
      regParam = args(6).toDouble
      fitIntercept = args(7).toBoolean

      eagerPersist = args(8).toBoolean
      openUI = args(9).toBoolean

      dataPath = args(10)

    } catch {
      case _: Throwable =>
        println("Wrong params.")
        println("Params: "
          + "maxIter dimension colsPerBlock rowsPerBlock colPartitions rowPartitions"
          + " regParam fitIntercept eagerPersist openUI dataPath")
        System.exit(-1)
    }

    val spark = SparkSession
      .builder()
      .appName("VLogistic Regression Example")
      .config("spark.ui.enabled", openUI)
      .config("spark.ui.port", 8899)
      .getOrCreate()

    if (openUI) println("UI port: 8899")

    val sc = spark.sparkContext

    try {
      println(s"begin load data from ${dataPath}")
      val dataset1: Dataset[_] = spark.read.format("libsvm")
        .option("numFeatures", dimension.toString)
        .load(dataPath)

      val vtrainer = new VLogisticRegression()
        .setMaxIter(maxIter)
        .setColsPerBlock(colsPerBlock)
        .setRowsPerBlock(rowsPerBlock)
        .setColPartitions(colPartitions)
        .setRowPartitions(rowPartitions)
        .setRegParam(regParam)
        .setFitIntercept(fitIntercept)
        .setEagerPersist(eagerPersist)

      val vmodel = vtrainer.fit(dataset1)

      println(s"VLogistic regression coefficients first partition:" +
        s" ${vmodel.coefficients.vecs.first()}")
    } finally {
      println("Press ENTER to exit.")
      System.in.read()
    }
    sc.stop()
  }

}
