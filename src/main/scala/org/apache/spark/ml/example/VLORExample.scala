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

import org.apache.spark.ml.classification.VLogisticRegression
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
    var eagerPersist: Boolean = true
    var elasticNetParam = 1.0

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
      elasticNetParam = args(9).toDouble

      dataPath = args(10)
    } catch {
      case _: Throwable =>
        println("Param list: "
          + "maxIter dimension colsPerBlock rowsPerBlock colPartitions rowPartitions"
          + " regParam fitIntercept elasticNetParam dataPath")
        println("parameter description:" +
          "\nmaxIter          max iteration number for VLogisticRegression" +
          "\ndimension        training data dimension number" +
          "\ncolsPerBlock     column number of each block in feature block matrix" +
          "\nrowsPerBlock     row number of each block in feature block matrix" +
          "\ncolPartitions    column partition number of feature block matrix" +
          "\nrowPartitions    row partition number of feature block matrix" +
          "\nregParam         regularization parameter" +
          "\nfitIntercept     whether to train intercept, true or false" +
          "\neagerPersist     whether to eager persist distributed vector, true or false" +
          "\nelasticNetParam  elastic net parameter for regulization" +
          "\ndataPath         training data path on HDFS")

        System.exit(-1)
    }

    val spark = SparkSession
      .builder()
      .appName("VLogistic Regression Example")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setCheckpointDir("/tmp/VLogisticRegression/checkpoint")

    try {
      println(s"begin load data from $dataPath")
      val dataset: Dataset[_] = spark.read.format("libsvm")
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
        .setElasticNetParam(elasticNetParam)

      val vmodel = vtrainer.fit(dataset)

      println(s"Vector-free logistic regression coefficients: ${vmodel.coefficients}")
    } catch {
      case e: Exception =>
        e.printStackTrace()
    }finally {
      println("Press ENTER to exit.")
      System.in.read()
    }
    sc.stop()
  }

}
