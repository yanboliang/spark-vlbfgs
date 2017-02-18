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

import org.apache.spark.ml.classification.MyLogisticRegression
import org.apache.spark.sql.{Dataset, SparkSession}

object LORExample2 {

  def main(args: Array[String]) = {

    var maxIter: Int = 100

    var dimension: Int = 780
    var regParam: Double = 0.5
    var fitIntercept: Boolean = true
    var elasticNetParam = 1.0

    var dataPath: String = null

    try {
      maxIter = args(0).toInt

      dimension = args(1).toInt

      regParam = args(2).toDouble
      fitIntercept = args(3).toBoolean
      elasticNetParam = args(4).toDouble

      dataPath = args(5)
    } catch {
      case _: Throwable =>
        println("Param list: "
          + "maxIter dimension"
          + " regParam fitIntercept elasticNetParam dataPath")
        println("parameter description:" +
          "\nmaxIter          max iteration number for VLogisticRegression" +
          "\ndimension        training data dimension number" +
          "\nregParam         regularization parameter" +
          "\nfitIntercept     whether to train intercept, true or false" +
          "\nelasticNetParam  elastic net parameter for regulization" +
          "\ndataPath         training data path on HDFS")

        System.exit(-1)
    }

    val spark = SparkSession
      .builder()
      .appName("LOR for testing")
      .getOrCreate()

    val sc = spark.sparkContext

    try {
      println(s"begin load data from $dataPath")
      val dataset: Dataset[_] = spark.read.format("libsvm")
        .option("numFeatures", dimension.toString)
        .load(dataPath)

      val trainer = new MyLogisticRegression()
        .setMaxIter(maxIter)
        .setRegParam(regParam)
        .setFitIntercept(fitIntercept)
        .setElasticNetParam(elasticNetParam)

      val model = trainer.fit(dataset)

      println(s"LOR done, coeffs non zeros: ${model.coefficients.numNonzeros}")
    } catch {
      case e: Exception =>
        e.printStackTrace()
    }finally {
      // println("Press ENTER to exit.")
      // System.in.read()
    }
    sc.stop()
  }

}
