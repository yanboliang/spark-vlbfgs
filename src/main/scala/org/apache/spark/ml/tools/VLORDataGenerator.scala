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

package org.apache.spark.ml.tools

import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object VLORDataGenerator {

  def genCoeffs(seed: Int): Array[Double] = {
    val coeffsRnd = new Random(seed)
    val coeffs = new Array[Double](1000000)
    var i = 0
    while(i < coeffs.length) {
      coeffs(i) = coeffsRnd.nextGaussian()
      i += 1
    }
    coeffs
  }

  def genRecord(rnd: Random, coeffs: Array[Double],
                dimension: Int, validFeatureNumPerRecord: Int): String = {

    val idxBuf = ArrayBuffer[Int](validFeatureNumPerRecord)
    var i = 0
    while (i < validFeatureNumPerRecord) {
      idxBuf += rnd.nextInt(dimension)
      i += 1
    }

    val idxArr = idxBuf.distinct.sortWith(_ < _).toArray
    val valArr = new Array[Double](idxArr.length)

    var wx = 0.0
    i = 0
    while (i < idxArr.length) {
      valArr(i) = rnd.nextGaussian()
      wx += valArr(i) * coeffs(idxArr(i) % coeffs.length)
      i += 1
    }

    if (rnd.nextDouble() < 0.1) wx = -wx // generate 10% wrong data.
    val label = if (wx < 0) 0 else 1

    val resBuf = new StringBuilder
    resBuf ++= "%d".format(label)

    i = 0
    while (i < idxArr.length) {
      resBuf ++= " %d:%.3f".format(idxArr(i) + 1, valArr(i))
      i += 1
    }
    resBuf.toString()
  }

  def main(args: Array[String]) = {

    var dimension = 100
    var numSplits = 100
    var numRecordsPerSplit = 100
    var validFeatureNumPerRecord = 10
    var seed = 7
    var outputPath = ""

    try {

      dimension = args(0).toInt
      numSplits = args(1).toInt
      numRecordsPerSplit = args(2).toInt
      validFeatureNumPerRecord = args(3).toInt
      seed = args(4).toInt
      outputPath = args(5)

    } catch {
      case _: Throwable =>
        println("Params: dimension numSplits numRecordsPerSplit validFeatureNumPerRecord seed outputPath")
        System.exit(-1)
    }

    val spark = SparkSession
      .builder()
      .appName("VLOR Data Generator")
      .getOrCreate()

    val sc = spark.sparkContext

    sc.parallelize(0 until numSplits, numSplits)
      .mapPartitions { iter =>

        val pid = iter.next()

        val coeffs = genCoeffs(seed)

        val rnd = new Random(seed + pid)

        new Iterator[String] {

          private var cnt = 0

          override def hasNext: Boolean = {
            cnt < numRecordsPerSplit
          }

          override def next(): String = {
            cnt += 1
            genRecord(rnd, coeffs, dimension, validFeatureNumPerRecord)
          }
        }
      }.saveAsTextFile(outputPath)

    sc.stop()
  }

}
