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

import java.util.Random

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.DistributedVector
import org.apache.spark.ml.optim.{DVDiffFunction, VectorFreeLBFGS}
import org.apache.spark.ml.util.VUtils
import org.apache.spark.sql.SparkSession

object RosenbrockExample {

  def rangeRandDouble(x: Double, y: Double, rand: Random): Double ={
    x + (y - x) * rand.nextDouble()
  }

  def runRosenbrock(
    sc: SparkContext,
    bm: Int,
    maxIter: Int,
    dimension: Int,
    partSize: Int): Unit = {

    val rand = new Random(100)

    val partNum = VUtils.getNumBlocks(partSize, dimension)
    println(s"----------run bm=$bm, dimension=$dimension, maxIter=$maxIter, partNum=${partNum}---------")

    val vf_lbfgs = new VectorFreeLBFGS(maxIter, bm)

    val initData: Array[Double] = Array.fill(dimension)(0.0)
      .map(x => rangeRandDouble(-10D, 10D, rand))

    val initDisV = VUtils.splitArrIntoDV(sc, initData, partSize, partNum).eagerPersist()

    def calc(x: BDV[Double]): (Double, BDV[Double]) = {

      var fx = 0.0
      val g = BDV.zeros[Double](x.length)

      for(i <- 0 until x.length by 2) {
        val t1 = 1.0 - x(i)
        val t2 = 10.0 * (x(i + 1) - (x(i) * x(i)))
        g(i + 1) = 20 * t2
        g(i) = -2 * (x(i) * g(i + 1) + t1)
        fx += t1 * t1 + t2 * t2
      }
      fx -> g
    }

    val vf_df = new DVDiffFunction {
      def calculate(x: DistributedVector) = {
        val r = calc(new BDV(x.toLocal.toArray))
        val rr = r._2.toArray
        (r._1, VUtils.splitArrIntoDV(sc, rr, partSize, partNum).eagerPersist())
      }
    }

    val vf_lbfgsIter = vf_lbfgs.iterations(vf_df, initDisV)

    var vf_state: vf_lbfgs.State = null

    var iterCnt = 0
    while (vf_lbfgsIter.hasNext) {
      iterCnt += 1
      System.out.print(s"press ENTER to start iter: ${iterCnt}")
      System.out.flush()
      System.in.read()
      vf_state = vf_lbfgsIter.next()
    }
    println("iteration finish.")
    println(s"result coeffs: ${vf_state.x.toLocal.toString}")
  }

  def main(args: Array[String]) = {
    val spark = SparkSession
      .builder()
      .appName("Rosenbrock Example")
      .config("spark.ui.enabled", true)
      .config("spark.ui.port", 8899)
      .getOrCreate()

    println("UI port: 8899")

    val sc = spark.sparkContext

    runRosenbrock(sc,
      bm = 4,
      maxIter = 100,
      dimension = 100,
      partSize = 10)

    sc.stop()
  }
}
