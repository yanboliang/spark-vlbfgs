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

package org.apache.spark.ml.optim

import java.util.Random

import breeze.linalg.{DenseVector => BDV, norm => Bnorm}
import breeze.optimize.{DiffFunction => BDF, OWLQN => BreezeOWLQN}
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.distributed.DistributedVector
import org.apache.spark.ml.util.VUtils
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.storage.StorageLevel

class VOWLQNSuite extends SparkFunSuite with MLlibTestSparkContext {

  def rangeRandDouble(x: Double, y: Double, rand: Random): Double ={
    x + (y - x) * rand.nextDouble()
  }

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  test("quadratic-test") {
    val rand = new Random(100)

    val owlqn = new BreezeOWLQN[Int, BDV[Double]](100, 10, _ => 3.0)
    val vowqln = new VOWLQN(100, 10, (3.0, false))

    val initData: Array[Double] = Array.fill(10)(0.0)
      .map(x => rangeRandDouble(-10D, 10D, rand))

    val initBDV = new BDV(initData.clone())

    val initDV = VUtils.splitArrIntoDV(sc, initData, 5, 2)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = true)

    val bDiffFun = new BDF[BDV[Double]] {
      def calculate(x: BDV[Double]) = {
        (Bnorm((x - 3.0) :^ 2.0, 1), (x :* 2.0) :- 6.0)
      }
    }
    val vDiffFun = new VDiffFunction {
      def calculate(x: DistributedVector) = {
        val n = x.add(-3.0).norm
        (n * n, x.scale(2.0).add(-6.0).persist(StorageLevel.MEMORY_AND_DISK, eager = true))
      }
    }

    val owlqnIter = owlqn.iterations(bDiffFun, initBDV)
    val vowlqnIter = vowqln.iterations(vDiffFun, initDV)

    var bState: owlqn.State = null
    var vState: vowqln.State = null

    while (owlqnIter.hasNext) {
      bState = owlqnIter.next()
    }

    while (vowlqnIter.hasNext) {
      vState = vowlqnIter.next()
    }

    assert(vState.x.toLocal ~== Vectors.fromBreeze(bState.x) relTol 1E-3)
  }


  def testRosenbrock(m: Int, maxIter: Int, dimension: Int): Unit = {
    val rand = new Random(100)

    val sizePerPart = 3
    val numPartitions = VUtils.getNumBlocks(sizePerPart, dimension)
    println(s"TEST: m=$m, dimension=$dimension, maxIter=$maxIter, numPartitions=$numPartitions")

    val owlqn = new BreezeOWLQN[Int, BDV[Double]](maxIter, m, _ => 3.0)
    val vowlqn = new VOWLQN(maxIter, m, (3.0, false))

    val initData: Array[Double] = Array.fill(dimension)(0.0)
      .map(x => rangeRandDouble(-10D, 10D, rand))

    val initBDV = new BDV(initData.clone())

    val initDV = VUtils.splitArrIntoDV(sc, initData, sizePerPart, numPartitions).persist()

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

    val bDiffFun = new BDF[BDV[Double]] {
      def calculate(x: BDV[Double]) = calc(x)
    }
    val vDiffFun = new VDiffFunction {
      def calculate(x: DistributedVector) = {
        val r = calc(new BDV(x.toLocal.toArray))
        val rr = r._2.toArray
        (r._1, VUtils.splitArrIntoDV(sc, rr, sizePerPart, numPartitions).persist())
      }
    }

    val owlqnIter = owlqn.iterations(bDiffFun, initBDV)
    val vowlqnIter = vowlqn.iterations(vDiffFun, initDV, iter => iter % 15 == 0)

    var bState: owlqn.State = null
    var vState: vowlqn.State = null

    while (owlqnIter.hasNext) {
      bState = owlqnIter.next()
      println(s"breeze owlqn x${bState.iter}: ${bState.x}")
    }

    while (vowlqnIter.hasNext) {
      vState = vowlqnIter.next()
      println(s"v-owlqn x${vState.iter}: ${vState.x.toLocal}")
    }

    assert(vState.x.toLocal ~== Vectors.fromBreeze(bState.x) relTol 0.1)

    val l1RegArr: Array[Double] = Array.fill(dimension)(0.0)
      .map(x => rangeRandDouble(1D, 10D, rand))

    val l1RegDV = VUtils.splitArrIntoDV(sc, l1RegArr, sizePerPart, numPartitions).persist()
    initDV.persist()
    val owlqn2 = new BreezeOWLQN[Int, BDV[Double]](maxIter, m, (k:Int) => l1RegArr(k))
    val vowlqn2 = new VOWLQN(maxIter, m, l1RegDV)

    val owlqnIter2 = owlqn2.iterations(bDiffFun, initBDV)
    val vowlqnIter2 = vowlqn2.iterations(vDiffFun, initDV, iter => iter % 15 == 0)

    var bState2: owlqn2.State = null
    var vState2: vowlqn2.State = null

    while (owlqnIter2.hasNext) {
      bState2 = owlqnIter2.next()
      println(s"breeze owlqn x${bState2.iter}: ${bState2.x}")
    }

    while (vowlqnIter2.hasNext) {
      vState2 = vowlqnIter2.next()
      println(s"v-owlqn x${vState2.iter}: ${vState2.x.toLocal}")
    }

    assert(vState2.x.toLocal ~== Vectors.fromBreeze(bState2.x) relTol 0.1)
  }

  test("lbfgs-c rosenbrock example") {
    for (m <- 4 to 6) {
      for (dimension <- 4 to 6 by 2) {
        testRosenbrock(m, 20, dimension)
      }
    }
  }

}
