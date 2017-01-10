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

class VectorFreeOWLQNSuite extends SparkFunSuite with MLlibTestSparkContext {

  def rangeRandDouble(x: Double, y: Double, rand: Random): Double ={
    x + (y - x) * rand.nextDouble()
  }

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  test ("quadratic-test") {
    val rand = new Random(100)

    val owlqn = new BreezeOWLQN[Int, BDV[Double]](100, 10, _ => 3.0)
    val vf_owqln = new VectorFreeOWLQN(100, 10, (3.0, false))

    val initData: Array[Double] = Array.fill(10)(0.0)
      .map(x => rangeRandDouble(-10D, 10D, rand))

    val initBDV = new BDV(initData.clone())

    val initDisV = VUtils.splitArrIntoDV(sc, initData, 5, 2)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = true)

    val df = new BDF[BDV[Double]] {
      def calculate(x: BDV[Double]) = {
        (Bnorm((x - 3.0) :^ 2.0, 1), (x :* 2.0) :- 6.0)
      }
    }
    val vf_df = new VDiffFunction {
      def calculate(x: DistributedVector) = {
        val n = x.add(-3.0).norm
        (n * n, x.scale(2.0).add(-6.0).persist(StorageLevel.MEMORY_AND_DISK, eager = true))
      }
    }

    val owlqnIter = owlqn.iterations(df, initBDV)
    val vf_owlqnIter = vf_owqln.iterations(vf_df, initDisV)

    var state: owlqn.State = null
    var vf_state: vf_owqln.State = null

    while (owlqnIter.hasNext) {
      state = owlqnIter.next()
    }

    while (vf_owlqnIter.hasNext) {
      vf_state = vf_owlqnIter.next()
    }

    assert(vf_state.x.toLocal ~== Vectors.fromBreeze(state.x) relTol 1E-3)
  }


  def testRosenbrock(bm: Int, maxIter: Int, dimension: Int): Unit = {
    val rand = new Random(100)

    val partSize = 3
    val partNum = VUtils.getNumBlocks(partSize, dimension)
    println(s"----------test bm=$bm, dimension=$dimension, maxIter=$maxIter, partNum=${partNum}---------")

    val totalSize = dimension

    val owlqn = new BreezeOWLQN[Int, BDV[Double]](maxIter, bm, _ => 3.0)
    val vf_owlqn = new VectorFreeOWLQN(maxIter, bm, (3.0, false))

    val initData: Array[Double] = Array.fill(dimension)(0.0)
      .map(x => rangeRandDouble(-10D, 10D, rand))

    val initBDV = new BDV(initData.clone())

    val initDisV = VUtils.splitArrIntoDV(sc, initData, partSize, partNum).persist()

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

    val df = new BDF[BDV[Double]] {
      def calculate(x: BDV[Double]) = calc(x)
    }
    val vf_df = new VDiffFunction {
      def calculate(x: DistributedVector) = {
        val r = calc(new BDV(x.toLocal.toArray))
        val rr = r._2.toArray
        (r._1, VUtils.splitArrIntoDV(sc, rr, partSize, partNum).persist())
      }
    }

    val owlqnIter = owlqn.iterations(df, initBDV)
    val vf_owlqnIter = vf_owlqn.iterations(vf_df, initDisV, iter => iter % 15 == 0)

    var state: owlqn.State = null
    var vf_state: vf_owlqn.State = null

    while (owlqnIter.hasNext) {
      state = owlqnIter.next()
      println(s"br_x${state.iter}: ${state.x}")
    }

    while (vf_owlqnIter.hasNext) {
      vf_state = vf_owlqnIter.next()
      println(s"vf_x${vf_state.iter}: ${vf_state.x.toLocal}")
    }

    assert(vf_state.x.toLocal ~== Vectors.fromBreeze(state.x) relTol 0.1)

    val l1RegArr: Array[Double] = Array.fill(dimension)(0.0)
      .map(x => rangeRandDouble(1D, 10D, rand))

    val l1RegDV = VUtils.splitArrIntoDV(sc, l1RegArr, partSize, partNum).persist()
    initDisV.persist()
    val owlqn2 = new BreezeOWLQN[Int, BDV[Double]](maxIter, bm, (k:Int) => l1RegArr(k))
    val vf_owlqn2 = new VectorFreeOWLQN(maxIter, bm, l1RegDV)

    val owlqnIter2 = owlqn2.iterations(df, initBDV)
    val vf_owlqnIter2 = vf_owlqn2.iterations(vf_df, initDisV, iter => iter % 15 == 0)

    var state2: owlqn2.State = null
    var vf_state2: vf_owlqn2.State = null

    while (owlqnIter2.hasNext) {
      state2 = owlqnIter2.next()
      println(s"br2_x${state2.iter}: ${state2.x}")
    }

    while (vf_owlqnIter2.hasNext) {
      vf_state2 = vf_owlqnIter2.next()
      println(s"vf2_x${vf_state2.iter}: ${vf_state2.x.toLocal}")
    }

    assert(vf_state2.x.toLocal ~== Vectors.fromBreeze(state2.x) relTol 0.1)
  }

  test("lbfgs-c rosenbrock example") {

    for (bm <- 4 to 6) {
      for (dimension <- 4 to 6 by 2) {
        testRosenbrock(bm, 20, dimension)
      }
    }
  }

}
