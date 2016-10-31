package org.apache.spark.ml.optim

import java.util.Random

import breeze.linalg.{norm => Bnorm, DenseVector => BDV}
import breeze.optimize.{DiffFunction => BDF, LBFGS => BreezeLBFGS}

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.linalg.DistributedVector
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTestingUtils}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext

class VectorFreeLBFGSSuite extends SparkFunSuite with MLlibTestSparkContext {

  def rangeRandDouble(x: Double, y: Double, rand: Random): Double = x + (y - x) * rand.nextDouble()

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  test ("quadratic-test") {
    val rand = new Random(100)

    val lbfgs = new BreezeLBFGS[BDV[Double]](100, 4)
    val vf_lbfgs = new VectorFreeLBFGS(100, 10)

    val initData: Array[Double] = Array.fill(10)(0.0).map(x => rangeRandDouble(-10D, 10D, rand))

    val initBDV = new BDV(initData.clone())

    val initDisV = VFUtils.splitArrIntoDV(sc, initData, 5, 2)

    val df = new BDF[BDV[Double]] {
      def calculate(x: BDV[Double]) = {
        (Bnorm((x - 3.0) :^ 2.0, 1), (x :* 2.0) :- 6.0)
      }
    }
    val vf_df = new DVDiffFunction {
      def calculate(x: DistributedVector) = {
        val n = x.add(-3.0).norm
        (n * n, x.scale(2.0).add(-6.0))
      }
    }

    val lbfgsIter = lbfgs.iterations(df, initBDV)
    val vf_lbfgsIter = vf_lbfgs.iterations(vf_df, initDisV)

    var state: lbfgs.State = null
    var vf_state: vf_lbfgs.State = null

    while (lbfgsIter.hasNext) {
      state = lbfgsIter.next()
    }

    while (vf_lbfgsIter.hasNext) {
      vf_state = vf_lbfgsIter.next()
    }

    assert(vf_state.x.toLocal ~== Vectors.fromBreeze(state.x) relTol 1E-3)
  }


  def testRosenbrock(bm: Int, maxIter: Int, dimension: Int): Unit = {
    val rand = new Random(100)

    val partSize = 3
    val partNum = VFUtils.getSplitPartNum(partSize, dimension)
    println(s"----------test bm=$bm, dimension=$dimension, maxIter=$maxIter, partNum=${partNum}---------")

    val totalSize = dimension

    val lbfgs = new BreezeLBFGS[BDV[Double]](maxIter, bm)
    val vf_lbfgs = new VectorFreeLBFGS(maxIter, bm)

    // FIX: dimension == 4, hasNext != , when iter == 94
    // FIX: dimension == 10, hasNext != , when iter == 99

    val initData: Array[Double] = Array.fill(dimension)(0.0)
      .map(x => rangeRandDouble(-10D, 10D, rand))

    val initBDV = new BDV(initData.clone())

    val initDisV = VFUtils.splitArrIntoDV(sc, initData, partSize, partNum).eagerPersist()

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
    val vf_df = new DVDiffFunction {
      def calculate(x: DistributedVector) = {
        val r = calc(new BDV(x.toLocal.toArray))
        val rr = r._2.toArray
        (r._1, VFUtils.splitArrIntoDV(sc, rr, partSize, partNum).eagerPersist())
      }
    }

    val lbfgsIter = lbfgs.iterations(df, initBDV)
    val vf_lbfgsIter = vf_lbfgs.iterations(vf_df, initDisV)

    var state: lbfgs.State = null
    var vf_state: vf_lbfgs.State = null

    while (lbfgsIter.hasNext) {
      state = lbfgsIter.next()
    }

    while (vf_lbfgsIter.hasNext) {
      vf_state = vf_lbfgsIter.next()
    }

    assert(vf_state.x.toLocal ~== Vectors.fromBreeze(state.x) relTol 0.1)
  }

  test("lbfgs-c rosenbrock example") {

    for (bm <- 4 to 10) {
      for (dimension <- 4 to 6 by 2) {
        testRosenbrock(bm, 10, dimension)
        // println(s"bm=$bm, dimension=$dimension test passed.")
      }
    }
    // println("test done.(useVector-free true)")
  }

}
