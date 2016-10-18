package org.apache.spark.ml.linalg

import breeze.linalg.{norm => Bnorm, DenseVector => BDV}

import org.scalatest.FunSuite

import org.apache.spark.ml.optim.VFUtils
import org.apache.spark.mllib.util.MLlibTestSparkContext

class DistributedVectorSuite extends FunSuite with MLlibTestSparkContext {

  def testVecEq(v1: BDV[Double], v2: BDV[Double]): Boolean = {
    Bnorm(v1 - v2) < 1E-8
  }

  def testDoubleEq(v1: Double, v2: Double): Boolean = {
    math.abs(v1 - v2) < 1E-8
  }

  var BV1: BDV[Double] = null
  var BV2: BDV[Double] = null
  var BV3: BDV[Double] = null
  var DV1: DistributedVector = null
  var DV2: DistributedVector = null
  var DV3: DistributedVector = null

  override def beforeAll(): Unit = {
    super.beforeAll()

    val V1Array: Array[Double] = Seq(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).toArray
    val V2Array: Array[Double] = Seq(-1.0, -2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0).toArray
    val V3Array: Array[Double] = Seq(-1.0, -2.0, -3.0, 5.0, -6.0, 7.0, 8.0, 9.0).toArray

    BV1 = new BDV(V1Array)
    BV2 = new BDV(V2Array)
    BV3 = new BDV(V3Array)

    val partSize = 3
    val partNum = VFUtils.getSplitPartNum(partSize, V1Array.length)

    DV1 = VFUtils.splitArrIntoDV(sc, V1Array, partSize, partNum).eagerPersist()
    DV2 = VFUtils.splitArrIntoDV(sc, V2Array, partSize, partNum).eagerPersist()
    DV3 = VFUtils.splitArrIntoDV(sc, V3Array, partSize, partNum).eagerPersist()
  }

  test("toLocal") {
    val localDV1 = DV1.toLocal
    assert(testVecEq(localDV1.asBreeze.toDenseVector, BV1))
  }

  test("add") {
    val local1 = DV1.add(2.0).eagerPersist().toLocal
    val local2 = DV1.add(DV2).eagerPersist().toLocal
    assert(testVecEq(local1.asBreeze.toDenseVector, BV1 + 2.0))
    assert(testVecEq(local2.asBreeze.toDenseVector, BV1 + BV2))
  }

  test("scale") {
    val local1 = DV1.scale(2.0).eagerPersist().toLocal
    assert(testVecEq(local1.asBreeze.toDenseVector, BV1 * 2.0))
  }

  test("addScalVec") {
    val res = DV1.addScalVec(3.0, DV2).eagerPersist().toLocal
    assert(testVecEq(res.asBreeze.toDenseVector, BV1 + (BV2 * 3.0)))
  }

  test("dot") {
    val dotVal = DV1.dot(DV2)
    val bDotVal = BV1.dot(BV2)
    assert(testDoubleEq(dotVal, bDotVal))
  }

  test("norm") {
    val normVal = DV1.norm
    val bnormVal = Bnorm(BV1)
    assert(testDoubleEq(normVal, bnormVal))
  }

  test("combine") {
    val combineVecLocal = DistributedVectors.combine((10.0, DV1), (100.0, DV2), (18.0, DV3)).eagerPersist().toLocal
    val bCombineVec = (BV1 * 10.0) + (BV2 * 100.0) + (BV3 * 18.0)
    assert(testVecEq(combineVecLocal.asBreeze.toDenseVector, bCombineVec))
  }

  test("zeros") {
    val res1 = VFUtils.zipRDDWithPartitionIDAndCollect(DistributedVectors.zeros(sc, 3, 2, 5).vecs)
    val res2 = Array((0, Vectors.dense(0.0, 0.0, 0.0)), (1, Vectors.dense(0.0, 0.0)))
    val res3 = Array((0, Vectors.dense(0.0, 0.0, 0.0)), (1, Vectors.dense(0.0, 0.1)))
//    assert(VFUtilsSuite.arrEq(res1, res2))
//    assert(!VFUtilsSuite.arrEq(res1, res3))
  }

}
