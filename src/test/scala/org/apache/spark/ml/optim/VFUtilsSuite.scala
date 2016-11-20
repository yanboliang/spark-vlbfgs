package org.apache.spark.ml.optim

import org.apache.spark.SparkFunSuite
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.util.MLlibTestSparkContext

import scala.reflect.ClassTag

/**
  * Created by ThinkPad on 2016/10/7.
  */
class VFUtilsSuite extends SparkFunSuite with MLlibTestSparkContext {

  import VFUtilsSuite._

  override def beforeAll(): Unit = {
    super.beforeAll()
    testZipIdxRdd = sc.parallelize(Seq((0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4), (2,5)))
      .partitionBy(new DistributedVectorPartitioner(3)).map(_._2)
  }
  var testZipIdxRdd: RDD[Int] = null

  test ("triple equal") {
    val a1 = Array(1.0, 2.0)
    val a2 = Array(1.0, 2.0)

    assert(! (a1 == a2))
    assert(a1 === a2)
  }

  test ("getSplitPartNum") {
    assert(VFUtils.getSplitPartNum(2, 3) == 2)
    assert(VFUtils.getSplitPartNum(2, 4) == 2)
    assert(VFUtils.getSplitPartNum(2, 5) == 3)
    assert(VFUtils.getSplitPartNum(2, 6) == 3)
    assert(VFUtils.getSplitPartNum(3, 4) == 2)
    assert(VFUtils.getSplitPartNum(3, 5) == 2)
    assert(VFUtils.getSplitPartNum(3, 6) == 2)
    assert(VFUtils.getSplitPartNum(3, 7) == 3)
  }

  test ("splitArrIntoDV") {
    val arrs = VFUtils.splitArrIntoDV(sc, Array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), 3, 3).vecs.collect()
    assert(arrEq(arrs(0).toArray, Array(1.0, 2.0, 3.0)))
    assert(arrEq(arrs(1).toArray, Array(4.0, 5.0, 6.0)))
    assert(arrEq(arrs(2).toArray, Array(7.0)))
  }

  test ("splitSparseVector") {
    val sv1 = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0, 6.0).toSparse
    val splist = VFUtils.splitSparseVector(sv1, 2)
    assert(
        splist(0) == Vectors.dense(1.0, 2.0) &&
        splist(1) == Vectors.dense(3.0, 4.0) &&
        splist(2) == Vectors.dense(5.0, 6.0)
    )

    val sv2 = Vectors.dense(1.0, 2.0, 3.0, 4.0, 5.0).toSparse
    val splist2 = VFUtils.splitSparseVector(sv2, 3)
    assert(
        splist2(0) == Vectors.dense(1.0, 2.0, 3.0) &&
        splist2(1) == Vectors.dense(4.0, 5.0)
    )
  }


  test ("computePartitionStartIndices") {
    val inds = VFUtils.computePartitionStartIndices(testZipIdxRdd).map(_.toInt)
    assert(arrEq(inds, Array(2, 3, 5)))
  }

  test ("zipRDDWithIndex") {
    val res = VFUtils.zipRDDWithIndex(Array(2L, 3L, 5L), testZipIdxRdd).collect().map(x => (x._1.toInt, x._2))
    assert(arrEq(res, Array((0, 1), (1, 2), (2, 1), (3, 2), (4, 3), (5, 1), (6, 2), (7, 3), (8, 4), (9, 5))))
  }

  test ("vertcatSparseVectorIntoCSRMatrix") {
    assert(VFUtils.vertcatSparseVectorIntoCSRMatrix(Array(
      Vectors.dense(1.0, 2.0, 3.0).toSparse,
      Vectors.dense(8.0, 7.0, 9.0).toSparse
    )).toDense == Matrices.dense(2, 3, Array(1.0, 8.0, 2.0, 7.0, 3.0, 9.0)))
  }

  def testBlockMatrixHorzZipVecFunc(rows: Int, cols: Int, rowsPerPart: Int, colsPerPart: Int) = {
    val arrMatrix = Array.tabulate(rows * cols)(idx => {
      val rowIdx = idx % rows
      val colIdx = idx / rows
      ((rowIdx, colIdx), SparseMatrix.fromCOO(1, 2, Array((0, 0, rowIdx.toDouble), (0, 1, colIdx.toDouble))))
    })
    val gridPartitioner = GridPartitionerV2(rows, cols, rowsPerPart, colsPerPart)
    val blockMatrix = sc.parallelize(arrMatrix).partitionBy(gridPartitioner)
    val arrVec = Array.tabulate(cols)(idx => idx.toDouble)
    val dvec = VFUtils.splitArrIntoDV(sc, arrVec, 1, cols)
    val f = (sv: SparseMatrix, v: Vector) => {
      (sv(0, 0), sv(0, 1), v(0))
    }
    val res0 = VFUtils.blockMatrixHorzZipVec(blockMatrix, dvec, gridPartitioner, f)
      .map(x => (x._1._1, x._2))
    val res = res0.map(v => (v._1, v._2._1.toInt, v._2._2.toInt, v._2._3.toInt))
      .collect().map{ v =>
      assert(v._1 == v._2 && v._3 == v._4)
      (v._2, v._3)
    }.sortBy(v => v._1 + v._2 * 1000)
    // println(s"arr res: ${res.mkString(",")}")
    assert(res === Array.tabulate(rows * cols)(idx => {
      val rowIdx = idx % rows
      val colIdx = idx / rows
      (rowIdx, colIdx)
    }))
  }

  def testBlockMatrixVertZipVecFunc(rows: Int, cols: Int, rowsPerPart: Int, colsPerPart: Int) = {
    val arrMatrix = Array.tabulate(rows * cols)(idx => {
      val rowIdx = idx % rows
      val colIdx = idx / rows
      ((rowIdx, colIdx), SparseMatrix.fromCOO(1, 2, Array((0, 0, rowIdx.toDouble), (0, 1, colIdx.toDouble))))
    })
    val gridPartitioner = GridPartitionerV2(rows, cols, rowsPerPart, colsPerPart)
    val blockMatrix = sc.parallelize(arrMatrix).partitionBy(gridPartitioner)
    val arrVec = Array.tabulate(rows)(idx => idx.toDouble)
    val dvec = VFUtils.splitArrIntoDV(sc, arrVec, 1, rows)
    val f = (sv: SparseMatrix, v: Vector) => {
      (sv(0, 0), sv(0, 1), v(0))
    }
    val res0 = VFUtils.blockMatrixVertZipVec(blockMatrix, dvec, gridPartitioner, f)
      .map(x => (x._1._2, x._2))
    val res = res0.map(v => (v._1, v._2._1.toInt, v._2._2.toInt, v._2._3.toInt))
      .collect().map{ v =>
      assert(v._1 == v._3 && v._2 == v._4)
      (v._2, v._3)
    }.sortBy(v => v._1 + v._2 * 1000)
    // println(s"arr res: ${res.mkString(",")}")
    assert(res === Array.tabulate(rows * cols)(idx => {
      val rowIdx = idx % rows
      val colIdx = idx / rows
      (rowIdx, colIdx)
    }))
  }

  test ("blockMatrixHorzZipVec") {
    testBlockMatrixHorzZipVecFunc(5, 4, 2, 3)
    testBlockMatrixHorzZipVecFunc(8, 6, 2, 3)
    testBlockMatrixHorzZipVecFunc(3, 5, 3, 5)
    testBlockMatrixHorzZipVecFunc(15, 4, 6, 1)
    testBlockMatrixHorzZipVecFunc(15, 3, 6, 2)

    testBlockMatrixHorzZipVecFunc(4, 5, 3, 2)
    testBlockMatrixHorzZipVecFunc(6, 8, 3, 2)
    testBlockMatrixHorzZipVecFunc(5, 3, 5, 3)
    testBlockMatrixHorzZipVecFunc(4, 15, 1, 6)
    testBlockMatrixHorzZipVecFunc(3, 15, 2, 6)
  }
  test ("blockMatrixVertZipVec") {
    testBlockMatrixVertZipVecFunc(5, 4, 2, 3)
    testBlockMatrixVertZipVecFunc(8, 6, 2, 3)
    testBlockMatrixVertZipVecFunc(3, 5, 3, 5)
    testBlockMatrixVertZipVecFunc(15, 4, 6, 1)
    testBlockMatrixVertZipVecFunc(15, 3, 6, 2)

    testBlockMatrixVertZipVecFunc(4, 5, 3, 2)
    testBlockMatrixVertZipVecFunc(6, 8, 3, 2)
    testBlockMatrixVertZipVecFunc(5, 3, 5, 3)
    testBlockMatrixVertZipVecFunc(4, 15, 1, 6)
    testBlockMatrixVertZipVecFunc(3, 15, 2, 6)
  }
}
object VFUtilsSuite {
  def arrEq[T: ClassTag](arr: Array[T], arr2: Array[T]): Boolean = {
    arr.length == arr2.length && arr.zip(arr2).forall(x => x._1 == x._2)
  }
}
