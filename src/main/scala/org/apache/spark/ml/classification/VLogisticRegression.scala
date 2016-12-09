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

package org.apache.spark.ml.classification

import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{DistributedVector => DV, DistributedVectors => DVs, _}
import org.apache.spark.ml.optim.{DVDiffFunction, VectorFreeLBFGS}
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.{MultivariateOnlineSummarizer, OptimMultivariateOnlineSummarizer}
import org.apache.spark.SparkException
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.DoubleAccumulator

/**
 * Params for vector-free logistic regression.
 */
private[classification] trait VLogisticRegressionParams
  extends ProbabilisticClassifierParams with HasRegParam with HasMaxIter
    with HasTol with HasStandardization with HasWeightCol with HasThreshold
    with HasFitIntercept

object VLogisticRegression {
  val storageLevel = StorageLevel.MEMORY_AND_DISK
}

/**
 * Logistic regression.
 */
class VLogisticRegression(override val uid: String)
  extends ProbabilisticClassifier[Vector, VLogisticRegression, VLogisticRegressionModel]
    with VLogisticRegressionParams with Logging {

  import VLogisticRegression._

  def this() = this(Identifiable.randomUID("vector-free-logreg"))

  // colsPerBlock
  // This equals to the number of partitions of coefficients.
  val colsPerBlock: IntParam = new IntParam(this, "colsPerBlock",
    "Number of columns of each block matrix.", ParamValidators.gt(0))
  setDefault(colsPerBlock -> 10000)

  def setColsPerBlock(value: Int): this.type = set(colsPerBlock, value)

  // rowsPerBlock
  val rowsPerBlock: IntParam = new IntParam(this, "rowsPerBlock",
    "Number of rows of each block matrix.", ParamValidators.gt(0))
  setDefault(rowsPerBlock -> 10000)

  def setRowsPerBlock(value: Int): this.type = set(rowsPerBlock, value)

  // rowPartitions
  val rowPartitions: IntParam = new IntParam(this, "rowPartitions",
    "Number of partitions in the row direction.", ParamValidators.gt(0))
  setDefault(rowPartitions -> 10)

  def setRowPartitions(value: Int): this.type = set(rowPartitions, value)

  // colPartitions
  val colPartitions: IntParam = new IntParam(this, "colPartitions",
    "Number of partitions in the column direction.", ParamValidators.gt(0))
  setDefault(colPartitions -> 10)

  def setColPartitions(value: Int): this.type = set(colPartitions, value)

  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
  setDefault(fitIntercept -> true)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  override protected[spark] def train(dataset: Dataset[_]): VLogisticRegressionModel = {

    val sc = dataset.sparkSession.sparkContext
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)).cast(DoubleType), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }

    val numFeatures: Long = instances.first().features.size
    val localColsPerBlock: Int = $(colsPerBlock)
    val localRowsPerBlock: Int = $(rowsPerBlock)
    // Get number of blocks in column direction.
    val colBlocks: Int = VUtils.getNumBlocks(localColsPerBlock, numFeatures)

    // 1. features statistics
    val featuresSummarizer: RDD[(Int, OptimMultivariateOnlineSummarizer)] = {
      val features = instances.flatMap {
        case Instance(label, weight, features) =>
          val featuresArray = VUtils.splitSparseVector(features.toSparse, localColsPerBlock)
          featuresArray.zipWithIndex.map { case (partFeatures, partId) =>
            (partId, (partFeatures, weight))
          }
      }
      val seqOp = (s: OptimMultivariateOnlineSummarizer, partFeatures: (Vector, Double)) =>
        s.add(partFeatures._1, partFeatures._2)
      val comOp = (s1: OptimMultivariateOnlineSummarizer, s2: OptimMultivariateOnlineSummarizer) =>
        s1.merge(s2)

      features.aggregateByKey(
        new OptimMultivariateOnlineSummarizer(OptimMultivariateOnlineSummarizer.varianceMask),
        new DistributedVectorPartitioner(colBlocks)
      )(seqOp, comOp).persist(storageLevel)
    }

    // featuresStd is a distributed vector.
    val featuresStd: DV = VUtils.kvRDDToDV(
      featuresSummarizer.mapValues { summarizer =>
        Vectors.dense(summarizer.variance.toArray.map(math.sqrt))
      }, localColsPerBlock, colBlocks, numFeatures).eagerPersist(storageLevel)

    // println(s"dv feature std: ${featuresStd.toLocal().toString}")

    featuresSummarizer.unpersist()

    val labelAndWeightRDD: RDD[(Double, Double)] = instances.map { instance =>
      (instance.label, instance.weight)
    }.persist(storageLevel)

    // 3. statistic each partition size.

    val partitionSizes: Array[Long] = VUtils.computePartitionSize(labelAndWeightRDD)
    val numInstances = partitionSizes.sum
    val rowBlocks = VUtils.getNumBlocks(localRowsPerBlock, numInstances)

    val labelAndWeight: RDD[(Array[Double], Array[Double])] =
      VUtils.zipRDDWithIndex(partitionSizes, labelAndWeightRDD)
        .map { case (rowIdx: Long, (label: Double, weight: Double)) =>
          val rowBlockIdx = (rowIdx / localRowsPerBlock).toInt
          val inBlockIdx = (rowIdx % localRowsPerBlock).toInt
          (rowBlockIdx, (inBlockIdx, label, weight))
        }
        .groupByKey(new DistributedVectorPartitioner(rowBlocks))
        .map { case (blockRowIdx: Int, iter: Iterable[(Int, Double, Double)]) =>
          val tupleArr = iter.toArray.sortWith(_._1 < _._1)
          val labelArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._2)
          val weightArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._3)
          (labelArr, weightArr)
        }.persist(storageLevel)

    val weightSum = labelAndWeight.map(_._2.sum).sum()

    // println(s"weightSum: ${weightSum}")

    val labelSummarizer = {
      val seqOp = (c: MultiClassSummarizer, instance: Instance)
        => c.add(instance.label, instance.weight)

      val combOp = (c1: MultiClassSummarizer, c2: MultiClassSummarizer)
        => c1.merge(c2)

      instances.treeAggregate(new MultiClassSummarizer)(seqOp, combOp, 2)
    }

    val histogram = labelSummarizer.histogram


    // column-majar grid partitioner.
    var localRowPartitions = $(rowPartitions)
    var localColPartitions = $(colPartitions)
    if (localRowPartitions > rowBlocks) localRowPartitions = rowBlocks
    if (localColPartitions > colBlocks) localColPartitions = colBlocks

    val gridPartitioner = new GridPartitionerV2(
      rowBlocks, colBlocks,
      rowBlocks / localRowPartitions,
      colBlocks / localColPartitions
    )

    // 5. pack features into blcok matrix
    val rawFeatures: RDD[((Int, Int), SparseMatrix)] =
      VUtils.zipRDDWithIndex(partitionSizes, instances)
        .flatMap { case (rowIdx: Long, Instance(label, weight, features)) =>
          val rowBlockIdx = (rowIdx / localRowsPerBlock).toInt
          val inBlockIdx = (rowIdx % localRowsPerBlock).toInt
          val featuresArray = VUtils.splitSparseVector(features.toSparse, localColsPerBlock)
          featuresArray.zipWithIndex.map { case (partFeatures, partId) =>
            // partId corresponds to colBlockIdx
            ((rowBlockIdx, partId), (inBlockIdx, partFeatures))
          }
        }
        .groupByKey(gridPartitioner)
        .map { case ((rowBlockIdx: Int, colBlockIdx: Int), iter: Iterable[(Int, SparseVector)]) =>
          val vecs = iter.toArray.sortWith(_._1 < _._1).map(_._2)
          val matrix = VUtils.vertcatSparseVectorIntoCSRMatrix(vecs)
          ((rowBlockIdx, colBlockIdx), matrix)
        }

    val features: RDD[((Int, Int), SparseMatrix)] = VUtils.blockMatrixHorzZipVec(
        rawFeatures, featuresStd, gridPartitioner,
        (blockCoords: (Int, Int), sm: SparseMatrix, partFeatureStdVector: Vector) => {
          val partFeatureStdArr = partFeatureStdVector.asInstanceOf[DenseVector].values
          val arrBuf = new ArrayBuffer[(Int, Int, Double)]()
          sm.foreachActive { case (i: Int, j: Int, value: Double) =>
            if (partFeatureStdArr(j) != 0 && value != 0) {
              arrBuf.append((j, i, value / partFeatureStdArr(j)))
            }
          }
          SparseMatrix.fromCOO(sm.numCols, sm.numRows, arrBuf).transpose
        }).persist(storageLevel)

    val localFitIntercept = $(fitIntercept)
    val costFun = new VBinomialLogisticCostFun(
      numFeatures,
      localColsPerBlock,
      numInstances,
      localRowsPerBlock,
      features,
      gridPartitioner,
      labelAndWeight,
      weightSum,
      rowBlocks,
      colBlocks,
      $(standardization),
      featuresStd,
      $(regParam),
      localFitIntercept)

    val optimizer = new VectorFreeLBFGS($(maxIter), 10, $(tol))

    val initCoeffs: DV = if (localFitIntercept) {
      /*
        For binary logistic regression, when we initialize the coefficients as zeros,
        it will converge faster if we initialize the intercept such that
        it follows the distribution of the labels.

        {{{
          P(0) = 1 / (1 + \exp(b)), and
          P(1) = \exp(b) / (1 + \exp(b))
        }}}, hence
        {{{
          b = \log{P(1) / P(0)} = \log{count_1 / count_0}
        }}}
      */
      val initIntercept = math.log(
        histogram(1) / histogram(0))
      DVs.zeros(
        sc, localColsPerBlock, colBlocks, numFeatures + 1, initIntercept)
    } else {
      DVs.zeros(
        sc, localColsPerBlock, colBlocks, numFeatures)
    }.eagerPersist()

    val states = optimizer.iterations(costFun, initCoeffs)

    var state: optimizer.State = null
    // var iterCnt = 0
    while (states.hasNext) {
      // iterCnt += 1
      // println(s"LBFGS iter $iterCnt")
      state = states.next()
    }
    if (state == null) {
      val msg = s"${optimizer.getClass.getName} failed."
      logError(msg)
      throw new SparkException(msg)
    }

    val rawCoeffs = state.x // `x` already persisted.

    // println(s"rawCoeffs: ${rawCoeffs.toLocal.toString}")

    val interceptValAccu = sc.doubleAccumulator
    val coeffs = rawCoeffs.zipPartitionsWithIndex(
      featuresStd, rawCoeffs.sizePerPart, numFeatures
    ) {
      case (pid: Int, partCoeffs: Vector, partFeatursStd: Vector) =>
        val partFeatursStdArr = partFeatursStd.toDense.toArray
        val resArrSize =
          if (localFitIntercept && pid == colBlocks - 1) partCoeffs.size - 1
          else partCoeffs.size
        val res = Array.fill(resArrSize)(0.0)
        partCoeffs.foreachActive { case (idx: Int, value: Double) =>
          val isIntercept =
            (localFitIntercept && pid == colBlocks - 1 && idx == partCoeffs.size - 1)
          if (!isIntercept) {
            if (partFeatursStdArr(idx) != 0.0) {
              res(idx) = value / partFeatursStdArr(idx)
            }
          } else {
            interceptValAccu.add(value)
          }
        }
        Vectors.dense(res)
    }.eagerPersist()
    val interceptVal = interceptValAccu.value
    val model = copyValues(new VLogisticRegressionModel(uid, coeffs, interceptVal))
    model
  }

  override def copy(extra: ParamMap): VLogisticRegression = defaultCopy(extra)
}

private[ml] class VBinomialLogisticCostFun(
    _numFeatures: Long,
    _colsPerBlock: Int,
    _numInstances: Long,
    _rowsPerBlock: Int,
    _features: RDD[((Int, Int), SparseMatrix)],
    _gridPartitioner: GridPartitionerV2,
    _labelAndWeight: RDD[(Array[Double], Array[Double])],
    _weightSum: Double,
    _rowBlocks: Int,
    _colBlocks: Int,
    _standardization: Boolean,
    _featuresStd: DV,
    _regParamL2: Double,
    _fitIntercept: Boolean) extends DVDiffFunction {

  import VLogisticRegression._

  // Calculates both the value and the gradient at a point
  override def calculate(coeffs: DV): (Double, DV) = {

    val numFeatures: Long = _numFeatures
    val colsPerBlock: Int = _colsPerBlock
    val numInstances: Long = _numInstances
    val rowsPerBlock: Int = _rowsPerBlock
    val features: RDD[((Int, Int), SparseMatrix)] = _features
    val gridPartitioner: GridPartitionerV2 = _gridPartitioner
    val labelAndWeight: RDD[(Array[Double], Array[Double])] = _labelAndWeight
    val weightSum: Double = _weightSum
    val rowBlocks: Int = _rowBlocks
    val colBlocks: Int = _colBlocks
    val standardization: Boolean = _standardization
    val featuresStd: DV = _featuresStd
    val regParamL2: Double = _regParamL2
    val fitIntercept = _fitIntercept

    val lossAccu: DoubleAccumulator = features.sparkContext.doubleAccumulator
    val multipliers: RDD[Vector] = VUtils.blockMatrixHorzZipVec(
      features, coeffs, gridPartitioner,
      (blockCoords: (Int, Int), matrix: SparseMatrix, partCoeffs: Vector) => {
        val intercept = if (fitIntercept && blockCoords._2 == colBlocks - 1) {
          // Get intercept from the last element of coefficients vector
          partCoeffs(partCoeffs.size - 1)
        } else 0.0
        val partMarginArr = Array.fill[Double](matrix.numRows)(intercept)
        matrix.foreachActive { case (i: Int, j: Int, v: Double) =>
          partMarginArr(i) += (partCoeffs(j) * v)
        }
        Vectors.dense(partMarginArr).compressed
      }
    ).map { case ((rowBlockIdx: Int, colBlockIdx: Int), partMargins: Vector) =>
      (rowBlockIdx, partMargins)
    }.aggregateByKey(new VectorSummarizer, new DistributedVectorPartitioner(rowBlocks))(
      (s, v) => s.add(v),
      (s1, s2) => s1.merge(s2)
    ).zip(labelAndWeight)
      .map {
        case ((rowBlockIdx: Int, marginSummarizer: VectorSummarizer),
        (labelArr: Array[Double], weightArr: Array[Double])) =>
          val marginArr = marginSummarizer.toArray
          var lossSum = 0.0
          val multiplierArr = Array.fill(marginArr.length)(0.0)
          var i = 0
          while (i < marginArr.length) {
            val label = labelArr(i)
            val margin = -1.0 * marginArr(i)
            val weight = weightArr(i)
            if (label > 0) {
              lossSum += weight * MLUtils.log1pExp(margin)
            } else {
              lossSum += weight * (MLUtils.log1pExp(margin) - margin)
            }
            multiplierArr(i) = weight * (1.0 / (1.0 + math.exp(margin)) - label)
            i = i + 1
          }
          lossAccu.add(lossSum)
          Vectors.dense(multiplierArr)
      }

    val multipliersDV: DV = new DV(multipliers, rowsPerBlock, rowBlocks, numInstances)
      .eagerPersist(storageLevel)

    val lossSum = lossAccu.value / weightSum

    val grad: RDD[Vector] = VUtils.blockMatrixVertZipVec(features, multipliersDV, gridPartitioner,
      (blockCoords: (Int, Int), matrix: SparseMatrix, partMultipliers: Vector) => {
        val partGradArr = if (fitIntercept && blockCoords._2 == colBlocks - 1) {
          val arr = Array.fill[Double](matrix.numCols + 1)(0.0)
          arr(arr.length - 1) = partMultipliers.toArray.sum
          arr
        } else {
          Array.fill[Double](matrix.numCols)(0.0)
        }
        matrix.foreachActive { case (i: Int, j: Int, v: Double) =>
          partGradArr(j) += (partMultipliers(i) * v)
        }
        Vectors.dense(partGradArr).compressed
      }
    ).map { case ((rowBlockIdx: Int, colBlockIdx: Int), partGrads: Vector) =>
      (colBlockIdx, partGrads)
    }.aggregateByKey(new VectorSummarizer, new DistributedVectorPartitioner(colBlocks))(
      (s, v) => s.add(v),
      (s1, s2) => s1.merge(s2)
    ).map {
      case (colBlockIdx: Int, partGradsSummarizer: VectorSummarizer) =>
        val partGradsArr = partGradsSummarizer.toArray
        var i = 0
        while (i < partGradsArr.length) {
          partGradsArr(i) /= weightSum
          i += 1
        }
        Vectors.dense(partGradsArr)
      }

    val gradDV: DV = new DV(grad, colsPerBlock, colBlocks,
      if (fitIntercept) numFeatures + 1 else numFeatures
    ).eagerPersist(storageLevel)

    // println(s"gradDV: ${gradDV.toLocal.toString}")

    // compute regularization for grad & objective value
    val lossRegAccu = gradDV.vecs.sparkContext.doubleAccumulator
    val gradDVWithReg: DV = if (standardization) {
      gradDV.zipPartitionsWithIndex(coeffs) {
        case (pid: Int, partGrads: Vector, partCoeffs: Vector) =>
          var lossReg = 0.0
          val partGradArr = partGrads.toArray
          val res = Array.fill[Double](partGrads.size)(0.0)
          partCoeffs.foreachActive { case (i: Int, value: Double) =>
            val isIntercept = (fitIntercept && pid == colBlocks - 1 && i == partCoeffs.size - 1)
            if (!isIntercept) {
              res(i) = partGradArr(i) + regParamL2 * value
              lossReg += (value * value)
            } else {
              res(i) = partGradArr(i)
            }
          }
          lossRegAccu.add(lossReg)
          Vectors.dense(res)
      }
    } else {
      gradDV.zipPartitionsWithIndex(coeffs, featuresStd) {
        case (pid: Int, partGrads: Vector, partCoeffs: Vector, partFeaturesStds: Vector) =>
          var lossReg = 0.0
          val partGradArr = partGrads.toArray
          val partFeaturesStdArr = partFeaturesStds.toArray
          val res = Array.fill[Double](partGradArr.length)(0.0)
          partCoeffs.foreachActive { case (i: Int, value: Double) =>
            val isIntercept = (fitIntercept && pid == colBlocks - 1 && i == partCoeffs.size - 1)
            if (!isIntercept) {
              if (partFeaturesStdArr(i) != 0.0) {
                val temp = value / (partFeaturesStdArr(i) * partFeaturesStdArr(i))
                res(i) = partGradArr(i) + regParamL2 * temp
                lossReg += (value * temp)
              }
            } else {
              res(i) = partGradArr(i)
            }
          }
          lossRegAccu.add(lossReg)
          Vectors.dense(res)
      }
    }

    gradDVWithReg.eagerPersist(storageLevel)

    // println(s"gradDVWithReg: ${gradDVWithReg.toLocal.toString}")

    val regSum = lossRegAccu.value
    (lossSum + 0.5 * regParamL2 * regSum, gradDVWithReg)
  }
}

/**
 * Model produced by [[VLogisticRegression]].
 */
class VLogisticRegressionModel private[spark](
    override val uid: String,
    val coefficients: DV,
    val intercept: Double)
  extends ProbabilisticClassificationModel[Vector, VLogisticRegressionModel]
  with VLogisticRegressionParams {

  /** Margin (rawPrediction) for class label 1.  For binary classification only. */
  private val margin: Vector => Double = (features) => {
    // features.dot(coefficients)
    throw new UnsupportedOperationException("unsupported operation.")
  }

  /** Score (probability) for class label 1.  For binary classification only. */
  private val score: Vector => Double = (features) => {
    val m = margin(features)
    1.0 / (1.0 + math.exp(-m))
  }

  override val numFeatures: Int = {
    require(coefficients.nSize < Int.MaxValue)
    coefficients.nSize.toInt
  }

  override val numClasses: Int = 2

  override protected def predict(features: Vector): Double = {
    if (score(features) > getThreshold) 1 else 0
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        var i = 0
        val size = dv.size
        while (i < size) {
          dv.values(i) = 1.0 / (1.0 + math.exp(-dv.values(i)))
          i += 1
        }
        dv
      case _ => throw new RuntimeException("Unexcepted error in LogisticRegressionModel.")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    val m = margin(features)
    Vectors.dense(-m, m)
  }

  override def copy(extra: ParamMap): VLogisticRegressionModel = {
    val newModel = copyValues(
      new VLogisticRegressionModel(uid, coefficients, intercept),
      extra)
    newModel.setParent(parent)
  }
}
