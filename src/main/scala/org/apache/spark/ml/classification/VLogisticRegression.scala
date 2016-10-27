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

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{DistributedVector => DV, DistributedVectors => DVs, _}
import org.apache.spark.ml.optim.{DVDiffFunction, GridPartitionerV2, OneDimGridPartitioner, VFUtils, VectorFreeLBFGS}
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.mllib.linalg.distributed.GridPartitioner
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.{Partitioner, SparkException}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils

/**
 * Params for vector-free logistic regression.
 */
private[classification] trait VLogisticRegressionParams
  extends ProbabilisticClassifierParams with HasRegParam with HasMaxIter
    with HasTol with HasStandardization with HasWeightCol with HasThreshold{
}
object VLogisticRegression {
  val storageLevel = StorageLevel.MEMORY_AND_DISK
}
/**
 * Logistic regression.
 */
@Since("2.1.0")
class VLogisticRegression @Since("2.1.0")(
                                           @Since("2.1.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, VLogisticRegression, VLogisticRegressionModel]
    with VLogisticRegressionParams with DefaultParamsWritable with Logging {

  import VLogisticRegression._

  @Since("2.1.0")
  def this() = this(Identifiable.randomUID("vector-free-logreg"))

  val featuresPartSize: IntParam
  = new IntParam(this, "vecPartSize", "each part size of large vector.", ParamValidators.gt(0))
  setDefault(featuresPartSize -> 10000)

  @Since("2.1.0")
  def setFeaturesPartSize(value: Int): this.type = set(featuresPartSize, value)

  val instanceStackSize: IntParam
  = new IntParam(this, "instanceStackSize", "instance stack size.", ParamValidators.gt(0))
  setDefault(instanceStackSize -> 10000)

  @Since("2.1.0")
  def setInstanceStackSize(value: Int): this.type = set(instanceStackSize, value)

  val blockMatrixRowPartNum: IntParam
  = new IntParam(this, "blockMatrixRowPartNum", "block matrix row partition number.", ParamValidators.gt(0))
  setDefault(blockMatrixRowPartNum -> 10)

  @Since("2.1.0")
  def setBlockMatrixRowPartNum(value: Int): this.type = set(blockMatrixRowPartNum, value)

  val blockMatrixColPartNum: IntParam
  = new IntParam(this, "blockMatrixColPartNum", "block matrix col partition number.", ParamValidators.gt(0))
  setDefault(blockMatrixColPartNum -> 10)

  @Since("2.1.0")
  def setBlockMatrixColPartNum(value: Int): this.type = set(blockMatrixColPartNum, value)

  /*
  val trainingDataMatrixVertPartNum: IntParam
    = new IntParam(this, "trainingDataMatrixVertPartNum", "trainingDataMatrixVertPartNum", ParamValidators.gt(0))
  setDefault(trainingDataMatrixVertPartNum -> 500)

  @Since("2.1.0")
  def setTrainingDataMatrixVertPartNum(value: Int): this.type = set(trainingDataMatrixVertPartNum, value)
  */

  @Since("2.1.0")
  def setRegParam(value: Double): this.type = set(regParam, value)

  setDefault(regParam -> 0.0)

  @Since("2.1.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  setDefault(maxIter -> 100)

  @Since("2.1.0")
  def setTol(value: Double): this.type = set(tol, value)

  setDefault(tol -> 1E-6)

  @Since("2.1.0")
  def setStandardization(value: Boolean): this.type = set(standardization, value)

  setDefault(standardization -> true)

  @Since("2.1.0")
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
    val localFeaturesPartSize = $(featuresPartSize)
    val localInstanceStackSize = $(instanceStackSize)
    val blockMatrixColNum = VFUtils.getSplitPartNum(localFeaturesPartSize, numFeatures)

    // 1. features statistics
    val featuresSummarizerRDD = {
      val featuresRDD = instances.flatMap {
        case Instance(label, weight, features) =>
          val featuresList = VFUtils.splitSparseVector(
            features.toSparse, localFeaturesPartSize)
          featuresList.zipWithIndex.map {
            case (partFeatures, partId) =>
              (partId, (partFeatures, weight))
          }
      }
      val seqOp = (s: MultivariateOnlineSummarizer, partFeatures: (Vector, Double)) =>
        s.add(partFeatures._1, partFeatures._2)
      val comOp = (s1: MultivariateOnlineSummarizer, s2: MultivariateOnlineSummarizer) =>
        s1.merge(s2)
      featuresRDD.aggregateByKey(new MultivariateOnlineSummarizer,
        new DistributedVectorPartitioner(blockMatrixColNum)
      )(seqOp, comOp)
        .persist(storageLevel)
    }
    val featuresStd = VFUtils.kvRDDToDV(
      featuresSummarizerRDD.mapValues(summarizer =>
        Vectors.dense(summarizer.variance.toArray.map(math.sqrt))),
      localFeaturesPartSize, blockMatrixColNum, numFeatures).eagerPersist(storageLevel)

    // println(s"dv feature std: ${featuresStd.toLocal().toString}")

    featuresSummarizerRDD.unpersist()

    val labelAndWeightRDD = instances.map(instance =>
      (instance.label, instance.weight)).persist(storageLevel)

    /*
    // 2. label statistics
    val labelSummarizer = {
      labelAndWeightRDD.aggregate(new MultiClassSummarizer)(
        (s: MultiClassSummarizer, v: (Double, Double)) => s.add(v._1, v._2),
        (s1: MultiClassSummarizer, s2: MultiClassSummarizer) => s1.merge(s2)
      )
    }
    val histogram = labelSummarizer.histogram
    val numInvalid = labelSummarizer.countInvalid
    val numClasses = histogram.length
    */

    // 3. statistic each partition size.

    val partitionSizeArray = VFUtils.computePartitionStartIndices(labelAndWeightRDD)
    val numInstances = partitionSizeArray.sum
    val blockMatrixRowNum = VFUtils.getSplitPartNum(localInstanceStackSize, numInstances)
    // val localVertPartsNum = $(trainingDataMatrixVertPartNum)
    // val vertBlocksPerVertPart = VFUtils.getSplitPartSize(localVertPartsNum ,blockMatrixRowNum)

    // 4. pack multiple instances labels together
    // (labelArr, weightArr)
    val labelAndWeight = VFUtils.zipRDDWithIndex(partitionSizeArray, labelAndWeightRDD).map {
      case (rowIdx: Long, (label: Double, weight: Double)) =>
        val blockRowIdx = (rowIdx / localInstanceStackSize).toInt
        val inBlockIdx = (rowIdx % localInstanceStackSize).toInt
        (blockRowIdx, (inBlockIdx, label, weight))
    }.groupByKey(new DistributedVectorPartitioner(blockMatrixRowNum)).map {
      case (blockRowIdx: Int, iter: Iterable[(Int, Double, Double)]) =>
        val tupleArr = iter.toArray.sortWith(_._1 < _._1)
        val labelArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._2)
        val weightArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._3)
        (labelArr, weightArr)
    }.persist(storageLevel)
    val weightSum = labelAndWeight.map(_._2.sum).sum()

    // println(s"weightSum: ${weightSum}")

    // column-majar grid partitioner.
    var localBlockMatrixRowPartNum = $(blockMatrixRowPartNum)
    var localBlockMatrixColPartNum = $(blockMatrixColPartNum)
    if (localBlockMatrixRowPartNum > blockMatrixRowNum) localBlockMatrixRowPartNum = blockMatrixRowNum
    if (localBlockMatrixColPartNum > blockMatrixColNum) localBlockMatrixColPartNum = blockMatrixColNum

    val gridPartitioner = new GridPartitionerV2(
      blockMatrixRowNum, blockMatrixColNum,
      blockMatrixRowNum / localBlockMatrixRowPartNum,
      blockMatrixColNum / localBlockMatrixColPartNum
    )

    // 5. pack features into blcok matrix
    val featuresBlockMatrix = VFUtils.zipRDDWithIndex(partitionSizeArray, instances).flatMap {
      case (rowIdx: Long, Instance(label, weight, features)) =>
        val blockRowIdx = (rowIdx / localInstanceStackSize).toInt
        val inBlockIdx = (rowIdx % localInstanceStackSize).toInt
        val featuresList = VFUtils.splitSparseVector(
          features.toSparse, localFeaturesPartSize)
        featuresList.zipWithIndex.map {
          case (partFeatures, partId) =>
            ((blockRowIdx, partId), (inBlockIdx, partFeatures))
        }
    }.groupByKey(new Partitioner() {
      override def getPartition(key: Any): Int = key match {
        case (blockRowIdx: Int, blockColIdx: Int) => blockColIdx
        case _ => throw new IllegalArgumentException(s"Unrecognized key: $key.")
      }
      override def numPartitions: Int = blockMatrixColNum
    }).map {
      case ((blockRowIdx: Int, partId: Int), iter: Iterable[(Int, SparseVector)]) =>
        val vecs = iter.toArray.sortWith(_._1 < _._1).map(_._2)
        val matrix = VFUtils.vertcatSparseVectorIntoCSRMatrix(vecs)
        ((blockRowIdx, partId), matrix)
    }.zipPartitions(featuresStd.vecs)(
      (matrixIter, featuresStdIter) => {
        val partFeatureStdVector = featuresStdIter.next()
        assert(!featuresStdIter.hasNext)
        matrixIter.map {
          case (matrixBlock: ((Int, Int), SparseMatrix)) =>
            val partFeatureStdArr = partFeatureStdVector.asInstanceOf[DenseVector].values
            val sm = matrixBlock._2
            val arrBuf = new ArrayBuffer[(Int, Int, Double)]()
            sm.foreachActive {
              (i: Int, j: Int, value: Double) =>
                if (partFeatureStdArr(j) != 0 && value != 0) {
                  arrBuf.append((j, i, value / partFeatureStdArr(j)))
                }
            }
            (matrixBlock._1, SparseMatrix.fromCOO(sm.numCols, sm.numRows, arrBuf).transpose)
        }
      }
    ).partitionBy(gridPartitioner).persist(storageLevel)

    val costFun = new VFBinomialLogisticCostFun(
      numFeatures,
      localFeaturesPartSize,
      numInstances,
      localInstanceStackSize,
      featuresBlockMatrix,
      gridPartitioner,
      labelAndWeight,
      weightSum,
      blockMatrixRowNum,
      blockMatrixColNum,
      $(standardization),
      featuresStd: DV,
      $(regParam))
    val optimizer = new VectorFreeLBFGS($(maxIter), 10, $(tol))
    val initCoeffs = DVs.zeros(sc, localFeaturesPartSize, blockMatrixColNum, numFeatures).eagerPersist()
    val states = optimizer.iterations(costFun, initCoeffs)

    var state: optimizer.State = null
    while (states.hasNext) {
      state = states.next()
    }
    if (state == null) {
      val msg = s"${optimizer.getClass.getName} failed."
      logError(msg)
      throw new SparkException(msg)
    }

    val rawCoeffs = state.x // `x` already persisted.
    val coeffs = rawCoeffs.zipPartitions(featuresStd) {
      case (partCoeffs: Vector, partFeatursStd: Vector) =>
        val partFeatursStdArr = partFeatursStd.toDense.toArray
        val res = Array.fill(partCoeffs.size)(0.0)
        partCoeffs.foreachActive {
          case (idx: Int, value: Double) =>
            if (partFeatursStdArr(idx) != 0.0) {
              res(idx) = value / partFeatursStdArr(idx)
            }
        }
        Vectors.dense(res)
    }.eagerPersist()
    val model = copyValues(new VLogisticRegressionModel(uid, coeffs))
    model
  }

  override def copy(extra: ParamMap): VLogisticRegression = defaultCopy(extra)
}

private class VFBinomialLogisticCostFun(
    _featuresNum: Long,
    _featuresPartSize: Int,
    _numInstances: Long,
    _instanceStackSize: Int,
    _featuresBlockMatrix: RDD[((Int, Int), SparseMatrix)],
    _gridParitioner: GridPartitionerV2,
    _labelAndWeight: RDD[(Array[Double], Array[Double])],
    _weightSum: Double,
    _blockMatrixRowNum: Int,
    _blockMatrixColNum: Int,
    _standardization: Boolean,
    _featuresStd: DV,
    _regParamL2: Double) extends DVDiffFunction {

  import VLogisticRegression._
  // Calculates both the value and the gradient at a point
  override def calculate(coeffs: DV): (Double, DV) = {
    val featuresNum = _featuresNum
    val featuresPartSize = _featuresPartSize
    val numInstances = _numInstances
    val instanceStackSize = _instanceStackSize
    val featuresBlockMatrix = _featuresBlockMatrix
    val gridPartitioner = _gridParitioner
    val labelAndWeight = _labelAndWeight
    val weightSum = _weightSum
    val blockMatrixRowNum = _blockMatrixRowNum
    val blockMatrixColNum = _blockMatrixColNum

    val standardization = _standardization
    val featuresStd = _featuresStd
    val regParamL2 = _regParamL2

    val lossAccu = featuresBlockMatrix.sparkContext.doubleAccumulator
    val multipliers = VFUtils.blockMatrixHorzZipVec(featuresBlockMatrix, coeffs, gridPartitioner, (matrix, partCoeffs) => {
      val partMarginArr = Array.fill[Double](matrix.numRows)(0.0)
      matrix.foreachActive { case (i: Int, j: Int, v: Double) =>
        partMarginArr(i) += (partCoeffs(j) * v)
      }
      new BDV(partMarginArr)
    })/*
    val multipliers = featuresBlockMatrixPartitionByBlockColIdx.zipPartitions(coeffs.vecs) {
      (matrixIter: Iterator[((Int, Int), SparseMatrix)], coeffsIter: Iterator[Vector]) =>
        val partCoeffs = coeffsIter.next()
        assert(!coeffsIter.hasNext)
        matrixIter.map {
          case ((rowIdx: Int, colIdx: Int), matrix: SparseMatrix) =>
            val partMarginArr = Array.fill[Double](matrix.numRows)(0.0)
            matrix.foreachActive { case (i: Int, j: Int, v: Double) =>
              partMarginArr(i) += (partCoeffs(j) * v)
            }
            (rowIdx, new BDV(partMarginArr))
        }
    }*/.reduceByKey(new DistributedVectorPartitioner(blockMatrixRowNum), _ + _)
      .zip(labelAndWeight).map {
      case ((rowIdx: Int, marginArr0: BDV[Double]), (labelArr: Array[Double], weightArr: Array[Double])) =>
        val marginArr = (marginArr0 * (-1.0)).toArray
        var lossSum = 0.0
        val multiplierArr = Array.fill(marginArr.length)(0.0)
        var i = 0
        while (i < marginArr.length) {
          val label = labelArr(i)
          val margin = marginArr(i)
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
    val multipliersDV = new DV(multipliers, instanceStackSize, blockMatrixRowNum, numInstances)
      .eagerPersist(storageLevel)

    val lossSum = lossAccu.value / weightSum
    val grad = VFUtils.blockMatrixVertZipVec(featuresBlockMatrix, multipliersDV, gridPartitioner,
      (matrix, partMultipliers) => {
        val partGradArr = Array.fill[Double](matrix.numCols)(0.0)
        matrix.foreachActive { case (i: Int, j: Int, v: Double) =>
          partGradArr(j) += (partMultipliers(i) * v)
        }
        new BDV[Double](partGradArr)
    })/*
    val gradDV = new DV(featuresBlockMatrixPartitionByBlockRowIdx.zipPartitions(multipliers) {
      (matrixIter: Iterator[((Int, Int), SparseMatrix)], multiplierIter: Iterator[Array[Double]]) =>
        val multiplierArr = multiplierIter.next()
        assert(!multiplierIter.hasNext)
        matrixIter.map {
          case ((rowIdx: Int, colIdx: Int), matrix: SparseMatrix) =>
            val partGradArr = Array.fill[Double](matrix.numCols)(0.0)
            matrix.foreachActive { case (i: Int, j: Int, v: Double) =>
              partGradArr(j) += (multiplierArr(i) * v)
            }
            (colIdx, new BDV[Double](partGradArr))
        }
    }*/.reduceByKey(new DistributedVectorPartitioner(blockMatrixColNum), _ + _)
      .map(bv => Vectors.fromBreeze(bv._2 / weightSum))
    val gradDV = new DV(grad, featuresPartSize, blockMatrixColNum, featuresNum)
      .eagerPersist(storageLevel)

    // compute regulation for grad & objective value
    val lossRegAccu = gradDV.vecs.sparkContext.doubleAccumulator
    val gradDVWithReg = if (standardization) {
      gradDV.zipPartitions(coeffs) {
        (partGrad: Vector, partCoeffs: Vector) =>
          var lossReg = 0.0
          val partGradArr = partGrad.toArray
          val res = Array.fill[Double](partGrad.size)(0.0)
          partCoeffs.foreachActive {
            case (i: Int, value: Double) =>
              res(i) = partGradArr(i) + regParamL2 * value
              lossReg += (value * value)
          }
          lossRegAccu.add(lossReg)
          Vectors.dense(res)
      }
    } else {
      gradDV.zipPartitions(coeffs, featuresStd) {
        (partGrad: Vector, partCoeffs: Vector, partFeaturesStd: Vector) =>
          var lossReg = 0.0
          val partGradArr = partGrad.toArray
          val partFeaturesStdArr = partFeaturesStd.toArray
          val res = Array.fill[Double](partGradArr.length)(0.0)
          partCoeffs.foreachActive {
            case (i: Int, value: Double) =>
              if (partFeaturesStdArr(i) != 0.0) {
                val temp = value / (partFeaturesStdArr(i) * partFeaturesStdArr(i))
                res(i) = partGradArr(i) + regParamL2 * temp
                lossReg += (value * temp)
              }
          }
          lossRegAccu.add(lossReg)
          Vectors.dense(res)
      }
    }
    gradDVWithReg.eagerPersist(storageLevel)
    val regSum = lossRegAccu.value
    (lossSum + 0.5 * regParamL2 * regSum, gradDVWithReg)
  }
}

/**
 * Model produced by [[VLogisticRegression]].
 */
@Since("1.4.0")
class VLogisticRegressionModel private[spark](
                                               @Since("1.4.0") override val uid: String,
                                               @Since("2.0.0") val coefficients: DV)
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

  @Since("1.6.0")
  override val numFeatures: Int = {
    require(coefficients.nSize < Int.MaxValue)
    coefficients.nSize.toInt
  }

  @Since("1.3.0")
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
    val newModel = copyValues(new VLogisticRegressionModel(uid, coefficients), extra)
    newModel.setParent(parent)
  }
}
