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
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.linalg.distributed._
import org.apache.spark.ml.optim.{VDiffFunction, VLBFGS, VOWLQN}
import org.apache.spark.ml.param.{BooleanParam, IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.stat.OptimMultivariateOnlineSummarizer
import org.apache.spark.SparkException
import org.apache.spark.ml.VParams
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.RDDUtils

/**
 * Params for vector-free softmax regression.
 */
private[classification] trait VSoftmaxRegressionParams
  extends ProbabilisticClassifierParams with VParams with HasRegParam with HasElasticNetParam
    with HasMaxIter with HasTol with HasStandardization with HasWeightCol with HasThresholds
    with HasCheckpointInterval {

}

/**
 * Softmax regression.
 */
class VSoftmaxRegression(override val uid: String)
  extends ProbabilisticClassifier[Vector, VSoftmaxRegression, VSoftmaxRegressionModel]
    with VSoftmaxRegressionParams with Logging {

  def this() = this(Identifiable.randomUID("vector-free-logreg"))

  /**
   * Set column number of each block in feature block matrix.
   * Default is 10000.
   *
   * @group setParam
   */
  def setColsPerBlock(value: Int): this.type = set(colsPerBlock, value)

  /**
   * Set row number of each block in feature block matrix.
   * Default is 10000.
   *
   * @group setParam
   */
  def setRowsPerBlock(value: Int): this.type = set(rowsPerBlock, value)

  /**
   * Set row partition number of feature block matrix.
   * Default is 10.
   *
   * @group setParam
   */
  def setRowPartitions(value: Int): this.type = set(rowPartitions, value)

  /**
   * Set column partition number of feature block matrix.
   * Default is 10.
   *
   * @group setParam
   */
  def setColPartitions(value: Int): this.type = set(colPartitions, value)

  /**
   * Set whether eagerly persist distributed vectors when calculating.
   * Default is 0.0.
   *
   * @group expertSetParam
   */
  def setEagerPersist(value: Boolean): this.type = set(eagerPersist, value)

  /**
   * Set the LBFGS correction number.
   * Default is 0.0.
   *
   * @group expertSetParam
   */
  def setNumCorrections(value: Int): this.type = set(numCorrections, value)

  /**
   * Set the regularization parameter.
   * Default is 0.0.
   *
   * @group setParam
   */
  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   *
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   *
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  /**
   * Whether to standardize the training features before fitting the model.
   * The coefficients of models will be always returned on the original scale,
   * so it will be transparent for users. Note that with/without standardization,
   * the models should be always converged to the same solution when no regularization
   * is applied. In R's GLMNET package, the default behavior is true as well.
   * Default is true.
   *
   * @group setParam
   */
  def setStandardization(value: Boolean): this.type = set(standardization, value)
  setDefault(standardization -> true)

  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  def setWeightCol(value: String): this.type = set(weightCol, value)

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty.
   * For alpha = 1, it is an L1 penalty.
   * For alpha in (0,1), the penalty is a combination of L1 and L2.
   * Default is 0.0 which is an L2 penalty.
   *
   * @group setParam
   */
  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
  setDefault(elasticNetParam -> 0.0)

  def setCheckpointInterval(interval: Int): this.type = set(checkpointInterval, interval)
  setDefault(checkpointInterval, 30)

  def setGeneratingFeatureMatrixBuffer(size: Int): this.type =
    set(generatingFeatureMatrixBuffer, size)

  def setRowPartitionSplitNumOnGeneratingFeatureMatrix(numSplits: Int): this.type =
    set(rowPartitionSplitNumOnGeneratingFeatureMatrix, numSplits)

  def setCompressFeatureMatrix(value: Boolean): this.type =
    set(compressFeatureMatrix, value)

  override protected[spark] def train(dataset: Dataset[_]): VSoftmaxRegressionModel = {
    logInfo("Start to train VSoftmaxRegressionModel.")

    val sc = dataset.sparkSession.sparkContext
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)).cast(DoubleType), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) {
      dataset.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val labelsAndWeights: RDD[(Double, Double)] = instances.map { instance =>
      (instance.label, instance.weight)
    }.persist(StorageLevel.MEMORY_AND_DISK)

    val partitionSizes: Array[Long] = VUtils.computePartitionSize(labelsAndWeights)
    val numInstances = partitionSizes.sum
    var rowsPerBlockParam: Int = $(rowsPerBlock)
    if (rowsPerBlockParam > numInstances) rowsPerBlockParam = numInstances.toInt
    // Get number of blocks in column direction.
    val rowBlocks = VUtils.getNumBlocks(rowsPerBlockParam, numInstances)

    val numFeatures: Long = instances.first().features.size
    var colsPerBlockParam: Int = $(colsPerBlock)

    if (colsPerBlockParam > numFeatures) colsPerBlockParam = numFeatures.toInt
    // Get number of blocks in column direction.
    val colBlocks: Int = VUtils.getNumBlocks(colsPerBlockParam, numFeatures)

    val labelAndWeight: RDD[(Array[Double], Array[Double])] =
      VUtils.packLabelsAndWeights(partitionSizes, labelsAndWeights, rowsPerBlockParam, rowBlocks)
      .persist(StorageLevel.MEMORY_AND_DISK)

    val weightSum = labelAndWeight.map(_._2.sum).sum()

    val labelSummarizer: MultiClassSummarizer = {
      val seqOp = (c: MultiClassSummarizer, tuple: (Double, Double))
        => c.add(tuple._1, tuple._2)
      val combOp = (c1: MultiClassSummarizer, c2: MultiClassSummarizer)
        => c1.merge(c2)
      labelsAndWeights.treeAggregate(new MultiClassSummarizer)(seqOp, combOp)
    }

    // because `labelAndWeight` has been persisted and `labelsAndWeights` won't be used in future,
    // the parent RDD `labelsAndWeights` can be unpersisted.
    labelsAndWeights.unpersist()
    logInfo("summarize label done.")

    val numClasses = labelSummarizer.numClasses

    val rawFeatures = VUtils.genRawFeatureBlocks(
      partitionSizes,
      instances,
      numFeatures,
      numInstances,
      colBlocks,
      rowBlocks,
      colsPerBlockParam,
      rowsPerBlockParam,
      $(colPartitions),
      $(rowPartitions),
      $(compressFeatureMatrix),
      $(generatingFeatureMatrixBuffer),
      $(rowPartitionSplitNumOnGeneratingFeatureMatrix)
    ).persist(StorageLevel.MEMORY_AND_DISK_SER)

    // force trigger raw features block matrix persist. So that it avoid spark pipeline the
    // group reducer and the following mapJoinPartitions mapper together, because the two
    // stage both need huge memory, separate them help reduce the memory cost and can avoid
    // OOM.
    rawFeatures.blocks.count()
    logInfo("raw feature std generated.")

    // summarize feature std.
    val weightDV = new DistributedVector(
      labelAndWeight.map(tuple => Vectors.dense(tuple._2)),
      rowsPerBlockParam, rowBlocks, numInstances)

    val featuresStd =
      VUtils.genFeatureStd(rawFeatures, weightDV, colsPerBlockParam, colBlocks, numFeatures)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = true)
    logInfo("feature std generated.")

    val features: VBlockMatrix = VUtils.genFeatureBlocks(
        rawFeatures, $(compressFeatureMatrix), featuresStd)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

    features.blocks.count() // force trigger persist
    rawFeatures.blocks.unpersist()
    if (handlePersistence) {
      dataset.unpersist()
    }
    logInfo("features block matrix generated.")

    val regParamL1 = $(elasticNetParam) * $(regParam)
    val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

    val costFun = new VSoftmaxCostFun(
      numClasses,
      features,
      numFeatures,
      numInstances,
      labelAndWeight,
      weightSum,
      $(standardization),
      featuresStd,
      regParamL2,
      $(eagerPersist))

    val optimizer = if ($(elasticNetParam) == 0.0 || $(regParam) == 0.0) {
      new VLBFGS(
        maxIter = $(maxIter),
        m = $(numCorrections),
        tolerance = $(tol),
        checkpointInterval = $(checkpointInterval),
        eagerPersist = $(eagerPersist)
      )
    } else {
      // with L1 regularization, use Vector-free OWLQN optimizer
      if ($(standardization)) {
        new VOWLQN(
          maxIter = $(maxIter),
          m = $(numCorrections),
          l1RegValue = (regParamL1, false),
          tolerance = $(tol),
          checkpointInterval = $(checkpointInterval),
          eagerPersist = $(eagerPersist)
        )
      } else {
        // If `standardization` is false, we still standardize the data
        // to improve the rate of convergence; as a result, we have to
        // perform this reverse standardization by penalizing each component
        // differently to get effectively the same objective function when
        // the training dataset is not standardized.
        val regParamL1DV = {
          val rdd = featuresStd.values.map {
            (partFeatureStd: Vector) =>
              val res = Array.fill(partFeatureStd.size * numClasses)(0.0)
              partFeatureStd.foreachActive { case (index: Int, value: Double) =>
                val regVal = if (partFeatureStd(index) != 0.0) {
                  regParamL1 / partFeatureStd(index)
                } else {
                  0.0
                }
                var i = index * numClasses
                val endPos = i + numClasses
                while (i < endPos) {
                  res(i) = regVal
                  i += 1
                }
              }
              Vectors.dense(res)
          }
          new DistributedVector(rdd, featuresStd.sizePerPart * numClasses,
            featuresStd.numPartitions, featuresStd.size * numClasses)
        }
        new VOWLQN(
          maxIter = $(maxIter),
          m = $(numCorrections),
          l1Reg = regParamL1DV,
          tolerance = $(tol),
          checkpointInterval = $(checkpointInterval),
          eagerPersist = $(eagerPersist)
        )
      }
    }

    val initCoeffs: DistributedVector =
      DistributedVectors.zeros(sc, colsPerBlockParam * numClasses, colBlocks, numFeatures * numClasses)

    initCoeffs.persist(StorageLevel.MEMORY_AND_DISK, eager = $(eagerPersist))

    var state: optimizer.State = null
    val states = optimizer.iterations(costFun, initCoeffs)

    while (states.hasNext) {
      val startTime = System.currentTimeMillis()
      state = states.next()
      val endTime = System.currentTimeMillis()
      logInfo(s"VLogisticRegression iteration ${state.iter} finished, spends ${endTime - startTime} ms.")
      logInfo(s"new X sparsity = ${state.x.values.map(_.numNonzeros).sum() / state.x.size}")
    }
    if (state == null) {
      val msg = s"${optimizer.getClass.getName} failed."
      logError(msg)
      throw new SparkException(msg)
    }

    val rawCoeffs = state.x // `x` already persisted.
    assert(rawCoeffs.isPersisted)

    /**
     * The coefficients are trained in the scaled space; we're converting them back to
     * the original space.
     * Note that the intercept in scaled space and original space is the same;
     * as a result, no scaling is needed.
     */
    val coeffs = rawCoeffs.zipPartitionsWithIndex(featuresStd, rawCoeffs.sizePerPart, rawCoeffs.size) {
      case (pid: Int, partCoeffs: Vector, partFeatursStd: Vector) =>
        val partFeatursStdArr = partFeatursStd.toArray
        val res = Array.fill(partCoeffs.size)(0.0)
        partCoeffs.foreachActive { case (idx: Int, value: Double) =>
          val stdIdx = idx / numClasses
          if (partFeatursStdArr(stdIdx) != 0.0) {
            res(idx) = value / partFeatursStdArr(stdIdx)
          }
        }
        Vectors.dense(res)
    }.compressed // OWLQN will return sparse model, so here compress it.
     .persist(StorageLevel.MEMORY_AND_DISK, eager = true)
    // here must eager persist the RDD, because we need the interceptValAccu value now.

    val interceptVec = Vectors.sparse(numClasses, Seq())
    val coeffsMatrix = {
      val localCoeffsVec = coeffs.toLocalSparse.asInstanceOf[SparseVector]
      VUtils.sparseVectorToColumnMajorSparseMatrix(numClasses, numFeatures.toInt, localCoeffsVec)
    }

    val model = copyValues(new VSoftmaxRegressionModel(uid, coeffsMatrix, interceptVec, numClasses))
    state.dispose(true)
    optimizer.dispose()
    model
  }

  override def copy(extra: ParamMap): VSoftmaxRegression = defaultCopy(extra)
}

private[ml] class VSoftmaxCostFun(
    _numClasses: Int,
    _features: VBlockMatrix,
    _numFeatures: Long,
    _numInstances: Long,
    _labelAndWeight: RDD[(Array[Double], Array[Double])],
    _weightSum: Double,
    _standardization: Boolean,
    _featuresStd: DistributedVector,
    _regParamL2: Double,
    eagerPersist: Boolean) extends VDiffFunction(eagerPersist) {

  // Calculates both the value and the gradient at a point
  override def calculate(coefficients: DistributedVector, checkAndMarkCheckpoint: DistributedVector => Unit):
      (Double, DistributedVector) = {

    val numClasses = _numClasses
    val features: VBlockMatrix = _features
    val numFeatures: Long = _numFeatures
    val numInstances: Long = _numInstances
    val rowsPerBlock = features.rowsPerBlock
    val colsPerBlock = features.colsPerBlock
    val labelAndWeight: RDD[(Array[Double], Array[Double])] = _labelAndWeight
    val weightSum: Double = _weightSum
    val rowBlocks: Int = features.gridPartitioner.rowBlocks
    val colBlocks: Int = features.gridPartitioner.colBlocks
    val standardization: Boolean = _standardization
    val featuresStd: DistributedVector = _featuresStd
    val regParamL2: Double = _regParamL2

    val lossAccu = features.blocks.sparkContext.doubleAccumulator

    assert(RDDUtils.isRDDPersisted(features.blocks))
    assert(coefficients.isPersisted)
    assert(RDDUtils.isRDDPersisted(labelAndWeight))

    val multiplierRDD: RDD[Vector] = features.horizontalZipVector(coefficients) {
      (blockCoordinate: (Int, Int), block: VMatrix, partCoefficients: Vector) =>
        val partMarginArr = Array.fill[Double](block.numRows * numClasses)(0.0)
        block.foreachActive { case (i: Int, j: Int, v: Double) =>
          var k = 0
          while (k < numClasses) {
            partMarginArr(i * numClasses + k) += (partCoefficients(j * numClasses + k) * v)
            k += 1
          }
        }
        Vectors.dense(partMarginArr).compressed
    }.map { case ((rowBlockIdx: Int, colBlockIdx: Int), partMargins: Vector) =>
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
          while (i < labelArr.length) {
            var k = 0
            var maxMagin = Double.NegativeInfinity
            while (k < numClasses) {
              val margin = marginArr(i * numClasses + k)
              if (margin > maxMagin) {
                maxMagin = margin
              }
              k += 1
            }

            /**
             * When maxMargin is greater than 0, the original formula could cause overflow.
             * We address this by subtracting maxMargin from all the margins, so it's guaranteed
             * that all of the new margins will be smaller than zero to prevent arithmetic overflow.
             */
            k = 0
            var expSum = 0.0
            while (k < numClasses) {
              val idx = i * numClasses + k
              if (maxMagin > 0) {
                marginArr(idx) = marginArr(idx) - maxMagin
              }
              val exp = math.exp(marginArr(idx))
              expSum += exp
              multiplierArr(idx) = exp
              k += 1
            }

            k = 0
            while (k < numClasses) {
              val idx = i * numClasses + k
              multiplierArr(idx) = weightArr(i) * (multiplierArr(idx) / expSum
                - (if (labelArr(i).toInt == k) 1.0 else 0.0))
              k += 1
            }

            val marginOfLabel = marginArr(i * numClasses + labelArr(i).toInt)
            lossSum += weightArr(i) * (math.log(expSum) - marginOfLabel)

            i += 1
          }

          lossAccu.add(lossSum)
          Vectors.dense(multiplierArr)
      }

    // here must eager persist the RDD, because we need the lossAccu value now.
    val multiplier: DistributedVector =
      new DistributedVector(multiplierRDD, rowsPerBlock * numClasses, rowBlocks, numInstances * numClasses)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = true)

    val lossSum = lossAccu.value / weightSum

    val gradientRDD: RDD[Vector] = features.verticalZipVector(multiplier) {
      (blockCoordinate: (Int, Int), block: VMatrix, partMultipliers: Vector) =>
        val partGradArr = Array.fill[Double](block.numCols * numClasses)(0.0)

        var k = 0
        while (k < numClasses) {
          block.foreachActive { case (i: Int, j: Int, v: Double) =>
            partGradArr(j * numClasses + k) += (partMultipliers(i * numClasses + k) * v)
          }
          k += 1
        }
        Vectors.dense(partGradArr).compressed
    }.map { case ((rowBlockIdx: Int, colBlockIdx: Int), partGrads: Vector) =>
      (colBlockIdx, partGrads)
    }.aggregateByKey(new VectorSummarizer, new DistributedVectorPartitioner(colBlocks))(
      (s, v) => s.add(v),
      (s1, s2) => s1.merge(s2)
    ).map { case (colBlockIdx: Int, gradientSummarizer: VectorSummarizer) =>
      val partGradArr = gradientSummarizer.toArray
      var i = 0
      while (i < partGradArr.length) {
        partGradArr(i) /= weightSum
        i += 1
      }
      Vectors.dense(partGradArr)
    }

    val gradient: DistributedVector = new DistributedVector(gradientRDD,
      colsPerBlock * numClasses, colBlocks, numFeatures * numClasses
    ).persist(StorageLevel.MEMORY_AND_DISK, eager = eagerPersist)

    assert(featuresStd.isPersisted)

    // compute regularization for grad & objective value
    val lossRegAccu = gradient.values.sparkContext.doubleAccumulator
    val gradWithReg: DistributedVector = if (standardization) {
      gradient.zipPartitionsWithIndex(coefficients) {
        case (pid: Int, partGradients: Vector, partCoefficients: Vector) =>
          var lossReg = 0.0
          val partGradArr = partGradients.toArray
          val res = Array.fill[Double](partGradients.size)(0.0)
          partCoefficients.foreachActive { case (i: Int, value: Double) =>
            res(i) = partGradArr(i) + regParamL2 * value
            lossReg += (value * value)
          }
          lossRegAccu.add(lossReg)
          Vectors.dense(res)
      }
    } else {
      gradient.zipPartitionsWithIndex(coefficients, featuresStd, coefficients.sizePerPart, coefficients.size) {
        case (pid: Int, partGradients: Vector, partCoefficients: Vector, partFeaturesStd: Vector) =>
          var lossReg = 0.0
          val partGradArr = partGradients.toArray
          val partFeaturesStdArr = partFeaturesStd.toArray
          val res = Array.fill[Double](partGradArr.length)(0.0)
          partCoefficients.foreachActive { case (i: Int, value: Double) =>
            // If `standardization` is false, we still standardize the data
            // to improve the rate of convergence; as a result, we have to
            // perform this reverse standardization by penalizing each component
            // differently to get effectively the same objective function when
            // the training dataset is not standardized.
            val stdIdx = i / numClasses
            if (partFeaturesStdArr(stdIdx) != 0.0) {
              val temp = value / (partFeaturesStdArr(stdIdx) * partFeaturesStdArr(stdIdx))
              res(i) = partGradArr(i) + regParamL2 * temp
              lossReg += (value * temp)
            }
          }
          lossRegAccu.add(lossReg)
          Vectors.dense(res)
      }
    }

    // possible checkpoint(determined by the checkpointInterval) first, then eager persist
    if (checkAndMarkCheckpoint != null) {
      checkAndMarkCheckpoint(gradWithReg)
    }
    // here must eager persist the RDD, because we need the lossRegAccu value now.
    gradWithReg.persist(StorageLevel.MEMORY_AND_DISK, eager = true)

    // because gradWithReg already eagerly persisted, now we can release multiplier & gradient
    multiplier.unpersist()
    gradient.unpersist()

    val regSum = lossRegAccu.value
    (lossSum + 0.5 * regParamL2 * regSum, gradWithReg)
  }
}

/**
 * Model produced by [[VSoftmaxRegression]].
 */
class VSoftmaxRegressionModel private[spark] (
    override val uid: String,
    val coefficientMatrix: Matrix,
    val interceptVector: Vector,
    override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, VSoftmaxRegressionModel]
    with LogisticRegressionParams {

  require(coefficientMatrix.numRows == interceptVector.size, s"Dimension mismatch! Expected " +
    s"coefficientMatrix.numRows == interceptVector.size, but ${coefficientMatrix.numRows} != " +
    s"${interceptVector.size}")

  override def setThresholds(value: Array[Double]): this.type = super.setThresholds(value)

  override def getThresholds: Array[Double] = super.getThresholds

  /** Margin (rawPrediction) for each class label. */
  private val margins: Vector => Vector = (features) => {
    val m = interceptVector.toDense.copy
    BLAS.gemv(1.0, coefficientMatrix, features, 1.0, m)
    m
  }

  override val numFeatures: Int = coefficientMatrix.numCols

  /**
   * Predict label for the given feature vector.
   * The behavior of this can be adjusted using `thresholds`.
   */
  override protected def predict(features: Vector): Double = super.predict(features)

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>

        val size = dv.size
        val values = dv.values

        // get the maximum margin
        val maxMarginIndex = rawPrediction.argmax
        val maxMargin = rawPrediction(maxMarginIndex)

        if (maxMargin == Double.PositiveInfinity) {
          var k = 0
          while (k < size) {
            values(k) = if (k == maxMarginIndex) 1.0 else 0.0
            k += 1
          }
        } else {
          val sum = {
            var temp = 0.0
            var k = 0
            while (k < numClasses) {
              values(k) = if (maxMargin > 0) {
                math.exp(values(k) - maxMargin)
              } else {
                math.exp(values(k))
              }
              temp += values(k)
              k += 1
            }
            temp
          }
          BLAS.scal(1 / sum, dv)
        }
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in VSoftmaxRegressionModel:" +
          " raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    margins(features)
  }

  override def copy(extra: ParamMap): VSoftmaxRegressionModel = {
    val newModel = copyValues(new VSoftmaxRegressionModel(uid, coefficientMatrix, interceptVector,
      numClasses), extra)
    newModel.setParent(parent)
  }

  override protected def raw2prediction(rawPrediction: Vector): Double = {
    super.raw2prediction(rawPrediction)
  }

  override protected def probability2prediction(probability: Vector): Double = {
    super.probability2prediction(probability)
  }
}
