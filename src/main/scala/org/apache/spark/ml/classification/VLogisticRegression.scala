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
import org.apache.spark.ml.linalg.{BLAS, DenseVector, SparseMatrix, SparseVector, Vector, Vectors}
import org.apache.spark.ml.linalg.distributed._
import org.apache.spark.ml.optim.{VDiffFunction, VectorFreeLBFGS, VectorFreeOWLQN}
import org.apache.spark.ml.param.{BooleanParam, IntParam, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.stat.OptimMultivariateOnlineSummarizer
import org.apache.spark.SparkException
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.RDDUtils

/**
 * Params for vector-free logistic regression.
 */
private[classification] trait VLogisticRegressionParams
  extends ProbabilisticClassifierParams with HasRegParam with HasElasticNetParam
    with HasMaxIter with HasTol with HasStandardization with HasWeightCol with HasThreshold
    with HasFitIntercept with HasCheckpointInterval

/**
 * Logistic regression.
 */
class VLogisticRegression(override val uid: String)
  extends ProbabilisticClassifier[Vector, VLogisticRegression, VLogisticRegressionModel]
    with VLogisticRegressionParams with Logging {

  def this() = this(Identifiable.randomUID("vector-free-logreg"))

  // column number of each block in feature block matrix
  val colsPerBlock: IntParam = new IntParam(this, "colsPerBlock",
    "column number of each block in feature block matrix.", ParamValidators.gt(0))
  setDefault(colsPerBlock -> 10000)

  /**
   * Set column number of each block in feature block matrix.
   * Default is 10000.
   *
   * @group setParam
   */
  def setColsPerBlock(value: Int): this.type = set(colsPerBlock, value)

  // row number of each block in feature block matrix
  val rowsPerBlock: IntParam = new IntParam(this, "rowsPerBlock",
    "row number of each block in feature block matrix.", ParamValidators.gt(0))
  setDefault(rowsPerBlock -> 10000)

  /**
   * Set row number of each block in feature block matrix.
   * Default is 10000.
   *
   * @group setParam
   */
  def setRowsPerBlock(value: Int): this.type = set(rowsPerBlock, value)

  // row partition number of feature block matrix
  // equals to partition number of coefficient vector
  val rowPartitions: IntParam = new IntParam(this, "rowPartitions",
    "row partition number of feature block matrix.", ParamValidators.gt(0))
  setDefault(rowPartitions -> 10)

  /**
   * Set row partition number of feature block matrix.
   * Default is 10.
   *
   * @group setParam
   */
  def setRowPartitions(value: Int): this.type = set(rowPartitions, value)

  // column partition number of feature block matrix
  val colPartitions: IntParam = new IntParam(this, "colPartitions",
    "column partition number of feature block matrix.", ParamValidators.gt(0))
  setDefault(colPartitions -> 10)

  /**
   * Set column partition number of feature block matrix.
   * Default is 10.
   *
   * @group setParam
   */
  def setColPartitions(value: Int): this.type = set(colPartitions, value)

  // Whether to eager persist distributed vector
  val eagerPersist: BooleanParam = new BooleanParam(this, "eagerPersist",
    "Whether to eager persist distributed vector.")
  setDefault(eagerPersist -> false)

  /**
   * Set whether eagerly persist distributed vectors when calculating.
   * Default is 0.0.
   *
   * @group expertSetParam
   */
  def setEagerPersist(value: Boolean): this.type = set(eagerPersist, value)

  // LBFGS Corrections number
  val numLBFGSCorrections: IntParam = new IntParam(this, "numLBFGSCorrections",
    "number of LBFGS Corrections")
  setDefault(numLBFGSCorrections -> 10)

  /**
   * Set the LBFGS correction number.
   * Default is 0.0.
   *
   * @group expertSetParam
   */
  def setNumLBFGSCorrections(value: Int): this.type = set(numLBFGSCorrections, value)

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
   * Whether to fit an intercept term.
   * Default is true.
   *
   * @group setParam
   */
  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)

  setDefault(fitIntercept -> true)

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

  setDefault(checkpointInterval, 15)

  override protected[spark] def train(dataset: Dataset[_]): VLogisticRegressionModel = {
    logInfo("Begin training VLogisticRegression")

    val sc = dataset.sparkSession.sparkContext
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

    val instances: RDD[Instance] =
      dataset.select(col($(labelCol)).cast(DoubleType), w, col($(featuresCol))).rdd.map {
        case Row(label: Double, weight: Double, features: Vector) =>
          Instance(label, weight, features)
      }

    val numFeatures: Long = instances.first().features.size
    var colsPerBlockParam: Int = $(colsPerBlock)

    if (colsPerBlockParam > numFeatures) colsPerBlockParam = numFeatures.toInt
    // Get number of blocks in column direction.
    val colBlocks: Int = VUtils.getNumBlocks(colsPerBlockParam, numFeatures)

    val featuresSummarizer: RDD[(Int, OptimMultivariateOnlineSummarizer)] = {
      val features = instances.flatMap {
        case Instance(label, weight, features) =>
          val featuresArray = VUtils.splitSparseVector(features.toSparse, colsPerBlockParam)
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
      )(seqOp, comOp).persist(StorageLevel.MEMORY_AND_DISK)
    }

    val featuresStd: DistributedVector = VUtils.kvRDDToDV(
      featuresSummarizer.mapValues { summarizer =>
        Vectors.dense(summarizer.variance.toArray.map(math.sqrt))
      }, colsPerBlockParam, colBlocks, numFeatures)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = $(eagerPersist))

    featuresSummarizer.unpersist()

    val labelAndWeightRDD: RDD[(Double, Double)] = instances.map { instance =>
      (instance.label, instance.weight)
    }.persist(StorageLevel.MEMORY_AND_DISK)

    val partitionSizes: Array[Long] = VUtils.computePartitionSize(labelAndWeightRDD)
    val numInstances = partitionSizes.sum
    var rowsPerBlockParam: Int = $(rowsPerBlock)

    if (rowsPerBlockParam > numInstances) rowsPerBlockParam = numInstances.toInt
    // Get number of blocks in column direction.
    val rowBlocks = VUtils.getNumBlocks(rowsPerBlockParam, numInstances)

    val labelAndWeight: RDD[(Array[Double], Array[Double])] =
      VUtils.zipRDDWithIndex(partitionSizes, labelAndWeightRDD)
        .map { case (rowIdx: Long, (label: Double, weight: Double)) =>
          val rowBlockIdx = (rowIdx / rowsPerBlockParam).toInt
          val inBlockIdx = (rowIdx % rowsPerBlockParam).toInt
          (rowBlockIdx, (inBlockIdx, label, weight))
        }
        .groupByKey(new DistributedVectorPartitioner(rowBlocks))
        .map { case (blockRowIdx: Int, iter: Iterable[(Int, Double, Double)]) =>
          val tupleArr = iter.toArray.sortWith(_._1 < _._1)
          val labelArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._2)
          val weightArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._3)
          (labelArr, weightArr)
        }.persist(StorageLevel.MEMORY_AND_DISK)

    val weightSum = labelAndWeight.map(_._2.sum).sum()

    val labelSummarizer = {
      val seqOp = (c: MultiClassSummarizer, instance: Instance)
      => c.add(instance.label, instance.weight)

      val combOp = (c1: MultiClassSummarizer, c2: MultiClassSummarizer)
      => c1.merge(c2)

      instances.aggregate(new MultiClassSummarizer)(seqOp, combOp)
    }

    val histogram = labelSummarizer.histogram

    var localRowPartitions = $(rowPartitions)
    var localColPartitions = $(colPartitions)
    if (localRowPartitions > rowBlocks) localRowPartitions = rowBlocks
    if (localColPartitions > colBlocks) localColPartitions = colBlocks

    val gridPartitioner = new VGridPartitioner(
      rowBlocks, colBlocks,
      rowBlocks / localRowPartitions,
      colBlocks / localColPartitions
    )

    val rawFeaturesBlocks: RDD[((Int, Int), SparseMatrix)] =
      VUtils.zipRDDWithIndex(partitionSizes, instances)
        .flatMap { case (rowIdx: Long, Instance(label, weight, features)) =>
          val rowBlockIdx = (rowIdx / rowsPerBlockParam).toInt
          val inBlockIdx = (rowIdx % rowsPerBlockParam).toInt
          val featuresArray = VUtils.splitSparseVector(features.toSparse, colsPerBlockParam)
          featuresArray.zipWithIndex.map { case (partFeatures, partId) =>
            // partId corresponds to colBlockIdx
            ((rowBlockIdx, partId), (inBlockIdx, partFeatures))
          }
        }
        .groupByKey(gridPartitioner)
        .map { case ((rowBlockIdx: Int, colBlockIdx: Int), iter: Iterable[(Int, SparseVector)]) =>
          val vecs = iter.toArray.sortWith(_._1 < _._1).map(_._2)
          val matrix = VUtils.vertcatSparseVectorIntoMatrix(vecs)
          ((rowBlockIdx, colBlockIdx), matrix)
        }

    val rawFeatures = new VBlockMatrix(rowsPerBlockParam, colsPerBlockParam, rawFeaturesBlocks,
      gridPartitioner)

    val features: VBlockMatrix = rawFeatures.horizontalZipVecMap(featuresStd) {
      (blockCoords: (Int, Int), sm: SparseMatrix, partFeatureStdVector: Vector) =>
        val partFeatureStdArr = partFeatureStdVector.asInstanceOf[DenseVector].values
        val arrBuf = new ArrayBuffer[(Int, Int, Double)]()
        sm.foreachActive { case (i: Int, j: Int, value: Double) =>
          if (partFeatureStdArr(j) != 0 && value != 0) {
            arrBuf.append((j, i, value / partFeatureStdArr(j)))
          }
        }
        SparseMatrix.fromCOO(sm.numCols, sm.numRows, arrBuf).transpose
    }.persist(StorageLevel.MEMORY_AND_DISK)

    val fitInterceptParam = $(fitIntercept)
    val regParamL1 = $(elasticNetParam) * $(regParam)
    val regParamL2 = (1.0 - $(elasticNetParam)) * $(regParam)

    val costFun = new VBinomialLogisticCostFun(
      features,
      numFeatures,
      numInstances,
      labelAndWeight,
      weightSum,
      $(standardization),
      featuresStd,
      regParamL2,
      fitInterceptParam,
      $(eagerPersist))

    val optimizer = if ($(elasticNetParam) == 0.0 || $(regParam) == 0.0) {
      new VectorFreeLBFGS(
        maxIter = $(maxIter),
        m = $(numLBFGSCorrections),
        tolerance = $(tol),
        eagerPersist = $(eagerPersist)
      )
    } else {
      // with L1 regulazation, use Vector-free OWLQN optimizer
      if ($(standardization)) {
        new VectorFreeOWLQN(
          maxIter = $(maxIter),
          m = $(numLBFGSCorrections),
          l1RegValue = (regParamL1, fitInterceptParam),
          tolerance = $(tol),
          eagerPersist = $(eagerPersist)
        )
      } else {
        // If `standardization` is false, we still standardize the data
        // to improve the rate of convergence; as a result, we have to
        // perform this reverse standardization by penalizing each component
        // differently to get effectively the same objective function when
        // the training dataset is not standardized.
        val regParamL1DV = featuresStd.mapPartitionsWithIndex {
          (pid: Int, partFeatureStd: Vector) =>
            val blockSize = if (fitInterceptParam && pid == colBlocks - 1) {
              partFeatureStd.size + 1 // add element slot for intercept
            } else {
              partFeatureStd.size
            }
            val res = Array.fill(blockSize)(0.0)
            partFeatureStd.foreachActive(
              (index: Int, value: Double) =>
                res(index) = if (partFeatureStd(index) != 0.0) {
                  regParamL1 / partFeatureStd(index)
                } else {
                  0.0
                }
            )
            Vectors.dense(res)
        }
        new VectorFreeOWLQN(
          maxIter = $(maxIter),
          m = $(numLBFGSCorrections),
          l1Reg = regParamL1DV,
          tolerance = $(tol),
          eagerPersist = $(eagerPersist)
        )
      }
    }

    val initCoeffs: DistributedVector = if (fitInterceptParam) {
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
      DistributedVectors.zeros(
        sc, colsPerBlockParam, colBlocks, numFeatures + 1, initIntercept)
    } else {
      DistributedVectors.zeros(
        sc, colsPerBlockParam, colBlocks, numFeatures)
    }
    initCoeffs.persist(StorageLevel.MEMORY_AND_DISK, eager = $(eagerPersist))

    var state: optimizer.State = null
    val shouldCheckpoint: Int => Boolean = (iter) => {
      if (sc.checkpointDir.isEmpty && $(checkpointInterval) != -1)
        logWarning("Cannot checkpoint because checkpointDir is not set.")
      sc.checkpointDir.isDefined && $(checkpointInterval) != -1 &&
        (iter % $(checkpointInterval) == 0)
    }
    val states = optimizer.iterations(costFun, initCoeffs, shouldCheckpoint)

    while (states.hasNext) {
      val startTime = System.currentTimeMillis()
      state = states.next()
      val endTime = System.currentTimeMillis()
      logInfo(s"VLogisticRegression iter ${state.iter} finish, cost ${endTime - startTime}ms.")
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
     * Additionally, since the coefficients were laid out in column major order during training
     * to avoid extra computation, we convert them back to row major before passing them to the
     * model.
     * Note that the intercept in scaled space and original space is the same;
     * as a result, no scaling is needed.
     */
    val interceptValAccu = sc.doubleAccumulator
    val coeffs = rawCoeffs.zipPartitionsWithIndex(
      featuresStd, rawCoeffs.sizePerBlock, numFeatures
    ) {
      case (pid: Int, partCoeffs: Vector, partFeatursStd: Vector) =>
        val partFeatursStdArr = partFeatursStd.toDense.toArray
        val resArrSize =
          if (fitInterceptParam && pid == colBlocks - 1) partCoeffs.size - 1
          else partCoeffs.size
        val res = Array.fill(resArrSize)(0.0)
        partCoeffs.foreachActive { case (idx: Int, value: Double) =>
          val isIntercept =
            (fitInterceptParam && pid == colBlocks - 1 && idx == partCoeffs.size - 1)
          if (!isIntercept) {
            if (partFeatursStdArr(idx) != 0.0) {
              res(idx) = value / partFeatursStdArr(idx)
            }
          } else {
            interceptValAccu.add(value)
          }
        }
        Vectors.dense(res)
    }.compressed // OWLQN will return sparse model, so here compress it.
     .persist(StorageLevel.MEMORY_AND_DISK, eager = true)
    // here must eager persist the RDD, because we need the interceptValAccu value now.

    val interceptVal = interceptValAccu.value
    val model = copyValues(new VLogisticRegressionModel(uid, coeffs.toLocal, interceptVal))
    state.checkpointList.foreach(_.deleteCheckpoint())
    state.dispose(true)
    model
  }

  override def copy(extra: ParamMap): VLogisticRegression = defaultCopy(extra)
}

private[ml] class VBinomialLogisticCostFun(
    _features: VBlockMatrix,
    _numFeatures: Long,
    _numInstances: Long,
    _labelAndWeight: RDD[(Array[Double], Array[Double])],
    _weightSum: Double,
    _standardization: Boolean,
    _featuresStd: DistributedVector,
    _regParamL2: Double,
    _fitIntercept: Boolean,
    eagerPersist: Boolean) extends VDiffFunction(eagerPersist) {

  // Calculates both the value and the gradient at a point
  override def calculate(coeffs: DistributedVector): (Double, DistributedVector) = {

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
    val fitIntercept = _fitIntercept

    val lossAccu = features.blocks.sparkContext.doubleAccumulator

    assert(RDDUtils.isRDDPersisted(features.blocks))
    assert(coeffs.isPersisted)
    assert(RDDUtils.isRDDPersisted(labelAndWeight))

    val multipliers: RDD[Vector] = features.horizontalZipVec(coeffs) {
      (blockCoords: (Int, Int), matrix: SparseMatrix, partCoeffs: Vector) =>
        val intercept = if (fitIntercept && blockCoords._2 == colBlocks - 1) {
          // Get intercept from the last element of coefficients vector
          partCoeffs(partCoeffs.size - 1)
        } else 0.0
        val partMarginArr = Array.fill[Double](matrix.numRows)(intercept)
        matrix.foreachActive { case (i: Int, j: Int, v: Double) =>
          partMarginArr(i) += (partCoeffs(j) * v)
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

    // here must eager persist the RDD, because we need the lossAccu value now.
    val multipliersDV: DistributedVector =
      new DistributedVector(multipliers, rowsPerBlock, rowBlocks, numInstances)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = true)

    val lossSum = lossAccu.value / weightSum

    val grad: RDD[Vector] = features.verticalZipVec(multipliersDV) {
      (blockCoords: (Int, Int), matrix: SparseMatrix, partMultipliers: Vector) =>
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
    }.map { case ((rowBlockIdx: Int, colBlockIdx: Int), partGrads: Vector) =>
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

    val gradDV: DistributedVector = new DistributedVector(grad, colsPerBlock, colBlocks,
      if (fitIntercept) numFeatures + 1 else numFeatures
    ).persist(StorageLevel.MEMORY_AND_DISK, eager = eagerPersist)

    assert(featuresStd.isPersisted)

    // compute regularization for grad & objective value
    val lossRegAccu = gradDV.blocks.sparkContext.doubleAccumulator
    val gradDVWithReg: DistributedVector = if (standardization) {
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
              // If `standardization` is false, we still standardize the data
              // to improve the rate of convergence; as a result, we have to
              // perform this reverse standardization by penalizing each component
              // differently to get effectively the same objective function when
              // the training dataset is not standardized.
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

    // here must eager persist the RDD, because we need the lossRegAccu value now.
    gradDVWithReg.persist(StorageLevel.MEMORY_AND_DISK, eager = true)

    // because gradDVWithReg already eagerly persisted, now we can release multipliersDV & gradDV
    multipliersDV.unpersist()
    gradDV.unpersist()

    val regSum = lossRegAccu.value
    (lossSum + 0.5 * regParamL2 * regSum, gradDVWithReg)
  }
}

/**
 * Model produced by [[VLogisticRegression]].
 */
class VLogisticRegressionModel private[spark](
    override val uid: String,
    val coefficients: Vector,
    val intercept: Double)
  extends ProbabilisticClassificationModel[Vector, VLogisticRegressionModel]
  with VLogisticRegressionParams {

  /**
   * Margin (rawPrediction) for class label 1.
   * For binary classification only.
   */
  private val margin: Vector => Double = (features) => {
    BLAS.dot(features, coefficients) + intercept
  }

  /**
   * Score (probability) for class label 1.
   * For binary classification only.
   */
  private val score: Vector => Double = (features) => {
    val m = margin(features)
    1.0 / (1.0 + math.exp(-m))
  }

  override val numFeatures: Int = coefficients.size


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
      case _ => throw new RuntimeException("Unexcepted error in VLogisticRegressionModel: " +
        "raw2probabilitiesInPlace encountered SparseVector")
    }
  }

  override protected def predictRaw(features: Vector): Vector = {
    val m = margin(features)
    Vectors.dense(-m, m)
  }

  override def copy(extra: ParamMap): VLogisticRegressionModel = {
    val newModel = copyValues(new VLogisticRegressionModel(uid, coefficients, intercept), extra)
    newModel.setParent(parent)
  }
}
