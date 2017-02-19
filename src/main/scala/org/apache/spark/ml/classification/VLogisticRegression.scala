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
    with HasFitIntercept with HasCheckpointInterval {

  // column number of each block in feature block matrix
  val colsPerBlock: IntParam = new IntParam(this, "colsPerBlock",
    "column number of each block in feature block matrix.", ParamValidators.gt(0))
  setDefault(colsPerBlock -> 10000)

  def getColsPerBlock: Int = $(colsPerBlock)

  // row number of each block in feature block matrix
  val rowsPerBlock: IntParam = new IntParam(this, "rowsPerBlock",
    "row number of each block in feature block matrix.", ParamValidators.gt(0))
  setDefault(rowsPerBlock -> 10000)

  def getRowsPerBlock: Int = $(rowsPerBlock)

  // row partition number of feature block matrix
  // equals to partition number of coefficient vector
  val rowPartitions: IntParam = new IntParam(this, "rowPartitions",
    "row partition number of feature block matrix.", ParamValidators.gt(0))
  setDefault(rowPartitions -> 10)

  def getRowPartitions: Int = $(rowPartitions)

  // column partition number of feature block matrix
  val colPartitions: IntParam = new IntParam(this, "colPartitions",
    "column partition number of feature block matrix.", ParamValidators.gt(0))
  setDefault(colPartitions -> 10)

  def getColPartitions: Int = $(colPartitions)

  // Whether to eager persist distributed vector.
  val eagerPersist: BooleanParam = new BooleanParam(this, "eagerPersist",
    "Whether to eager persist distributed vector.")
  setDefault(eagerPersist -> false)

  def getEagerPersist: Boolean = $(eagerPersist)

  // The number of corrections used in the LBFGS update.
  val numCorrections: IntParam = new IntParam(this, "numCorrections",
    "The number of corrections used in the LBFGS update.")
  setDefault(numCorrections -> 10)

  def getNumCorrections: Int = $(numCorrections)

  val generatingFeatureMatrixBuffer: IntParam = new IntParam(this, "generatingFeatureMatrixBuffer",
    "Buffer size when generating features block matrix.")
  setDefault(generatingFeatureMatrixBuffer -> 1000)

  def getGeneratingFeatureMatrixBuffer: Int = $(generatingFeatureMatrixBuffer)

  val rowPartitionSplitNumOnGeneratingFeatureMatrix: IntParam = new IntParam(this,
    "rowPartitionSplitsNumOnGeneratingFeatureMatrix",
    "row partition splits number on generating features matrix."
  )
  setDefault(rowPartitionSplitNumOnGeneratingFeatureMatrix -> 1)

  def getRowPartitionSplitNumOnGeneratingFeatureMatrix: Int =
    $(rowPartitionSplitNumOnGeneratingFeatureMatrix)

  val compressFeatureMatrix: BooleanParam = new BooleanParam(this,
    "compressFeatureMatrix",
    "compress feature matrix."
  )
  setDefault(compressFeatureMatrix -> false)

  def getCompressFeatureMatrix: Boolean = $(compressFeatureMatrix)
}

/**
 * Logistic regression.
 */
class VLogisticRegression(override val uid: String)
  extends ProbabilisticClassifier[Vector, VLogisticRegression, VLogisticRegressionModel]
    with VLogisticRegressionParams with Logging {

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

  def setGeneratingFeatureMatrixBuffer(size: Int): this.type =
    set(generatingFeatureMatrixBuffer, size)

  def setRowPartitionSplitNumOnGeneratingFeatureMatrix(numSplits: Int): this.type =
    set(rowPartitionSplitNumOnGeneratingFeatureMatrix, numSplits)

  def setCompressFeatureMatrix(value: Boolean): this.type =
    set(compressFeatureMatrix, value)

  override protected[spark] def train(dataset: Dataset[_]): VLogisticRegressionModel = {
    logInfo("Start to train VLogisticRegressionModel.")

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

    val generatingFeatureMatrixBufferParam = $(generatingFeatureMatrixBuffer)

    val labelAndWeight: RDD[(Array[Double], Array[Double])] =
      VUtils.zipRDDWithIndex(partitionSizes, labelsAndWeights)
        .map { case (rowIdx: Long, (label: Double, weight: Double)) =>
          val rowBlockIdx = (rowIdx / rowsPerBlockParam).toInt
          val inBlockIdx = (rowIdx % rowsPerBlockParam).toInt
          (rowBlockIdx, (inBlockIdx, label, weight))
        }
        .groupByKey(new DistributedVectorPartitioner(rowBlocks))
        .map { case (rowBlockIdx: Int, iter: Iterable[(Int, Double, Double)]) =>
          val tupleArr = iter.toArray.sortWith(_._1 < _._1)
          val labelArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._2)
          val weightArr = Array.tabulate(tupleArr.length)(idx => tupleArr(idx)._3)
          (labelArr, weightArr)
        }.persist(StorageLevel.MEMORY_AND_DISK)

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

    val histogram = labelSummarizer.histogram

    var localRowPartitions = $(rowPartitions)
    var localColPartitions = $(colPartitions)
    if (localRowPartitions > rowBlocks) localRowPartitions = rowBlocks
    if (localColPartitions > colBlocks) localColPartitions = colBlocks

    val gridPartitioner = new VGridPartitioner(
      rowBlocks,
      colBlocks,
      rowBlocks / localRowPartitions,
      colBlocks / localColPartitions
    )
    val rowSplitedGridPartitioner = gridPartitioner.getRowSplitedGridPartitioner(
     $(rowPartitionSplitNumOnGeneratingFeatureMatrix)
    )

    val lastBlockColSize = (numFeatures - (colBlocks - 1) * colsPerBlockParam).toInt
    val lastBlockRowSize = (numInstances - (rowBlocks - 1) * rowsPerBlockParam).toInt

    val compressFeatureMatrixParam = $(compressFeatureMatrix)
    var rawFeatureBlocks: RDD[((Int, Int), VMatrix)] =
      VUtils.zipRDDWithIndex(partitionSizes, instances)
        .mapPartitions { iter: Iterator[(Long, Instance)] =>
          new Iterator[Array[((Int, Int), (Array[Int], Array[Int], Array[Double]))]] {

            override def hasNext: Boolean = iter.hasNext

            override def next(): Array[((Int, Int), (Array[Int], Array[Int], Array[Double]))] = {
              val buffArr = Array.fill(colBlocks)(
                Tuple3(new ArrayBuffer[Int], new ArrayBuffer[Int], new ArrayBuffer[Double]))

              var shouldBreak = false
              var blockRowIndex = -1
              while (iter.hasNext && !shouldBreak) {
                val (rowIndex: Long, instance: Instance) = iter.next()
                if (blockRowIndex == -1) {
                  blockRowIndex = (rowIndex / rowsPerBlockParam).toInt
                }
                val inBlockRowIndex = (rowIndex % rowsPerBlockParam).toInt
                instance.features.foreachActive { (colIndex: Int, value: Double) =>
                  val blockColIndex = colIndex / colsPerBlockParam
                  val inBlockColIndex = colIndex % colsPerBlockParam
                  val COOBuffTuple = buffArr(blockColIndex)
                  COOBuffTuple._1 += inBlockRowIndex
                  COOBuffTuple._2 += inBlockColIndex
                  COOBuffTuple._3 += value
                }
                if (inBlockRowIndex == rowsPerBlockParam - 1 ||
                  inBlockRowIndex % generatingFeatureMatrixBufferParam ==
                    generatingFeatureMatrixBufferParam - 1
                ) {
                  shouldBreak = true
                }
              }
              buffArr.zipWithIndex.map { case (tuple: (ArrayBuffer[Int], ArrayBuffer[Int],
                  ArrayBuffer[Double]), blockColIndex: Int) =>
                ((blockRowIndex, blockColIndex),
                  (tuple._1.toArray, tuple._2.toArray, tuple._3.toArray))
              }
            }
          }.flatMap(_.toIterator)
        }
        .groupByKey(rowSplitedGridPartitioner)
        .map { case (coodinate: (Int, Int), cooList: Iterable[(Array[Int], Array[Int], Array[Double])]) =>
          val cooBuff = new ArrayBuffer[(Int, Int, Double)]
          cooList.foreach { case (rowIndices: Array[Int], colIndices: Array[Int], values: Array[Double]) =>
            var i = 0
            while (i < rowIndices.length) {
              cooBuff += Tuple3(rowIndices(i), colIndices(i), values(i))
              i += 1
            }
          }
          val numRows = if (coodinate._1 == rowBlocks - 1) lastBlockRowSize else rowsPerBlockParam
          val numCols = if (coodinate._2 == colBlocks - 1) lastBlockColSize else colsPerBlockParam
          (coodinate, VMatrices.COO(numRows, numCols, cooBuff, compressFeatureMatrixParam))
        }
    if ($(rowPartitionSplitNumOnGeneratingFeatureMatrix) > 1) {
      rawFeatureBlocks = rawFeatureBlocks.partitionBy(gridPartitioner)
    }

    val rawFeatures: VBlockMatrix = new VBlockMatrix(
      rowsPerBlockParam, colsPerBlockParam, rawFeatureBlocks, gridPartitioner)
      .persist(StorageLevel.MEMORY_AND_DISK_SER)

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

    val summarizerSeqOp = (summarizer: OptimMultivariateOnlineSummarizer,
                           tuple: (VMatrix, Vector)) => {
      val (block, weight) = tuple
      block.rowIter.zip(weight.toArray.toIterator).foreach {
        case (rowVector: Vector, weight: Double) =>
          summarizer.add(rowVector, weight)
      }
      summarizer
    }
    val summarizerCombineOp = (s1: OptimMultivariateOnlineSummarizer,
                               s2: OptimMultivariateOnlineSummarizer) => s1.merge(s2)

    val featureStdSummarizerRDD = rawFeatures.verticalZipVector(weightDV) {
      (blockCoordinate: (Int, Int), block: VMatrix, weight: Vector) => (block, weight)
    }.map { case ((rowBlockIdx: Int, colBlockIdx: Int), tuple: (VMatrix, Vector)) =>
      (colBlockIdx, tuple)
    }.aggregateByKey(
      new OptimMultivariateOnlineSummarizer(
        OptimMultivariateOnlineSummarizer.varianceMask),
      new DistributedVectorPartitioner(colBlocks)
    )(summarizerSeqOp, summarizerCombineOp)

    val featureStdRDD: RDD[Vector] = featureStdSummarizerRDD.values.map { summarizer =>
      Vectors.dense(summarizer.variance.toArray.map(math.sqrt))
    }

    val featuresStd = new DistributedVector(
      featureStdRDD, colsPerBlockParam, colBlocks, numFeatures)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = true)
    logInfo("feature std generated.")

    val features: VBlockMatrix = rawFeatures.horizontalZipVector2(featuresStd) {
      (blockCoordinate: (Int, Int), block: VMatrix, partFeaturesStd: Vector) =>
        val partFeatureStdArr = partFeaturesStd.asInstanceOf[DenseVector].values
        val arrBuf = new ArrayBuffer[(Int, Int, Double)]()
        block.foreachActive { case (i: Int, j: Int, value: Double) =>
          if (partFeatureStdArr(j) != 0 && value != 0) {
            arrBuf.append((i, j, value / partFeatureStdArr(j)))
          }
        }
        VMatrices.COO(block.numRows, block.numCols, arrBuf, compressFeatureMatrixParam)
    }.persist(StorageLevel.MEMORY_AND_DISK_SER)

    features.blocks.count() // force trigger persist
    rawFeatureBlocks.unpersist()
    if (handlePersistence) {
      dataset.unpersist()
    }
    logInfo("features block matrix generated.")

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
          l1RegValue = (regParamL1, fitInterceptParam),
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
        val regParamL1DV = featuresStd.mapPartitionsWithIndex {
          case (pid: Int, partFeatureStd: Vector) =>
            val sizeOfPart = if (fitInterceptParam && pid == colBlocks - 1) {
              partFeatureStd.size + 1 // add element slot for intercept
            } else {
              partFeatureStd.size
            }
            val res = Array.fill(sizeOfPart)(0.0)
            partFeatureStd.foreachActive { case (index: Int, value: Double) =>
              res(index) = if (partFeatureStd(index) != 0.0) {
                regParamL1 / partFeatureStd(index)
              } else {
                0.0
              }
            }
            Vectors.dense(res)
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
      val initIntercept = math.log(histogram(1) / histogram(0))
      DistributedVectors.zeros(sc, colsPerBlockParam, colBlocks, numFeatures + 1, initIntercept)
    } else {
      DistributedVectors.zeros(sc, colsPerBlockParam, colBlocks, numFeatures)
    }
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
    val interceptValAccu = sc.doubleAccumulator
    val coeffs = rawCoeffs.zipPartitionsWithIndex(featuresStd, rawCoeffs.sizePerPart, numFeatures) {
      case (pid: Int, partCoeffs: Vector, partFeatursStd: Vector) =>
        val partFeatursStdArr = partFeatursStd.toArray
        val sizeOfPart = if (fitInterceptParam && pid == colBlocks - 1) {
          partCoeffs.size - 1
        } else {
          partCoeffs.size
        }
        val res = Array.fill(sizeOfPart)(0.0)
        partCoeffs.foreachActive { case (idx: Int, value: Double) =>
          val isIntercept = fitInterceptParam && pid == colBlocks - 1 && idx == partCoeffs.size - 1
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
    val model = copyValues(new VLogisticRegressionModel(uid, coeffs.toLocalSparse, interceptVal))
    state.dispose(true)
    optimizer.dispose()
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
  override def calculate(coefficients: DistributedVector, checkAndMarkCheckpoint: DistributedVector => Unit):
      (Double, DistributedVector) = {

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
    assert(coefficients.isPersisted)
    assert(RDDUtils.isRDDPersisted(labelAndWeight))

    val multiplierRDD: RDD[Vector] = features.horizontalZipVector(coefficients) {
      (blockCoordinate: (Int, Int), block: VMatrix, partCoefficients: Vector) =>
        val intercept = if (fitIntercept && blockCoordinate._2 == colBlocks - 1) {
          // Get intercept from the last element of coefficients vector
          partCoefficients(partCoefficients.size - 1)
        } else {
          0.0
        }
        val partMarginArr = Array.fill[Double](block.numRows)(intercept)
        block.foreachActive { case (i: Int, j: Int, v: Double) =>
          partMarginArr(i) += (partCoefficients(j) * v)
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
    val multiplier: DistributedVector =
      new DistributedVector(multiplierRDD, rowsPerBlock, rowBlocks, numInstances)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = true)

    val lossSum = lossAccu.value / weightSum

    val gradientRDD: RDD[Vector] = features.verticalZipVector(multiplier) {
      (blockCoordinate: (Int, Int), block: VMatrix, partMultipliers: Vector) =>
        val partGradArr = if (fitIntercept && blockCoordinate._2 == colBlocks - 1) {
          val arr = Array.fill[Double](block.numCols + 1)(0.0)
          arr(arr.length - 1) = partMultipliers.toArray.sum
          arr
        } else {
          Array.fill[Double](block.numCols)(0.0)
        }
        block.foreachActive { case (i: Int, j: Int, v: Double) =>
          partGradArr(j) += (partMultipliers(i) * v)
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

    val gradient: DistributedVector = new DistributedVector(gradientRDD, colsPerBlock, colBlocks,
      if (fitIntercept) numFeatures + 1 else numFeatures
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
            val isIntercept = fitIntercept && pid == colBlocks - 1 && i == partCoefficients.size - 1
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
      gradient.zipPartitionsWithIndex(coefficients, featuresStd) {
        case (pid: Int, partGradients: Vector, partCoefficients: Vector, partFeaturesStd: Vector) =>
          var lossReg = 0.0
          val partGradArr = partGradients.toArray
          val partFeaturesStdArr = partFeaturesStd.toArray
          val res = Array.fill[Double](partGradArr.length)(0.0)
          partCoefficients.foreachActive { case (i: Int, value: Double) =>
            val isIntercept = fitIntercept && pid == colBlocks - 1 && i == partCoefficients.size - 1
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
