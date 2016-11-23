package org.apache.spark.ml.classification

import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{DistributedVector => DV, DistributedVectors => DVs, _}
import org.apache.spark.ml.optim.{DVDiffFunction, VectorFreeLBFGS}
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
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
import org.apache.spark.util.DoubleAccumulator

/**
 * Params for vector-free logistic regression.
 */
private[classification] trait VLogisticRegressionParams
  extends ProbabilisticClassifierParams with HasRegParam with HasMaxIter
    with HasTol with HasStandardization with HasWeightCol with HasThreshold

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

  @Since("2.1.0")
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

    val partitionSize: Array[Long] = VUtils.computePartitionStartIndices(labelAndWeightRDD)
    val numInstances = partitionSize.sum
    val rowBlocks = VUtils.getNumBlocks(localRowsPerBlock, numInstances)

    val labelAndWeight: RDD[(Array[Double], Array[Double])] =
      VUtils.zipRDDWithIndex(partitionSize, labelAndWeightRDD)
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
      VUtils.zipRDDWithIndex(partitionSize, instances)
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
        .map { case ((blockRowIdx: Int, colBlockIdx: Int), iter: Iterable[(Int, SparseVector)]) =>
          val vecs = iter.toArray.sortWith(_._1 < _._1).map(_._2)
          val matrix = VUtils.vertcatSparseVectorIntoCSRMatrix(vecs)
          ((blockRowIdx, colBlockIdx), matrix)
        }

    val features: RDD[((Int, Int), SparseMatrix)] = VUtils.blockMatrixHorzZipVec(
        rawFeatures, featuresStd, gridPartitioner,
        (sm: SparseMatrix, partFeatureStdVector: Vector) => {
          val partFeatureStdArr = partFeatureStdVector.asInstanceOf[DenseVector].values
          val arrBuf = new ArrayBuffer[(Int, Int, Double)]()
          sm.foreachActive { case (i: Int, j: Int, value: Double) =>
            if (partFeatureStdArr(j) != 0 && value != 0) {
              arrBuf.append((j, i, value / partFeatureStdArr(j)))
            }
          }
          SparseMatrix.fromCOO(sm.numCols, sm.numRows, arrBuf).transpose
        }).persist(storageLevel)

    val costFun = new VFBinomialLogisticCostFun(
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
      featuresStd: DV,
      $(regParam))

    val optimizer = new VectorFreeLBFGS($(maxIter), 10, $(tol))
    val initCoeffs = DVs.zeros(sc, localColsPerBlock, colBlocks, numFeatures).eagerPersist()
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
        partCoeffs.foreachActive { case (idx: Int, value: Double) =>
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
    numFeatures: Long,
    colsPerBlock: Int,
    numInstances: Long,
    rowsPerBlock: Int,
    features: RDD[((Int, Int), SparseMatrix)],
    gridPartitioner: GridPartitionerV2,
    labelAndWeight: RDD[(Array[Double], Array[Double])],
    weightSum: Double,
    rowBlocks: Int,
    colBlocks: Int,
    standardization: Boolean,
    featuresStd: DV,
    regParamL2: Double) extends DVDiffFunction {

  import VLogisticRegression._

  // Calculates both the value and the gradient at a point
  override def calculate(coeffs: DV): (Double, DV) = {

    val lossAccu: DoubleAccumulator = features.sparkContext.doubleAccumulator
    val multipliers: RDD[Vector] = VUtils.blockMatrixHorzZipVec(features, coeffs, gridPartitioner,
      (matrix, partCoeffs) => {
        val partMarginArr = Array.fill[Double](matrix.numRows)(0.0)
        matrix.foreachActive { case (i: Int, j: Int, v: Double) =>
          partMarginArr(i) += (partCoeffs(j) * v)
        }
        new BDV(partMarginArr)
      }
    ).map(x => (x._1._1, x._2)).reduceByKey(new DistributedVectorPartitioner(rowBlocks), _ + _)
      .zip(labelAndWeight)
      .map { case ((rowIdx: Int, marginArr0: BDV[Double]), (labelArr: Array[Double], weightArr: Array[Double])) =>
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

    val multipliersDV: DV = new DV(multipliers, rowsPerBlock, rowBlocks, numInstances)
      .eagerPersist(storageLevel)

    val lossSum = lossAccu.value / weightSum
    val grad = VUtils.blockMatrixVertZipVec(features, multipliersDV, gridPartitioner,
      (matrix, partMultipliers) => {
        val partGradArr = Array.fill[Double](matrix.numCols)(0.0)
        matrix.foreachActive { case (i: Int, j: Int, v: Double) =>
          partGradArr(j) += (partMultipliers(i) * v)
        }
        new BDV[Double](partGradArr)
      }
    ).map(x => (x._1._2, x._2)).reduceByKey(new DistributedVectorPartitioner(colBlocks), _ + _)
      .map(bv => Vectors.fromBreeze(bv._2 / weightSum))
    val gradDV: DV = new DV(grad, colsPerBlock, colBlocks, numFeatures).eagerPersist(storageLevel)

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
class VLogisticRegressionModel private[spark](
    override val uid: String,
    val coefficients: DV) extends ProbabilisticClassificationModel[Vector, VLogisticRegressionModel]
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
