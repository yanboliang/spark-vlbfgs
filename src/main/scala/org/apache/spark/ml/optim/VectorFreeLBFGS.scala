package org.apache.spark.ml.optim

import breeze.optimize.{DiffFunction, StepSizeUnderflow, StrongWolfeLineSearch}
import breeze.util.Implicits.scEnrichIterator

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DistributedVector => DV, DistributedVectors => DVs}

class VectorFreeLBFGS(
    maxIter: Int = -1,
    m: Int = 7,
    tolerance: Double = 1E-9) extends Logging {
  import VectorFreeLBFGS._
  require(m > 0)

  val fvalMemory: Int = 20
  val relativeTolerance: Boolean = true

  type State = VectorFreeLBFGS.State

  protected def determineStepSize(
      state: State,
      fn: DVDiffFunction,
      direction: DV): (Double, VFLineSearchDiffFun) = {
    // using strong wolfe line search
    val x = state.x
    val grad = state.grad

    val lineSearchFn = fn.lineSearchDiffFunction(x, direction)
    val search = new StrongWolfeLineSearch(maxZoomIter = 10, maxLineSearchIter = 10)
    val alpha = search.minimize(lineSearchFn,
      if (state.iter == 0.0) 1.0 / direction.norm else 1.0)

    if (alpha * grad.norm < 1E-10) {
      throw new StepSizeUnderflow
    }
    (alpha, lineSearchFn)
  }

  protected def takeStep(state: State, dir: DV, stepSize: Double): DV = {
    state.x.addScalVec(stepSize, dir)
  }

  protected def adjust(newX: DV, newGrad: DV, newVal: Double): (Double, DV) = (newVal, newGrad)

  // convergence check here is exactly the same as the one in breeze LBFGS
  protected def checkConvergence(state: State): Int = {
    if (maxIter >= 0 && state.iter >= maxIter) {
      MaxIterationsReachedFlag
    } else if (
      state.fValHistory.length >= 2 &&
        (state.adjustedValue - state.fValHistory.max).abs
          <= tolerance * (if (relativeTolerance) state.initialAdjVal else 1.0)
    ) {
      FunctionValuesConvergedFlag
    } else if (
      state.adjustedGradient.norm <=
        math.max(tolerance * (if (relativeTolerance) state.adjustedValue else 1.0), 1E-8)
    ) {
      GradientConvergedFlag
    } else {
      NotConvergedFlag
    }
  }

  protected def initialState(fn: DVDiffFunction, init: DV): State = {
    val x = init
    val (value, grad) = fn.calculate(x)
    val (adjValue, adjGrad) = adjust(x, grad, value)
    State(x, value, grad, adjValue, adjGrad, 0, adjValue,
      IndexedSeq(Double.PositiveInfinity), NotConvergedFlag)
  }

  // the `fn` is responsible for output `grad` persist.
  // the `init` must be persisted before this interface called.
  def iterations(fn: DVDiffFunction, init: DV): Iterator[State] = {
    val history = new History(m)
    val state0 = initialState(fn, init)

    val infiniteIterations = Iterator.iterate(state0) { state =>
      try {
        val dir = history.computeDirection(state.x, state.adjustedGradient)
        val (stepSize, lineSearchDiffFun) = determineStepSize(state, fn, dir)

        val x = takeStep(state, dir, stepSize)
        val (value, grad) = fn.calculate(x)
        dir.unpersist()
        val (adjValue, adjGrad) = adjust(x, grad, value)

        val newState = State(x, value, grad, adjValue, adjGrad, state.iter + 1,
          state.initialAdjVal, (state.fValHistory :+ value).takeRight(fvalMemory),
          NotConvergedFlag)

        newState.convergenceFlag = checkConvergence(newState)
        newState
      } catch {
        case x: Exception =>
          state.convergenceFlag = SearchFailedFlag
          state
      }
    }

    infiniteIterations.takeUpToWhere { state =>
      val isFinished = state.convergenceFlag match {
        case NotConvergedFlag =>
          false
        case MaxIterationsReachedFlag =>
          logInfo("Vector-Free LBFGS reach max iterations.")
          true
        case FunctionValuesConvergedFlag =>
          logInfo("Vector-Free LBFGS function value converged.")
          true
        case GradientConvergedFlag =>
          logInfo("Vector-Free LBFGS gradient converged.")
          true
        case SearchFailedFlag =>
          logInfo("Vector-Free LBFGS search failed.")
          true
      }
      if (isFinished) history.dispose
      isFinished
    }
  }

  def minimize(fn: DVDiffFunction, init: DV): DV = {
    minimizeAndReturnState(fn, init).x
  }


  def minimizeAndReturnState(fn: DVDiffFunction, init: DV): State = {
    iterations(fn, init).last
  }

}

// Line Search DiffFunction
class VFLineSearchDiffFun(x: DV, direction: DV, outer: DVDiffFunction)
  extends DiffFunction[Double]{

  // calculates the value at a point
  override def valueAt(alpha: Double): Double = calculate(alpha)._1

  // calculates the gradient at a point
  override def gradientAt(alpha: Double): Double = calculate(alpha)._2

  // Calculates both the value and the gradient at a point
  def calculate(alpha: Double): (Double, Double) = {
    val (ff, grad) = outer.calculate(x.addScalVec(alpha, direction))
    ff -> (grad dot direction)
  }
}

trait DVDiffFunction { outer =>

  // calculates the gradient at a point
  def gradientAt(x: DV): DV = calculate(x)._2

  // calculates the value at a point
  def valueAt(x: DV): Double = calculate(x)._1

  final def apply(x: DV): Double = valueAt(x)

  // Calculates both the value and the gradient at a point
  def calculate(x: DV): (Double, DV)

  def lineSearchDiffFunction(x: DV, direction: DV): VFLineSearchDiffFun
    = new VFLineSearchDiffFun(x, direction, outer)
}

object VectorFreeLBFGS {

  val LBFGS_threadPool = new scala.concurrent.forkjoin.ForkJoinPool(100)

  val NotConvergedFlag = 0
  val MaxIterationsReachedFlag = 1
  val FunctionValuesConvergedFlag = 2
  val GradientConvergedFlag = 3
  val SearchFailedFlag = 4

  case class State(
      x: DV,
      value: Double,
      grad: DV,
      adjustedValue: Double,
      adjustedGradient: DV,
      iter: Int,
      initialAdjVal: Double,
      fValHistory: IndexedSeq[Double],
      var convergenceFlag: Int) {
  }

  case class History(m: Int) {
    require(m > 0)

    private var k = 0
    private val m1 = m + 1
    private val X: Array[DV] = new Array[DV](m1)
    private val G: Array[DV] = new Array[DV](m1)
    private val XXdot: Array[Array[Double]] = Array.ofDim[Double](m1, m1)
    private val GGdot: Array[Array[Double]] = Array.ofDim[Double](m1, m1)
    private val XGdot: Array[Array[Double]] = Array.ofDim[Double](m1, m1)

    def dispose = {
      for (i <- 0 to m) {
        if (X(i) != null) {
          X(i).unpersist()
          X(i) = null
        }
        if (G(i) != null) {
          G(i).unpersist()
          G(i) = null
        }
      }
    }

    private def shift(vv: Array[DV], v: DV): Unit = {
      if (vv(0) != null) vv(0).unpersist()
      for (i <- 0 until m) {
        vv(i) = vv(i + 1)
      }
      vv(m) = v
      // v.persist() do not need persist here because `v` has already persisted.
    }

    private def shift(VV: Array[Array[Double]]): Unit = {
      for (i <- 0 until m; j <- 0 until m) {
        VV(i)(j) = VV(i + 1)(j + 1)
      }
    }

    private def updateDotProduct(task: (String, Int, Int)): Unit = task match {
      case ("XX", i, j) =>
        val d = X(i).dot(X(j))
        XXdot(i)(j) = d
        XXdot(j)(i) = d
      case ("XG", i, j) =>
        XGdot(i)(j) = X(i).dot(G(j))
      case ("GG", i, j) =>
        val d = G(i).dot(G(j))
        GGdot(i)(j) = d
        GGdot(j)(i) = d
    }

    def computeDirection(newX: DV, newGrad: DV): DV = {
      // shift in new X & grad, oldest ones shift out
      shift(X, newX)
      shift(G, newGrad)

      // pre-compute dot-product
      shift(XXdot)
      shift(GGdot)
      shift(XGdot)

      val start = math.max(m - k, 0)
      val taskList =
        (start to m).map(i => ("XX", i, m)) ++
          (start to m).map(i => ("XG", i, m)) ++
          (start until m).map(i => ("XG", m, i)) ++
          (start to m).map(i => ("GG", i, m))
      val parTaskList = taskList.par
      parTaskList.tasksupport = new scala.collection.parallel.ForkJoinTaskSupport(LBFGS_threadPool)
      parTaskList.foreach(updateDotProduct)

      var dir: DV = null
      // compute direction, Vector-free two loop recursion, generate coefficients theta & tau
      dir = if (k == 0) {
        G(m).scale(-1).eagerPersist()
      } else {
        val theta = Array.fill(m1)(0.0)
        val tau = Array.fill(m1)(0.0)

        tau(m) = -1.0
        val alpha = new Array[Double](m)
        for (i <- (m - 1) to start by (-1)) {
          val i1 = i + 1
          var sum = 0.0
          for (j <- 0 to m) {
            sum += (XXdot(i1)(j) - XXdot(i)(j)) * theta(j) + (XGdot(i1)(j) - XGdot(i)(j)) * tau(j)
          }
          val a = sum / (XGdot(i1)(i1) - XGdot(i1)(i) - XGdot(i)(i1) + XGdot(i)(i))
          assert(!a.isNaN)
          alpha(i) = a
          tau(i + 1) -= a
          tau(i) += a
        }

        val mm1 = m - 1
        val scale = (XGdot(m)(m) - XGdot(m)(mm1) - XGdot(mm1)(m) + XGdot(mm1)(mm1)) /
          (GGdot(m)(m) - 2.0 * GGdot(m)(mm1) + GGdot(mm1)(mm1))
        for (i <- 0 to m) {
          theta(i) *= scale
          tau(i) *= scale
        }
        for (i <- start until m) {
          val i1 = i + 1
          var sum = 0.0
          for (j <- 0 to m) {
            sum += (XGdot(j)(i1) - XGdot(j)(i)) * theta(j) + (GGdot(i1)(j) - GGdot(i)(j)) * tau(j)
          }
          val b = alpha(i) - sum / (XGdot(i1)(i1) - XGdot(i1)(i) - XGdot(i)(i1) + XGdot(i)(i))
          assert(!b.isNaN)
          theta(i + 1) += b
          theta(i) -= b
        }

        DVs.combine((theta.toSeq.zip(X) ++ tau.toSeq.zip(G)).filter(_._2 != null): _*).eagerPersist()
      }
      k += 1
      dir
    }
  }
}