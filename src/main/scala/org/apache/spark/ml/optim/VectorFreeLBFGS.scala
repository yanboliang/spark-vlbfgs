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

package org.apache.spark.ml.optim

import breeze.optimize.{DiffFunction, StepSizeUnderflow, StrongWolfeLineSearch}
import breeze.util.Implicits.scEnrichIterator
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DistributedVector => DV, DistributedVectors => DVs}
import org.apache.spark.storage.StorageLevel

class VectorFreeLBFGS(
    maxIter: Int = -1,
    m: Int = 7,
    tolerance: Double = 1E-9,
    eagerPersist: Boolean = true,
    useNewHistoryClass: Boolean = true) extends Logging {

  import VectorFreeLBFGS._
  require(m > 0)

  val fvalMemory: Int = 20
  val relativeTolerance: Boolean = true

  type State = VectorFreeLBFGS.State
  type History = VectorFreeLBFGS.History

  protected def determineStepSize(
      state: State,
      fn: VDiffFunction,
      direction: DV): Double = {
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

    // release unused RDDs
    lineSearchFn.disposeLastResult()

    alpha
  }

  protected def takeStep(state: this.State, dir: DV, stepSize: Double): DV = {
    state.x.addScalVec(stepSize, dir)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = eagerPersist)
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

  protected def initialState(fn: VDiffFunction, init: DV): State = {
    val x = init
    val (value, grad) = fn.calculate(x)
    val (adjValue, adjGrad) = adjust(x, grad, value)
    State(x, value, grad, adjValue, adjGrad, 0, adjValue,
      IndexedSeq(Double.PositiveInfinity), NotConvergedFlag)
  }

  // VectorFreeOWLQN will override this method
  def chooseDescentDirection(history: this.History, state: this.State): DV = {
    val dir = history.computeDirection(state.x, state.grad, state.adjustedGradient)
    dir.persist(StorageLevel.MEMORY_AND_DISK, eager = eagerPersist)
  }

  // the `fn` is responsible for output `grad` persist.
  // the `init` must be persisted before this interface called.
  def iterations(fn: VDiffFunction, init: DV): Iterator[State] = {
    val history: History =
      if (useNewHistoryClass) new History2(m, eagerPersist)
      else new History1(m, eagerPersist)

    val state0 = initialState(fn, init)

    val infiniteIterations = Iterator.iterate(state0) { state =>
      try {
        val dir = chooseDescentDirection(history, state)
        assert(dir.isPersisted)
        val stepSize = determineStepSize(state, fn, dir)

        val x = takeStep(state, dir, stepSize)
        assert(x.isPersisted)
        val (value, grad) = fn.calculate(x)
        assert(grad.isPersisted)
        dir.unpersist()
        val (adjValue, adjGrad) = adjust(x, grad, value)
        assert(adjGrad.isPersisted)

        // in order to save memory, release ununsed adjGrad DV
        if (state.adjustedGradient != state.grad) {
          state.adjustedGradient.unpersist()
        }
        state.adjustedGradient = null

        val newState = State(x, value, grad, adjValue, adjGrad, state.iter + 1,
          state.initialAdjVal, (state.fValHistory :+ value).takeRight(fvalMemory),
          NotConvergedFlag)

        newState.convergenceFlag = checkConvergence(newState)
        newState
      } catch {
        case x: Exception =>
          state.convergenceFlag = SearchFailedFlag
          x.printStackTrace(System.err)
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
          logError("Vector-Free LBFGS search failed.")
          true
      }
      if (isFinished) history.dispose
      isFinished
    }
  }

  def minimize(fn: VDiffFunction, init: DV): DV = {
    minimizeAndReturnState(fn, init).x
  }


  def minimizeAndReturnState(fn: VDiffFunction, init: DV): State = {
    iterations(fn, init).last
  }

}

// Line Search DiffFunction
class VLineSearchDiffFun(x: DV, direction: DV, outer: VDiffFunction, eagerPersist: Boolean)
  extends DiffFunction[Double]{

  // store last point vector
  var lastX: DV = null

  // store last gradient vector
  var lastGrad: DV = null

  // calculates the value at a point
  override def valueAt(alpha: Double): Double = calculate(alpha)._1

  // calculates the gradient at a point
  override def gradientAt(alpha: Double): Double = calculate(alpha)._2

  // Calculates both the value and the gradient at a point
  def calculate(alpha: Double): (Double, Double) = {
    // release unused RDDs
    disposeLastResult()

    lastX = x.addScalVec(alpha, direction)
      .persist(StorageLevel.MEMORY_AND_DISK, eager = eagerPersist)
    val (ff, grad) = outer.calculate(lastX)

    assert(grad.isPersisted)

    lastGrad = grad

    ff -> (grad dot direction)
  }

  def disposeLastResult() = {
    // release last point vector
    if (lastX != null) {
      lastX.unpersist()
      lastX = null
    }

    // release last gradient vector
    if (lastGrad != null) {
      lastGrad.unpersist()
      lastGrad = null
    }
  }
}

abstract class VDiffFunction(eagerPersist: Boolean = true) { outer =>

  // calculates the gradient at a point
  def gradientAt(x: DV): DV = calculate(x)._2

  // calculates the value at a point
  def valueAt(x: DV): Double = calculate(x)._1

  final def apply(x: DV): Double = valueAt(x)

  // Calculates both the value and the gradient at a point
  def calculate(x: DV): (Double, DV)

  def lineSearchDiffFunction(x: DV, direction: DV): VLineSearchDiffFun
    = new VLineSearchDiffFun(x, direction, outer, eagerPersist)
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
      var adjustedGradient: DV,
      iter: Int,
      initialAdjVal: Double,
      fValHistory: IndexedSeq[Double],
      var convergenceFlag: Int) {
  }

  trait History {
    def dispose()
    def computeDirection(newX: DV, newGrad: DV, newAdjGrad: DV): DV
  }

  /**
   * Old version History implementation, may cause numeric stability problem.
   */
  case class History1(m: Int, eagerPersist: Boolean) extends History {
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
      val end = vv.length - 1
      if (vv(0) != null) vv(0).unpersist()
      for (i <- 0 until end) {
        vv(i) = vv(i + 1)
      }
      vv(end) = v
      // v.persist() do not need persist here because `v` has already persisted.
    }

    private def shift(VV: Array[Array[Double]]): Unit = {
      val end = VV.length - 1
      for (i <- 0 until end; j <- 0 until end) {
        VV(i)(j) = VV(i + 1)(j + 1)
      }
    }

    def computeDirection(newX: DV, newGrad: DV, newAdjGrad: DV): DV = {
      // shift in new X & grad, oldest ones shift out
      shift(X, newX)
      shift(G, newGrad)

      // pre-compute dot-product
      shift(XXdot)
      shift(GGdot)
      shift(XGdot)

      for (i <- 0 until m1) {
        if (X(i) != null) {
          assert(X(i).isPersisted)
        }
        if (G(i) != null) {
          assert(G(i).isPersisted)
        }
      }

      val hasAdjGrad = (newGrad != newAdjGrad)
      if (hasAdjGrad) {
        assert(newAdjGrad.isPersisted)
      }
      val XAdjGdot: Array[Double] = new Array[Double](m1)
      val GAdjGdot: Array[Double] = new Array[Double](m1)

      val start = math.max(m - k, 0)
      var taskList =
        (start to m).map(i => ("XX", i, m)) ++
          (start to m).map(i => ("XG", i, m)) ++
          (start until m).map(i => ("XG", m, i)) ++
          (start to m).map(i => ("GG", i, m))

      if (hasAdjGrad) {
        taskList = taskList ++
          (start to m).map(i => ("XAG", i, 0)) ++
          (start to m).map(i => ("GAG", i, 0))
      }

      val parTaskList = taskList.par
      parTaskList.tasksupport = new scala.collection.parallel.ForkJoinTaskSupport(LBFGS_threadPool)
      parTaskList.foreach(task => task match {
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
        case ("XAG", i, _) =>
          XAdjGdot(i) = X(i).dot(newAdjGrad)
        case ("GAG", i, _) =>
          GAdjGdot(i) = G(i).dot(newAdjGrad)
      })

      var dir: DV = null
      // compute direction, Vector-free two loop recursion, generate coefficients theta & tau
      dir = if (!hasAdjGrad) {
        if (k == 0) {
          G(m).scale(-1)
        } else {
          val theta = Array.fill(m1)(0.0)
          val tau = Array.fill(m1)(0.0)

          tau(m) = -1.0
          val alpha = new Array[Double](m)

          // for debug
          // val thetaRaw = Array.fill(m)(0.0)
          // val tauRaw = Array.fill(m)(0.0)

          for (i <- (m - 1) to start by (-1)) {
            val i1 = i + 1
            var sum = 0.0
            for (j <- 0 to m) {
              sum +=
                (XXdot(i1)(j) - XXdot(i)(j)) * theta(j) + (XGdot(i1)(j) - XGdot(i)(j)) * tau(j)
            }
            val a = sum / (XGdot(i1)(i1) - XGdot(i1)(i) - XGdot(i)(i1) + XGdot(i)(i))
            assert(!a.isNaN)
            alpha(i) = a
            tau(i + 1) -= a
            tau(i) += a

            // tauRaw(i) -= a // for debug
          }
          // println(s"alpha: ${alpha.mkString(",")}")
          // println(s"tauRaw: ${tauRaw.mkString(",")}")

          val mm1 = m - 1
          val scale = (XGdot(m)(m) - XGdot(m)(mm1) - XGdot(mm1)(m) + XGdot(mm1)(mm1)) /
            (GGdot(m)(m) - 2.0 * GGdot(m)(mm1) + GGdot(mm1)(mm1))

          // println(s"scale: ${scale}")

          for (i <- 0 to m) {
            theta(i) *= scale
            tau(i) *= scale
          }
          for (i <- start until m) {
            val i1 = i + 1
            var sum = 0.0
            for (j <- 0 to m) {
              sum +=
                (XGdot(j)(i1) - XGdot(j)(i)) * theta(j) + (GGdot(i1)(j) - GGdot(i)(j)) * tau(j)
            }
            val b = alpha(i) - sum / (XGdot(i1)(i1) - XGdot(i1)(i) - XGdot(i)(i1) + XGdot(i)(i))
            assert(!b.isNaN)
            theta(i + 1) += b
            theta(i) -= b

            // thetaRaw(i) += b // for debug
          }
          // println(s"theta: ${thetaRaw.mkString(",")}")
          DVs.combine((theta.toSeq.zip(X) ++ tau.toSeq.zip(G))
            .filter(_._2 != null): _*)
        }
      } else {
        // used in adjusting grad case, such as OWLQN
        println("VF-OWLQN compute direction")
        if (k == 0) {
          newAdjGrad.scale(-1)
        } else {
          val theta = Array.fill(m1)(0.0)
          val tau = Array.fill(m1)(0.0)
          var tauAdjGrad = -1.0

          val alpha = new Array[Double](m)
          for (i <- (m - 1) to start by (-1)) {
            val i1 = i + 1
            var sum = 0.0
            for (j <- 0 to m) {
              sum +=
                (XXdot(i1)(j) - XXdot(i)(j)) * theta(j) + (XGdot(i1)(j) - XGdot(i)(j)) * tau(j)
            }
            sum += (XAdjGdot(i1) - XAdjGdot(i)) * tauAdjGrad
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
          tauAdjGrad *= scale
          for (i <- start until m) {
            val i1 = i + 1
            var sum = 0.0
            for (j <- 0 to m) {
              sum +=
                (XGdot(j)(i1) - XGdot(j)(i)) * theta(j) + (GGdot(i1)(j) - GGdot(i)(j)) * tau(j)
            }
            sum += (GAdjGdot(i1) - GAdjGdot(i)) * tauAdjGrad
            val b = alpha(i) - sum / (XGdot(i1)(i1) - XGdot(i1)(i) - XGdot(i)(i1) + XGdot(i)(i))
            assert(!b.isNaN)
            theta(i + 1) += b
            theta(i) -= b
          }

          DVs.combine(
            (theta.toSeq.zip(X) ++ tau.toSeq.zip(G) ++ Seq((tauAdjGrad, newAdjGrad)))
            .filter(_._2 != null): _*)
        }
      }
      k += 1
      dir
    }
  }

  case class History2(m: Int, eagerPersist: Boolean) extends History{
    require(m > 0)

    private var k = 0

    private val S: Array[DV] = new Array[DV](m)
    private val Y: Array[DV] = new Array[DV](m)

    private val SSdot: Array[Array[Double]] = Array.ofDim[Double](m, m)
    private val YYdot: Array[Array[Double]] = Array.ofDim[Double](m, m)
    private val SYdot: Array[Array[Double]] = Array.ofDim[Double](m, m)

    private var lastX: DV = null
    private var lastGrad: DV = null

    def dispose = {
      for (i <- 0 until m) {
        if (S(i) != null) {
          S(i).unpersist()
          S(i) = null
        }
        if (Y(i) != null) {
          Y(i).unpersist()
          Y(i) = null
        }
      }
      if (lastX != null) {
        lastX.unpersist()
        lastX = null
      }
      if (lastGrad != null) {
        lastGrad.unpersist()
        lastGrad = null
      }
    }

    private def push(vv: Array[DV], v: DV): Unit = {
      val end = vv.length - 1
      if (vv(0) != null) vv(0).unpersist()
      for (i <- 0 until end) {
        vv(i) = vv(i + 1)
      }
      vv(end) = v
      // v.persist() do not need persist here because `v` has already persisted.
    }

    private def shift(VV: Array[Array[Double]]): Unit = {
      val end = VV.length - 1
      for (i <- 0 until end; j <- 0 until end) {
        VV(i)(j) = VV(i + 1)(j + 1)
      }
    }

    // In LBFGS newAdjGrad == newGrad, but in OWLQN, newAdjGrad contains L1 pseudo-gradient
    // Note: The approximate Hessian computed in LBFGS must use `grad` without L1 pseudo-gradient
    def computeDirection(newX: DV, newGrad: DV, newAdjGrad: DV): DV = {
      val dir = if (k == 0) {
        lastX = newX
        lastGrad = newGrad
        newAdjGrad.scale(-1)
      } else {
        val newSYTaskList = Array("S", "Y").par
        newSYTaskList.tasksupport
          = new scala.collection.parallel.ForkJoinTaskSupport(LBFGS_threadPool)

        var newS: DV = null
        var newY: DV = null
        newSYTaskList.foreach(task => task match {
          case "S" =>
            newS = newX.sub(lastX).persist(StorageLevel.MEMORY_AND_DISK, eager = eagerPersist)
          case "Y" =>
            newY = newGrad.sub(lastGrad).persist(StorageLevel.MEMORY_AND_DISK, eager = eagerPersist)
        })

        // now we can release `lastX` and `lastGrad`
        lastX.unpersist()
        lastGrad.unpersist()

        lastX = newX
        lastGrad = newGrad

        // push `newS` and `newY` into LBFGS S & Y vector history.
        push(S, newS)
        push(Y, newY)

        // calculate dot products between all `S` and `Y` vectors
        shift(SSdot)
        shift(SYdot)
        shift(YYdot)

        val SGdot: Array[Double] = new Array[Double](m)
        val YGdot: Array[Double] = new Array[Double](m)

        val start = math.max(m - k, 0)
        val mm1 = m - 1
        val dotProductTaskList = (
          (start to mm1).map(i => ("SS", i, mm1)) ++
          (start to mm1).map(i => ("SY", i, mm1)) ++
          (start until mm1).map(i => ("SY", mm1, i)) ++
          (start to mm1).map(i => ("YY", i, mm1)) ++
          (start to mm1).map(i => ("SG", i, 0)) ++
          (start to mm1).map(i => ("YG", i, 0))
        ).par

        dotProductTaskList.tasksupport
          = new scala.collection.parallel.ForkJoinTaskSupport(LBFGS_threadPool)
        dotProductTaskList.foreach(task => task match {
          case ("SS", i, j) =>
            val d = S(i).dot(S(j))
            SSdot(i)(j) = d
            SSdot(j)(i) = d
          case ("SY", i, j) =>
            SYdot(i)(j) = S(i).dot(Y(j))
          case ("YY", i, j) =>
            val d = Y(i).dot(Y(j))
            YYdot(i)(j) = d
            YYdot(j)(i) = d
          case ("SG", i, _) =>
            SGdot(i) = S(i).dot(newAdjGrad)
          case ("YG", i, _) =>
            YGdot(i) = Y(i).dot(newAdjGrad)
        })

        val theta = Array.fill(m)(0.0)
        val tau = Array.fill(m)(0.0)
        var tauAdjGrad = -1.0

        val alpha = new Array[Double](m)
        for (i <- mm1 to start by (-1)) {
          var sum = 0.0
          for (j <- 0 until m) {
            sum +=
              (SSdot(i)(j) * theta(j) + SYdot(i)(j) * tau(j))
          }
          sum += SGdot(i) * tauAdjGrad
          val a = sum / SYdot(i)(i)
          assert(!a.isNaN)
          alpha(i) = a
          tau(i) -= a
        }

        // println(s"alpha: ${alpha.mkString(",")}")
        // println(s"tau: ${tau.mkString(",")}")

        val scale = SYdot(mm1)(mm1) / YYdot(mm1)(mm1)
        // println(s"scale: ${scale}")

        for (i <- 0 until m) {
          theta(i) *= scale
          tau(i) *= scale
        }
        tauAdjGrad *= scale
        for (i <- start to mm1) {
          var sum = 0.0
          for (j <- 0 until m) {
            sum +=
              (SYdot(j)(i) * theta(j) + YYdot(i)(j) * tau(j))
          }
          sum += YGdot(i) * tauAdjGrad
          val b = alpha(i) - sum / SYdot(i)(i)
          assert(!b.isNaN)
          theta(i) += b
        }

        // for debug
        /*
        println("cached rdd (S & Y) ID:")
        for (i <- 0 until m) {
          if (S(i) != null) print(" " + S(i).vecs.id)
          if (Y(i) != null) print(" " + Y(i).vecs.id)
        }
        println()
        println("lastX & lastGrad")
        println(s"${lastX.vecs.id} ${lastGrad.vecs.id}")
        */

        // println(s"theta: ${theta.mkString(",")}")
        DVs.combine(
          (theta.toSeq.zip(S) ++ tau.toSeq.zip(Y) ++ Seq((tauAdjGrad, newAdjGrad)))
            .filter(_._2 != null): _*)
      }
      k += 1
      dir
    }
  }
}
