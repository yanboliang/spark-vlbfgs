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

package org.apache.spark.ml.linalg.distributed

import org.apache.hadoop.fs.Path
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.RDDUtils
import org.apache.spark.{Partitioner, SparkContext}

import scala.collection.mutable.ArrayBuffer

class DistributedVector(
    var values: RDD[Vector],
    val sizePerPart: Int,
    val numPartitions: Int,
    val size: Long) extends Logging {

  require(numPartitions > 0 && sizePerPart > 0 && size > 0
    && (numPartitions - 1) * sizePerPart < size)

  def add(d: Double): DistributedVector = {
    val values2 = values.map { case vec: Vector =>
      Vectors.fromBreeze(vec.asBreeze + d)
    }
    new DistributedVector(values2, sizePerPart, numPartitions, size)
  }

  def addScaledVector(a: Double, other: DistributedVector): DistributedVector = {

    require(sizePerPart == other.sizePerPart && numPartitions == other.numPartitions
      && size == other.size)

    val resValues = values.zip(other.values).map { case (vec1: Vector, vec2: Vector) =>
      val vec3 = if (vec1.isInstanceOf[DenseVector]) {
        vec1.copy
      } else {
        vec1.toDense
      }
      BLAS.axpy(a, vec2, vec3)
      vec3
    }
    new DistributedVector(resValues, sizePerPart, numPartitions, size)
  }

  def add(other: DistributedVector): DistributedVector = {
    addScaledVector(1.0, other)
  }

  def sub(other: DistributedVector): DistributedVector = {
    addScaledVector(-1.0, other)
  }

  def dot(other: DistributedVector): Double = {

    require(sizePerPart == other.sizePerPart && numPartitions == other.numPartitions
      && size == other.size)

    values.zip(other.values).map { case (vec1: Vector, vec2: Vector) =>
        BLAS.dot(vec1, vec2)
    }.sum()
  }

  def scale(a: Double): DistributedVector = {
    val values2 = values.map { case (vec: Vector) =>
      val vec2 = vec.copy
      BLAS.scal(a, vec2)
      vec2
    }
    new DistributedVector(values2, sizePerPart, numPartitions, size)
  }

  /**
   * Returns the L^2-norm of this vector.
    **/
  def norm(): Double = {
    math.sqrt(values.map(Vectors.norm(_, 2)).map(x => x * x).sum())
  }

  def persist(
      storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      eager: Boolean = true): DistributedVector = {
    values.persist(storageLevel)
    if (eager) {
      values.count() // force eager cache.
    }
    this
  }

  def unpersist(): DistributedVector = {
    values.unpersist()
    this
  }

  def toKVRDD: RDD[(Int, Vector)] = {
    values.mapPartitionsWithIndex { case (pid: Int, iter: Iterator[Vector]) =>
      iter.map(v => (pid, v))
    }
  }

  def mapPartitionsWithIndex(f: (Int, Vector) => Vector): DistributedVector = {
    new DistributedVector(
      values.mapPartitionsWithIndex { case (pid: Int, iter: Iterator[Vector]) =>
        iter.map(v => f(pid, v))
      }, sizePerPart, numPartitions, size)
  }

  def compressed = new DistributedVector(
    values.map(_.compressed), sizePerPart,numPartitions, size)

  def zipPartitions(other: DistributedVector)(f: (Vector, Vector) => Vector): DistributedVector = {

    require(sizePerPart == other.sizePerPart && numPartitions == other.numPartitions)

    new DistributedVector(
      values.zip(other.values).map { case (vec1: Vector, vec2: Vector) =>
        f(vec1, vec2)
      }, sizePerPart, numPartitions, size)
  }

  def zipPartitionsWithIndex(
      other: DistributedVector,
      newSizePerPart: Int = 0,
      newSize: Long = 0)
      (f: (Int, Vector, Vector) => Vector): DistributedVector = {

    require(numPartitions == other.numPartitions)

    new DistributedVector(
      values.zip(other.values).mapPartitionsWithIndex {
        case (pid: Int, iter: Iterator[(Vector, Vector)]) =>
          iter.map { case (vec1: Vector, vec2: Vector) =>
            f(pid, vec1, vec2)
          }
        },
      if (newSizePerPart == 0) sizePerPart else newSizePerPart,
      numPartitions,
      if (newSize == 0) size else newSize
    )
  }

  def zipPartitions(
      dv2: DistributedVector,
      dv3: DistributedVector)
      (f: (Vector, Vector, Vector) => Vector): DistributedVector = {

    require(sizePerPart == dv2.sizePerPart && numPartitions == dv2.numPartitions)
    require(sizePerPart == dv3.sizePerPart && numPartitions == dv3.numPartitions)

    new DistributedVector(
      values.zip(dv2.values).zip(dv3.values).map {
        case ((vec1: Vector, vec2: Vector), vec3: Vector) =>
          f(vec1, vec2, vec3)
      }, sizePerPart, numPartitions, size)
  }

  def zipPartitionsWithIndex(
      dv2: DistributedVector,
      dv3: DistributedVector)
      (f: (Int, Vector, Vector, Vector) => Vector): DistributedVector = {

    require(sizePerPart == dv2.sizePerPart && numPartitions == dv2.numPartitions)
    require(sizePerPart == dv3.sizePerPart && numPartitions == dv3.numPartitions)

    new DistributedVector(
      values.zip(dv2.values).zip(dv3.values).mapPartitionsWithIndex {
        case (pid: Int, iter: Iterator[((Vector, Vector), Vector)]) =>
          iter.map { case ((vec1: Vector, vec2: Vector), vec3: Vector) =>
            f(pid, vec1, vec2, vec3)
          }
        }, sizePerPart, numPartitions, size)
  }

  def zipPartitionsWithIndex(
      dv2: DistributedVector,
      dv3: DistributedVector,
      newSizePerPart: Int,
      newSize: Long)
      (f: (Int, Vector, Vector, Vector) => Vector): DistributedVector = {

    require(numPartitions == dv2.numPartitions)
    require(numPartitions == dv3.numPartitions)

    new DistributedVector(
      values.zip(dv2.values).zip(dv3.values).mapPartitionsWithIndex {
        case (pid: Int, iter: Iterator[((Vector, Vector), Vector)]) =>
          iter.map { case ((vec1: Vector, vec2: Vector), vec3: Vector) =>
            f(pid, vec1, vec2, vec3)
          }
      }, newSizePerPart, numPartitions, newSize)
  }

  // transform into local sparse vector, optimized to save memory in best.
  def toLocalSparse: Vector = {
    require(size < Int.MaxValue)
    val numNonzeros = values.map(_.numNonzeros).sum().toInt
    val vecIndices = new Array[Int](numNonzeros)
    val vecValues = new Array[Double](numNonzeros)

    var pos = 0
    values.toLocalIterator.zipWithIndex.foreach { case (vec: Vector, pid: Int) =>
      vec.foreachActive { case (index: Int, value: Double) =>
        if (value != 0.0) {
          vecIndices(pos) = index + pid * sizePerPart
          vecValues(pos) = value
          pos += 1
        }
      }
    }
    Vectors.sparse(size.toInt, vecIndices, vecValues)
  }

  def toLocal: Vector = {
    toLocalSparse.compressed
  }

  def isPersisted: Boolean = {
    RDDUtils.isRDDPersisted(values)
  }

  def checkpoint(): DistributedVector = {
    if (values.sparkContext.checkpointDir.isDefined) {
      logInfo(s"checkpoint distributed vector ${values.id}")
      values.checkpoint()
    } else {
      logWarning(s"checkpointDir not set, checkpoint failed.")
    }
    this
  }

  def isCheckpointed: Boolean = values.isCheckpointed

  def deleteCheckpoint(): Unit = {
    try {
      val checkpointFile = new Path(values.getCheckpointFile.get)
      checkpointFile.getFileSystem(values.sparkContext.hadoopConfiguration)
        .delete(checkpointFile, true)
    } catch {
      case e: Exception =>
        logWarning(s"delete checkpoint fail: RDD_${values.id}")
    }
  }
}

private[spark] class DistributedVectorPartitioner(val nParts: Int) extends Partitioner {

  require(nParts > 0)
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
  override def numPartitions: Int = nParts
}

private class ScaledVectorAggregator extends Serializable {

  var values: DenseVector = null

  def add(instance: (Double, Vector)): ScaledVectorAggregator = {
    val a = instance._1
    val v = instance._2

    if (values == null) {
      values = new DenseVector(Array.ofDim(v.size))
    }

    BLAS.axpy(a, v, values)
    this
  }

  def merge(other: ScaledVectorAggregator): ScaledVectorAggregator = {
    if (values == null) {
      values = other.values.copy
    } else {
      BLAS.axpy(1.0, other.values, values)
    }
    this
  }
}

object DistributedVectors {

  def zeros(
      sc: SparkContext,
      sizePerPart: Int,
      numPartitions: Int,
      size: Long,
      lastValue: Double = 0.0): DistributedVector = {
    val lastPartSize = (size - sizePerPart * (numPartitions - 1)).toInt
    val values = sc.parallelize(Array.tabulate(numPartitions)(x => (x, x)).toSeq, numPartitions)
      .mapPartitions { iter =>
        Thread.sleep(2000) // add this sleep time will help spread the task into different node.
        iter
      }
      .partitionBy(new DistributedVectorPartitioner(numPartitions))
      .map { case (idx: Int, idx2: Int) =>
        if (idx < numPartitions - 1) {
          Vectors.zeros(sizePerPart)
        } else {
          val vec = Vectors.zeros(lastPartSize)
          vec.toArray(lastPartSize - 1) = lastValue
          vec
        }
      }
    new DistributedVector(values, sizePerPart, numPartitions, size)
  }

  def combine(scaledVectors: (Double, DistributedVector)*): DistributedVector = {

    require(scaledVectors.nonEmpty)

    val rddList = scaledVectors.map { case (a: Double, v: DistributedVector) =>
      v.values.mapPartitionsWithIndex { case (pid: Int, iter: Iterator[Vector]) =>
        iter.map { case (v: Vector) => (pid, (a, v)) }
      }
    }
    val firstDV = scaledVectors(0)._2
    val numPartitions = firstDV.numPartitions
    val sizePerPart = firstDV.sizePerPart
    val size = firstDV.size

    val combinedValues = rddList.head.context.union(rddList)
      .aggregateByKey(
      new ScaledVectorAggregator,
      new DistributedVectorPartitioner(numPartitions)
    )((sva: ScaledVectorAggregator, instance: (Double, Vector)) => sva.add(instance),
      (sva1: ScaledVectorAggregator, sva2: ScaledVectorAggregator) => sva1.merge(sva2)
    ).map { case (pid: Int, sva: ScaledVectorAggregator) =>
      sva.values.asInstanceOf[Vector]
    }

    new DistributedVector(combinedValues, sizePerPart, numPartitions, size)
  }
}
