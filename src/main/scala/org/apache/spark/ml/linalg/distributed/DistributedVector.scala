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
import org.apache.spark.rdd.{RDD, VRDDFunctions}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.RDDUtils
import org.apache.spark.{Partitioner, SparkContext}

import scala.collection.mutable.ArrayBuffer

class DistributedVector(
    var blocks: RDD[Vector],
    val sizePerBlock: Int,
    val numBlocks: Int,
    val size: Long) extends Logging {

  require(numBlocks > 0 && sizePerBlock > 0 && size > 0 && (numBlocks - 1) * sizePerBlock < size)

  def add(d: Double): DistributedVector = {
    val blocks2 = blocks.map((vec: Vector) => {
      Vectors.fromBreeze(vec.asBreeze + d)
    })
    new DistributedVector(blocks2, sizePerBlock, numBlocks, size)
  }

  def addScalVec(a: Double, dv2: DistributedVector): DistributedVector = {
    require(sizePerBlock == dv2.sizePerBlock && numBlocks == dv2.numBlocks && size == dv2.size)

    val resBlocks = blocks.zip(dv2.blocks).map {
      case (vec1: Vector, vec2: Vector) =>
        val vec3 = if (vec1.isInstanceOf[DenseVector]) {
          vec1.copy
        } else {
          vec1.toDense
        }
        BLAS.axpy(a, vec2, vec3)
        vec3
    }
    new DistributedVector(resBlocks, sizePerBlock, numBlocks, size)
  }

  def add(dv2: DistributedVector): DistributedVector = {
    addScalVec(1.0, dv2)
  }

  def sub(dv2: DistributedVector): DistributedVector = {
    addScalVec(-1.0, dv2)
  }

  def dot(dv2: DistributedVector): Double = {
    require(sizePerBlock == dv2.sizePerBlock && numBlocks == dv2.numBlocks && size == dv2.size)
    blocks.zip(dv2.blocks).map {
      case (vec1: Vector, vec2: Vector) =>
        BLAS.dot(vec1, vec2)
    }.sum()
  }

  def scale(a: Double): DistributedVector = {
    val blocks2 = blocks.map((vec: Vector) => {
      val vec2 = vec.copy
      BLAS.scal(a, vec2)
      vec2
    })
    new DistributedVector(blocks2, sizePerBlock, numBlocks, size)
  }

  def norm(): Double = {
    math.sqrt(blocks.map(Vectors.norm(_, 2)).map(x => x * x).sum())
  }

  def persist(storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK, eager: Boolean = true)
      : DistributedVector = {
    blocks.persist(storageLevel)
    if (eager) {
      blocks.count() // force eager cache.
    }
    this
  }

  def unpersist(): DistributedVector = {
    blocks.unpersist()
    this
  }

  def toKVRdd: RDD[(Int, Vector)] = {
    blocks.mapPartitionsWithIndex{
      case (pid: Int, iter: Iterator[Vector]) =>
        iter.map(v => (pid, v))
    }
  }

  def mapPartitionsWithIndex(f: (Int, Vector) => Vector) = {
    new DistributedVector(
      blocks.mapPartitionsWithIndex{ (pid: Int, iter: Iterator[Vector]) =>
        iter.map(v => f(pid, v))
      }, sizePerBlock, numBlocks, size
    )
  }

  def compressed = mapPartitionsWithIndex((pid: Int, v: Vector) => v.compressed)

  def zipPartitions(dv2: DistributedVector)(f: (Vector, Vector) => Vector) = {
    require(sizePerBlock == dv2.sizePerBlock && numBlocks == dv2.numBlocks)

    new DistributedVector(
      blocks.zip(dv2.blocks).map {
        case (vec1: Vector, vec2: Vector) =>
          f(vec1, vec2)
      }, sizePerBlock, numBlocks, size)
  }

  def zipPartitionsWithIndex(dv2: DistributedVector, newSizePerPart: Int = 0, newSize: Long = 0)
                            (f: (Int, Vector, Vector) => Vector) = {
    require(numBlocks == dv2.numBlocks)

    new DistributedVector(
      blocks.zip(dv2.blocks).mapPartitionsWithIndex(
        (pid: Int, iter: Iterator[(Vector, Vector)]) => {
          iter.map {
            case (vec1: Vector, vec2: Vector) =>
              f(pid, vec1, vec2)
          }
        }),
      if (newSizePerPart == 0) sizePerBlock else newSizePerPart,
      numBlocks,
      if (newSize == 0) size else newSize)
  }

  def zipPartitions(dv2: DistributedVector, dv3: DistributedVector)
                   (f: (Vector, Vector, Vector) => Vector) = {
    require(sizePerBlock == dv2.sizePerBlock && numBlocks == dv2.numBlocks)
    require(sizePerBlock == dv3.sizePerBlock && numBlocks == dv3.numBlocks)

    new DistributedVector(
      blocks.zip(dv2.blocks).zip(dv3.blocks).map {
        case ((vec1: Vector, vec2: Vector), vec3: Vector) =>
          f(vec1, vec2, vec3)
      }, sizePerBlock, numBlocks, size)
  }

  def zipPartitionsWithIndex(dv2: DistributedVector, dv3: DistributedVector)
                   (f: (Int, Vector, Vector, Vector) => Vector) = {
    require(sizePerBlock == dv2.sizePerBlock && numBlocks == dv2.numBlocks)
    require(sizePerBlock == dv3.sizePerBlock && numBlocks == dv3.numBlocks)

    new DistributedVector(
      blocks.zip(dv2.blocks).zip(dv3.blocks).mapPartitionsWithIndex(
        (pid: Int, iter: Iterator[((Vector, Vector), Vector)]) => {
          iter.map {
            case ((vec1: Vector, vec2: Vector), vec3: Vector) =>
              f(pid, vec1, vec2, vec3)
          }
        }), sizePerBlock, numBlocks, size)
  }

  def toLocal: Vector = {
    require(size < Int.MaxValue)
    val indicesBuff = new ArrayBuffer[Int]
    val valueBuff = new ArrayBuffer[Double]
    blocks.zipWithIndex().collect().sortWith(_._2 < _._2).foreach {
      case (v: Vector, index: Long) =>
        val sv = v.toSparse
        indicesBuff ++= sv.indices.map(_ + index.toInt * sizePerBlock)
        valueBuff ++= sv.values
    }
    Vectors.sparse(size.toInt, indicesBuff.toArray, valueBuff.toArray).compressed
  }

  def isPersisted: Boolean = {
    RDDUtils.isRDDPersisted(blocks)
  }

  def checkpoint(isRDDAlreadyComputed: Boolean, eager: Boolean): RDD[Vector] = {
    assert(isPersisted)
    logInfo(s"checkpoint distributed vector ${blocks.id}")
    var oldBlocks: RDD[Vector] = null
    if (isRDDAlreadyComputed) {
      val checkpointBlocks = blocks.map(x => x)
      checkpointBlocks.persist().checkpoint()
      oldBlocks = blocks
      blocks = checkpointBlocks
    } else {
      blocks.checkpoint()
    }
    if (eager) {
      blocks.count() // eager checkpoint
    }
    // return old blocks (only when `isRDDAlreadyComputed`),
    // so that caller can unpersist the old RDD later.
    oldBlocks
  }

  def deleteCheckpoint(): Unit = {
    try {
      val checkpointFile = new Path(blocks.getCheckpointFile.get)
      checkpointFile.getFileSystem(blocks.sparkContext.hadoopConfiguration)
        .delete(checkpointFile, true)
    } catch {
      case e: Exception =>
        logWarning(s"delete checkpoint fail: RDD_${blocks.id}")
    }
  }
}

private[spark] class DistributedVectorPartitioner(val nParts: Int) extends Partitioner {
  require(nParts > 0)
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
  override def numPartitions: Int = nParts
}

private class AggrScalVec(var vec: DenseVector) extends Serializable {

  def this() = this(null)

  def add(scalVec: (Double, Vector)): AggrScalVec = {
    val a = scalVec._1
    val v2 = scalVec._2
    if (vec == null) {
      vec = Vectors.zeros(v2.size).toDense
    }
    BLAS.axpy(a, v2, vec)
    this
  }

  def merge(asv: AggrScalVec): AggrScalVec = {
    if (vec == null) {
      vec = asv.vec.copy
    } else {
      BLAS.axpy(1.0, asv.vec, vec)
    }
    this
  }

}

object DistributedVectors {

  def zeros(sc: SparkContext, sizePerPart: Int, nParts: Int, nSize: Long, lastVal: Double = 0.0)
    : DistributedVector = {
    val lastPartSize = (nSize - sizePerPart * (nParts - 1)).toInt
    val vecs = sc.parallelize(Array.tabulate(nParts)(x => (x, x)).toSeq, nParts)
      .mapPartitions{iter => {
        Thread.sleep(2000) // add this sleep time will help spread the task into different node.
        iter
      }}
      .partitionBy(new DistributedVectorPartitioner(nParts))
      .map{ case(idx: Int, idx2: Int) =>
          if (idx < nParts - 1) { Vectors.zeros(sizePerPart) }
          else {
            val vec = Vectors.zeros(lastPartSize)
            vec.toArray(lastPartSize - 1) = lastVal
            vec
          }
      }
    new DistributedVector(vecs, sizePerPart, nParts, nSize)
  }

  def combine(vlist: (Double, DistributedVector)*): DistributedVector = {
    require(vlist.nonEmpty)
    val vecsList = vlist.map{case (a: Double, v: DistributedVector) =>
      v.blocks.mapPartitionsWithIndex((pid: Int, iter: Iterator[Vector]) => {
        iter.map((v: Vector) => (pid, (a, v)))
      })
    }
    val firstDV = vlist(0)._2
    val nParts = firstDV.numBlocks
    val sizePerPart = firstDV.sizePerBlock
    val nSize = firstDV.size
    val combinedVec = vecsList.head.context.union(vecsList).aggregateByKey(
      new AggrScalVec,
      new DistributedVectorPartitioner(nParts)
    )((asv: AggrScalVec, sv: (Double, Vector)) => asv.add(sv),
      (asv: AggrScalVec, asv2: AggrScalVec) => asv.merge(asv2)
    ).map {
      case (k, v) => v.vec.asInstanceOf[Vector]
    }

    new DistributedVector(combinedVec, sizePerPart, nParts, nSize)
  }
}
