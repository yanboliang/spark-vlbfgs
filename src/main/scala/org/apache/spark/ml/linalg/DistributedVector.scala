package org.apache.spark.ml.linalg

import org.apache.spark.{Partitioner, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

class DistributedVector(
    val vecs: RDD[Vector],
    val sizePerPart: Int,
    val nParts: Int,
    val nSize: Long) {

  require(nParts > 0 && sizePerPart > 0 && nSize > 0 && (nParts - 1) * sizePerPart < nSize)

  var isPersisted = false

  def add(d: Double): DistributedVector = {
    val vecs2 = vecs.map((vec: Vector) => {
      Vectors.fromBreeze(vec.asBreeze + d)
    })
    new DistributedVector(vecs2, sizePerPart, nParts, nSize)
  }

  def addScalVec(a: Double, dv2: DistributedVector): DistributedVector = {
    require(sizePerPart == dv2.sizePerPart && nParts == dv2.nParts && nSize == dv2.nSize)

    val resVecs = vecs.zip(dv2.vecs).map {
      case (vec1: Vector, vec2: Vector) =>
        val vec3 = vec1.copy
        BLAS.axpy(a, vec2, vec3)
        vec3
    }
    new DistributedVector(resVecs, sizePerPart, nParts, nSize)
  }

  def add(dv2: DistributedVector): DistributedVector = {
    addScalVec(1.0, dv2)
  }

  def sub(dv2: DistributedVector): DistributedVector = {
    addScalVec(-1.0, dv2)
  }

  def dot(dv2: DistributedVector): Double = {
    require(sizePerPart == dv2.sizePerPart && nParts == dv2.nParts && nSize == dv2.nSize)
    vecs.zip(dv2.vecs).map {
      case (vec1: Vector, vec2: Vector) =>
        BLAS.dot(vec1, vec2)
    }.sum()
  }

  def scale(a: Double): DistributedVector = {
    val vecs2 = vecs.map((vec: Vector) => {
      val vec2 = vec.copy
      BLAS.scal(a, vec2)
      vec2
    })
    new DistributedVector(vecs2, sizePerPart, nParts, nSize)
  }

  def norm(): Double = {
    math.sqrt(vecs.map(Vectors.norm(_, 2)).map(x => x * x).sum())
  }
  def eagerPersist(storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): DistributedVector = {
    vecs.persist(storageLevel)
    vecs.count() // force eager cache.
    isPersisted = true
    this
  }

  def unpersist(): DistributedVector = {
    vecs.unpersist()
    isPersisted = false
    this
  }

  def toKVRdd: RDD[(Int, Vector)] = {
    vecs.mapPartitionsWithIndex{
      case (pid: Int, iter: Iterator[Vector]) =>
        iter.map(v => (pid, v))
    }
  }

  def zipPartitions(dv2: DistributedVector)(f: (Vector, Vector) => Vector) = {
    require(sizePerPart == dv2.sizePerPart && nParts == dv2.nParts && nSize == dv2.nSize)

    new DistributedVector(
      vecs.zip(dv2.vecs).map {
        case (vec1: Vector, vec2: Vector) =>
          f(vec1, vec2)
      }, sizePerPart, nParts, nSize)
  }

  def zipPartitions(dv2: DistributedVector, dv3: DistributedVector)(f: (Vector, Vector, Vector) => Vector) = {
    require(sizePerPart == dv2.sizePerPart && nParts == dv2.nParts && nSize == dv2.nSize)
    require(sizePerPart == dv3.sizePerPart && nParts == dv3.nParts && nSize == dv3.nSize)

    new DistributedVector(
      vecs.zip(dv2.vecs).zip(dv3.vecs).map {
        case ((vec1: Vector, vec2: Vector), vec3: Vector) =>
          f(vec1, vec2, vec3)
      }, sizePerPart, nParts, nSize)
  }

  def toLocal: Vector = {
    require(nSize < Int.MaxValue)
    val v = Array.concat(vecs.zipWithIndex().collect().sortWith(_._2 < _._2).map(_._1.toArray): _*)
    Vectors.dense(v)
  }
}

private[ml] class DistributedVectorPartitioner(val nParts: Int) extends Partitioner {
  require(nParts > 0)
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
  override def numPartitions: Int = nParts
}

private class AggrScalVec(var vec: Vector) extends Serializable {

  def this() = this(null)

  def add(scalVec: (Double, Vector)): AggrScalVec = {
    val a = scalVec._1
    val v2 = scalVec._2
    if (vec == null) {
      vec = Vectors.zeros(v2.size)
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

  def zeros(sc: SparkContext, sizePerPart: Int, nParts: Int, nSize: Long): DistributedVector = {
    val lastPartSize = (nSize - sizePerPart * (nParts - 1)).toInt
    val vecs = sc.parallelize(Array.tabulate(nParts)(x => (x, x)).toSeq, nParts)
      .mapPartitions{iter => {
        Thread.sleep(1000) // add this sleep time will help spread the task into different node.
        iter
      }}
      .partitionBy(new DistributedVectorPartitioner(nParts))
      .map{ case(idx: Int, idx2: Int) =>
          if (idx < nParts - 1) { Vectors.zeros(sizePerPart) }
          else { Vectors.zeros(lastPartSize) }
      }
    new DistributedVector(vecs, sizePerPart, nParts, nSize)
  }

  def combine(vlist: (Double, DistributedVector)*): DistributedVector = {
    require(vlist.nonEmpty)
    val vecsList = vlist.map{case (a: Double, v: DistributedVector) =>
      v.vecs.mapPartitionsWithIndex((pid: Int, iter: Iterator[Vector]) => {
        iter.map((v: Vector) => (pid, (a, v)))
      })
    }
    val firstDV = vlist(0)._2
    val nParts = firstDV.nParts
    val sizePerPart = firstDV.sizePerPart
    val nSize = firstDV.nSize
    val combinedVec = vecsList.head.context.union(vecsList).aggregateByKey(
      new AggrScalVec,
      new DistributedVectorPartitioner(nParts)
    )((asv: AggrScalVec, sv: (Double, Vector)) => asv.add(sv),
      (asv: AggrScalVec, asv2: AggrScalVec) => asv.merge(asv2)
    ).map{
      case (k, v) => v.vec
    }

    new DistributedVector(combinedVec, sizePerPart, nParts, nSize)
  }
}
