package org.apache.spark.rdd

import scala.reflect.ClassTag
import org.apache.spark.internal.Logging

private[spark] class VRDDFunctions[A](self: RDD[A])
    (implicit at: ClassTag[A])
  extends Logging with Serializable {

  def mapJoinPartition[B: ClassTag, V: ClassTag](rdd2: RDD[B])(
    idxF: (Int) => Array[Int],
    f: (Int, Iterator[A], Array[(Int, Iterator[B])]) => Iterator[V]
  ) = self.withScope{
    val sc = self.sparkContext
    val cleanIdxF = sc.clean(idxF)
    val cleanF = sc.clean(f)
    new MapJoinPartitionsRDD(sc, cleanIdxF, cleanF, self, rdd2)
  }
}

private[spark] object VRDDFunctions {

  implicit def fromRDD[A: ClassTag](rdd: RDD[A]): VRDDFunctions[A] = {
    new VRDDFunctions(rdd)
  }
}