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

package org.apache.spark.rdd

import java.io.{IOException, ObjectOutputStream}

import org.apache.spark.serializer.Serializer
import org.apache.spark.{TaskContext, _}
import org.apache.spark.util.Utils

import scala.reflect.ClassTag

class MapJoinPartitionsPartitionV2(
    idx: Int,
    @transient private val rdd1: RDD[_],
    @transient private val rdd2: RDD[_],
    s2IdxArr: Array[Int]) extends Partition {

  var s1 = rdd1.partitions(idx)
  var s2Arr = s2IdxArr.map(s2Idx => rdd2.partitions(s2Idx))
  override val index: Int = idx

  @throws(classOf[IOException])
  private def writeObject(oos: ObjectOutputStream): Unit = Utils.tryOrIOException {
    s1 = rdd1.partitions(idx)
    s2Arr = s2IdxArr.map(s2Idx => rdd2.partitions(s2Idx))
    oos.defaultWriteObject()
  }
}

class MapJoinPartitionsRDDV2[A: ClassTag, B: ClassTag, V: ClassTag](
    sc: SparkContext,
    var idxF: (Int) => Array[Int],
    var f: (Int, Iterator[A], Array[(Int, Iterator[B])]) => Iterator[V],
    var rdd1: RDD[A],
    var rdd2: RDD[B],
    preservesPartitioning: Boolean = false)
  extends RDD[V](sc, Nil) {

  var rdd2WithPid = rdd2.mapPartitionsWithIndex((pid, iter) => iter.map(x => (pid, x)))

  private val serializer: Serializer = SparkEnv.get.serializer

  override def getPartitions: Array[Partition] = {
    val array = new Array[Partition](rdd1.partitions.length)
    for (s1 <- rdd1.partitions) {
      val idx = s1.index
      array(idx) = new MapJoinPartitionsPartitionV2(idx, rdd1, rdd2, idxF(idx))
    }
    array
  }

  override def getDependencies: Seq[Dependency[_]] = List(
    new OneToOneDependency(rdd1),
    new ShuffleDependency[Int, B, B](
      rdd2WithPid.asInstanceOf[RDD[_ <: Product2[Int, B]]],
      new IdentityPartitioner(rdd2WithPid.getNumPartitions), serializer)
  )

  override def getPreferredLocations(s: Partition): Seq[String] = {
    val fp = firstParent[A]
    // println(s"pref loc: ${fp.preferredLocations(fp.partitions(s.index))}")
    fp.preferredLocations(fp.partitions(s.index))
  }

  override def compute(split: Partition, context: TaskContext): Iterator[V] = {
    val currSplit = split.asInstanceOf[MapJoinPartitionsPartitionV2]
    val rdd2Dep = dependencies(1).asInstanceOf[ShuffleDependency[Int, Any, Any]]
    f(currSplit.s1.index, rdd1.iterator(currSplit.s1, context),
      currSplit.s2Arr.map(s2 => (s2.index,
        SparkEnv.get.shuffleManager
          .getReader[Int, B](rdd2Dep.shuffleHandle, s2.index, s2.index + 1, context)
          .read().map(x => x._2)
        ))
    )
  }

  override def clearDependencies() {
    super.clearDependencies()
    rdd1 = null
    rdd2 = null
    rdd2WithPid = null
    idxF = null
    f = null
  }
}

private[spark] class IdentityPartitioner(val numParts: Int) extends Partitioner {
  require(numPartitions > 0)
  override def getPartition(key: Any): Int = key.asInstanceOf[Int]
  override def numPartitions: Int = numParts
}
