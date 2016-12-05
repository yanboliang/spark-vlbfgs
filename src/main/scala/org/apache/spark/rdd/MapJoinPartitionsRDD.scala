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
import scala.reflect.ClassTag

import org.apache.spark._
import org.apache.spark.util.Utils

class MapJoinPartitionsPartition(
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

class MapJoinPartitionsRDD[A: ClassTag, B: ClassTag, V: ClassTag](
    sc: SparkContext,
    var idxF: (Int) => Array[Int],
    var f: (Int, Iterator[A], Array[(Int, Iterator[B])]) => Iterator[V],
    var rdd1: RDD[A],
    var rdd2: RDD[B])
  extends RDD[V](sc, Nil) {

  override val partitioner = None

  override def getPartitions: Array[Partition] = {
    val array = new Array[Partition](rdd1.partitions.length)
    for (s1 <- rdd1.partitions) {
      val idx = s1.index
      array(idx) = new MapJoinPartitionsPartition(idx, rdd1, rdd2, idxF(idx))
    }
    array
  }

  override def getDependencies: Seq[Dependency[_]] = List(
    new OneToOneDependency(rdd1),
    new NarrowDependency(rdd2) {
      override def getParents(partitionId: Int): Seq[Int] = {
        idxF(partitionId)
      }
    }
  )

  override def getPreferredLocations(s: Partition): Seq[String] = {
    val fp = firstParent[A]
    // println(s"pref loc: ${fp.preferredLocations(fp.partitions(s.index))}")
    fp.preferredLocations(fp.partitions(s.index))
  }

  override def compute(split: Partition, context: TaskContext): Iterator[V] = {
    val currSplit = split.asInstanceOf[MapJoinPartitionsPartition]
    f(currSplit.s1.index, rdd1.iterator(currSplit.s1, context),
      currSplit.s2Arr.map(s2 => (s2.index, rdd2.iterator(s2, context)))
    )
  }

  override def clearDependencies() {
    super.clearDependencies()
    rdd1 = null
    rdd2 = null
    idxF = null
    f = null
  }
}
