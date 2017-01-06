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

import org.apache.spark.{Partition, SparkContext, TaskContext}

import scala.reflect.ClassTag

private[spark] class MultiZippedPartitionsRDD[A: ClassTag, V: ClassTag](
    sc: SparkContext,
    var f: (List[Iterator[A]]) => Iterator[V],
    var rddList: List[RDD[A]],
    preservesPartitioning: Boolean = false)
  extends ZippedPartitionsBaseRDD[V](sc, rddList, preservesPartitioning) {

  override def compute(s: Partition, context: TaskContext): Iterator[V] = {
    val partitions = s.asInstanceOf[ZippedPartitionsPartition].partitions
    val iterList = rddList.zipWithIndex.map{ case (rdd: RDD[A], index: Int) =>
      rdd.iterator(partitions(index), context)
    }
    f(iterList)
  }

  override def clearDependencies() {
    super.clearDependencies()
    rddList = null
    f = null
  }
}
