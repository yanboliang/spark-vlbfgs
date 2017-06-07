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

import org.apache.spark.Partitioner
import org.apache.spark.internal.Logging
import org.apache.spark.util.collection.CompactBuffer

import scala.reflect.ClassTag

class VOrderedRDDFunctions[K, V](self: RDD[(K, V)])
    (implicit kt: ClassTag[K], vt: ClassTag[V], ord: Ordering[K])
  extends Logging with Serializable {

  def groupByKeyUsingSort(partitioner: Partitioner): RDD[(K, Iterable[V])] = {
    self.repartitionAndSortWithinPartitions(partitioner)
      .mapPartitions { (iter: Iterator[(K, V)]) =>
        new Iterator[(K, CompactBuffer[V])] {
          private var firstElemInNextGroup: (K, V) = null

          override def hasNext: Boolean = firstElemInNextGroup != null || iter.hasNext

          override def next(): (K, CompactBuffer[V]) = {
            if (firstElemInNextGroup == null) {
              firstElemInNextGroup = iter.next()
            }
            val key = firstElemInNextGroup._1
            val group = CompactBuffer[V](firstElemInNextGroup._2)
            firstElemInNextGroup = null
            var reachNewGroup = false
            while (iter.hasNext && !reachNewGroup) {
              val currElem = iter.next()
              if (currElem._1 == key) {
                group += currElem._2
              } else {
                firstElemInNextGroup = currElem
                reachNewGroup = true
              }
            }
            (key, group)
          }
        }
      }
  }
}

private[spark] object VOrderedRDDFunctions {

  implicit def fromRDD[K: ClassTag, V: ClassTag](rdd: RDD[(K, V)])(implicit ord: Ordering[K]):
      VOrderedRDDFunctions[K, V] = {
    new VOrderedRDDFunctions(rdd)
  }
}