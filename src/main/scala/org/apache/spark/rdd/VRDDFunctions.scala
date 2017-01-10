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

import scala.reflect.ClassTag
import org.apache.spark.internal.Logging

private[spark] class VRDDFunctions[A](self: RDD[A])
    (implicit at: ClassTag[A])
  extends Logging with Serializable {

  def mapJoinPartition[B: ClassTag, V: ClassTag](rdd2: RDD[B], shuffleRdd2: Boolean = true)(
    idxF: (Int) => Array[Int],
    f: (Int, Iterator[A], Array[(Int, Iterator[B])]) => Iterator[V]
  ) = self.withScope{
    val sc = self.sparkContext
    val cleanIdxF = sc.clean(idxF)
    val cleanF = sc.clean(f)
    if (shuffleRdd2) {
      new MapJoinPartitionsRDDV2(sc, cleanIdxF, cleanF, self, rdd2)
    }
    else {
      logWarning("mapJoinPartition not shuffle RDD2")
      new MapJoinPartitionsRDD(sc, cleanIdxF, cleanF, self, rdd2)
    }
  }
}

private[spark] object VRDDFunctions {

  implicit def fromRDD[A: ClassTag](rdd: RDD[A]): VRDDFunctions[A] = {
    new VRDDFunctions(rdd)
  }

  def zipMultiRDDs[A: ClassTag, V: ClassTag](rddList: List[RDD[A]])
      (f: (List[Iterator[A]]) => Iterator[V]): RDD[V] = {
    assert(rddList.length > 1)
    rddList(0).withScope{
      val sc = rddList(0).sparkContext
      val cleanF = sc.clean(f)
      new MultiZippedPartitionsRDD[A, V](sc, cleanF, rddList)
    }
  }

}
