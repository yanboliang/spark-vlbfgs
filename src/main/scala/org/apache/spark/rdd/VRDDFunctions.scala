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

import java.nio.ByteBuffer

import org.apache.spark.{Partitioner, SparkEnv}

import scala.reflect.ClassTag
import org.apache.spark.util.collection.OpenHashMap
import org.apache.spark.internal.Logging

import org.apache.spark.util.SizeEstimator

private[spark] class VRDDFunctions[A](self: RDD[A])
    (implicit at: ClassTag[A])
  extends Logging with Serializable {

  def mapJoinPartition[B: ClassTag, V: ClassTag](rdd2: RDD[B], shuffleRdd2: Boolean = true)(
    idxF: (Int) => Array[Int],
    f: (Int, Iterator[A], Array[(Int, Iterator[B])]) => Iterator[V]
  ): RDD[V] = self.withScope{
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

  // This method is only used for testing
  def addIterForGC(): RDD[A] = {
    if (System.getProperty("GCTest.addGCIter", "true").toBoolean) {
      self.mapPartitions { iter: Iterator[A] =>
        System.gc()
        var isFirstElem = true
        new Iterator[A] {
          override def hasNext: Boolean = iter.hasNext

          override def next(): A = {
            val elem = iter.next()
            if (isFirstElem) {
              System.gc()
              isFirstElem = false
            }
            elem
          }
        }
      }
    } else {
      self
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

class VPairRDDFunctions[K, V](self: RDD[(K, V)])
    (implicit kt: ClassTag[K], vt: ClassTag[V])
  extends Logging with Serializable {

  // This method is only used for testing
  def aggregateByKeyInMemory[U: ClassTag](zeroValue: U, partitioner: Partitioner)(
      seqOp: (U, V) => U, combOp: (U, U) => U): RDD[(K, U)] = self.withScope {

    // Serialize the zero value to a byte array so that we can get a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    lazy val cachedSerializer = SparkEnv.get.serializer.newInstance()
    val createZero = () => cachedSerializer.deserialize[U](ByteBuffer.wrap(zeroArray))

    self.mapPartitions { iter: Iterator[(K, V)] =>
      org.apache.spark.ml.util.VUtils.printUsedMemory("start aggr partition.")
      val hashMap = new OpenHashMap[K, U]()
      var iterIndex = 0
      iter.foreach { case (key: K, value: V) =>
        val combinedValue = if (hashMap.contains(key)) {
          hashMap(key)
        } else {
          createZero()
        }
        hashMap.update(key, seqOp(combinedValue, value))
        System.err.println(s"thread: ${Thread.currentThread().getId} idx: $iterIndex, " +
          s"estimate aggr map: ${SizeEstimator.estimate(hashMap)}")
        iterIndex += 1
      }
      val res = hashMap.toIterator
      org.apache.spark.ml.util.VUtils.printUsedMemory("End aggr partition.")
      res
    }.partitionBy(partitioner).mapPartitions { iter: Iterator[(K, U)] =>
      val hashMap = new OpenHashMap[K, U]()
      iter.foreach { case (key: K, value: U) =>
        if (hashMap.contains(key)) {
          val combinedValue = hashMap(key)
          hashMap.update(key, combOp(combinedValue, value))
        } else {
          hashMap.update(key, value)
        }
      }
      hashMap.toIterator
    }
  }
}

private[spark] object VPairRDDFunctions {

  implicit def fromRDD[K: ClassTag, V: ClassTag](rdd: RDD[(K, V)]): VPairRDDFunctions[K, V] = {
    new VPairRDDFunctions(rdd)
  }
}

