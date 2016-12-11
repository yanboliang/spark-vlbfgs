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

package org.apache.spark.util

import org.apache.spark.SparkEnv
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.{RDDBlockId, StorageLevel}

private[spark] object RDDUtils {

  /**
   * Check whether the rdd is actually persisted. (because after calling `rdd.persist`
   * the rdd is lazy persisted)
   * @param rdd
   */
  def isRDDRealPersisted[T](rdd: RDD[T]): Boolean = {
    if (rdd.partitions.length == 0)
      return true

    // only check partition-0
    val blockId = RDDBlockId(rdd.id, 0)
    SparkEnv.get.blockManager.get(blockId) match {
      case Some(block) => true
      case _ => false
    }
  }

  def isRDDPersisted[T](rdd: RDD[T]): Boolean = {
    rdd.getStorageLevel != StorageLevel.NONE
  }

}
