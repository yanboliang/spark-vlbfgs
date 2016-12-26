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

import org.apache.spark.{SharedSparkContext, SparkFunSuite}

class RDDUtilsSuite extends SparkFunSuite with SharedSparkContext {

  override def beforeAll(): Unit = {
    super.beforeAll()
  }

  /**
   * The `isRDDRealPersisted` implementation is not used currently
   * and needs to be updated.
   */
  ignore("isRDDRealPersisted & isRDDPersisted") {

    val rdd = sc.parallelize(Seq(1, 2, 3), 3).map(_ + 10)

    assert(!RDDUtils.isRDDPersisted(rdd))
    assert(!RDDUtils.isRDDRealPersisted(rdd))

    rdd.persist()

    assert(RDDUtils.isRDDPersisted(rdd))
    assert(!RDDUtils.isRDDRealPersisted(rdd))

    rdd.count() // eager persist

    assert(RDDUtils.isRDDPersisted(rdd))
    assert(RDDUtils.isRDDRealPersisted(rdd))
  }

}
