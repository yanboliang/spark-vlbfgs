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

package org.apache.spark.ml

import org.apache.spark.ml.param.{BooleanParam, IntParam, ParamValidators, Params}

private trait VParams extends Params{
  // column number of each block in feature block matrix
  val colsPerBlock: IntParam = new IntParam(this, "colsPerBlock",
    "column number of each block in feature block matrix.", ParamValidators.gt(0))
  setDefault(colsPerBlock -> 10000)

  def getColsPerBlock: Int = $(colsPerBlock)

  // row number of each block in feature block matrix
  val rowsPerBlock: IntParam = new IntParam(this, "rowsPerBlock",
    "row number of each block in feature block matrix.", ParamValidators.gt(0))
  setDefault(rowsPerBlock -> 10000)

  def getRowsPerBlock: Int = $(rowsPerBlock)

  // row partition number of feature block matrix
  // equals to partition number of coefficient vector
  val rowPartitions: IntParam = new IntParam(this, "rowPartitions",
    "row partition number of feature block matrix.", ParamValidators.gt(0))
  setDefault(rowPartitions -> 10)

  def getRowPartitions: Int = $(rowPartitions)

  // column partition number of feature block matrix
  val colPartitions: IntParam = new IntParam(this, "colPartitions",
    "column partition number of feature block matrix.", ParamValidators.gt(0))
  setDefault(colPartitions -> 10)

  def getColPartitions: Int = $(colPartitions)

  // Whether to eager persist distributed vector.
  val eagerPersist: BooleanParam = new BooleanParam(this, "eagerPersist",
    "Whether to eager persist distributed vector.")
  setDefault(eagerPersist -> false)

  def getEagerPersist: Boolean = $(eagerPersist)

  // The number of corrections used in the LBFGS update.
  val numCorrections: IntParam = new IntParam(this, "numCorrections",
    "The number of corrections used in the LBFGS update.")
  setDefault(numCorrections -> 10)

  def getNumCorrections: Int = $(numCorrections)

  val generatingFeatureMatrixBuffer: IntParam = new IntParam(this, "generatingFeatureMatrixBuffer",
    "Buffer size when generating features block matrix.")
  setDefault(generatingFeatureMatrixBuffer -> 1000)

  def getGeneratingFeatureMatrixBuffer: Int = $(generatingFeatureMatrixBuffer)

  val rowPartitionSplitNumOnGeneratingFeatureMatrix: IntParam = new IntParam(this,
    "rowPartitionSplitsNumOnGeneratingFeatureMatrix",
    "row partition splits number on generating features matrix."
  )
  setDefault(rowPartitionSplitNumOnGeneratingFeatureMatrix -> 1)

  def getRowPartitionSplitNumOnGeneratingFeatureMatrix: Int =
    $(rowPartitionSplitNumOnGeneratingFeatureMatrix)

  val compressFeatureMatrix: BooleanParam = new BooleanParam(this,
    "compressFeatureMatrix",
    "compress feature matrix."
  )
  setDefault(compressFeatureMatrix -> false)

  def getCompressFeatureMatrix: Boolean = $(compressFeatureMatrix)
}
