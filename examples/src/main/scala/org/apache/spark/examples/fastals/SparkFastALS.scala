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

package org.apache.spark.mllib.examples

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, DenseMatrix => BDM, sum, inv}
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scala.util._

/**
 * Fast ALS for Spark
 */
object SparkFastALS {
  // Number of movies
  val M = 100
  // Number of users
  val U = 100
  // Number of features
  val rank = 5
  // Number of iterations
  val ITERATIONS = 100
  // Regularization parameter
  val REG = 0.0001
  // Number of chunks for data (set to number of cores in cluster)
  val NUMCHUNKS = 4

  /**
   * Compute (Proj(X - AB^T) + AB^T) C
   * @param X A distributed Matrix
   * @param A local matrix of factors
   * @param B local matrix of factors
   * @param C local matrix to be multiplied by
   * @return Evaluate the expression (Proj(X - AB^T) + AB^T) C
   */
  def multByXstar(X: IndexedRowMatrix, A: BDM[Double],
                  B: BDM[Double], C: BDM[Double]) : BDM[Double] = {
    val sc = X.rows.context
    val Ab = sc.broadcast[BDM[Double]](A)
    val Bb = sc.broadcast[BDM[Double]](B)

    // Compute Proj(X - AB^T)
    val xminusab_rows = X.rows.map { row =>
      val v = row.vector.toBreeze.asInstanceOf[BDV[Double]].mapActivePairs((j, jVal) =>
        jVal - (Ab.value(row.index.toInt,::) * Bb.value(j,::).t)
      )
      val spRow  = Vectors.fromBreeze(v)
      new IndexedRow(row.index, spRow)
    }
    val xminusab = new IndexedRowMatrix(xminusab_rows, X.numRows().toInt, X.numCols().toInt)

    // Compute Proj(X - AB^T) * C
    val part1 = xminusab.multiply(Matrices.fromBreeze(C)).toBreeze()

    // Compute AB^T * C
    val part2 = A * (B.t * C)

    part1 + part2
  }

  /**
   * Compute (Proj(X - AB^T) + AB^T)^T C
   * @param Xt A distributed Matrix
   * @param A local matrix of factors
   * @param B local matrix of factors
   * @param C local matrix to be multiplied by
   * @return Evaluate the expression (Proj(X - AB^T) + AB^T) C
   */
  def multByXstarTranspose(Xt: IndexedRowMatrix, A: BDM[Double],
                  B: BDM[Double], C: BDM[Double]) : BDM[Double] = {
    val sc = Xt.rows.context
    val Ab = sc.broadcast[BDM[Double]](A)
    val Bb = sc.broadcast[BDM[Double]](B)

    // Compute Proj(X - AB^T)^T
    val xminusab_rows = Xt.rows.map { row =>
      val v = row.vector.toBreeze.asInstanceOf[BDV[Double]].mapActivePairs((j, jVal) =>
        jVal - (Ab.value(j,::) * Bb.value(row.index.toInt,::).t)
      )
      val spRow  = Vectors.fromBreeze(v)
      new IndexedRow(row.index, spRow)
    }
    val xminusabT = new IndexedRowMatrix(xminusab_rows, Xt.numRows().toInt, Xt.numCols().toInt)

    // Compute Proj(X - AB^T) * C
    val part1 = xminusabT.multiply(Matrices.fromBreeze(C)).toBreeze()

    // Compute BA^T * C
    val part2 = B * (A.t * C)

    part1 + part2
  }

  /**
   * Helper function to compute B * (B^T B + REG*I)^-1
   * @param B local matrix B
   * @return B * (B^T B + REG*I)^-1
   */
  def minimizer(B: BDM[Double]) : BDM[Double] = {
    val mid = (B.t * B)

    for(i <- 0 until rank)
      mid(i, i) += REG

    B * inv(mid)
  }

  /**
   * Compute loss for current model
   * @param A Dense matrix of factors
   * @param B Dense matrix of factors
   * @param X Distributed data matrix
   * @return RMSE for the given model factors
   */
  def computeLoss(A: BDM[Double], B: BDM[Double], X:IndexedRowMatrix) : Double = {
    val sc = X.rows.context
    val Ab = sc.broadcast[BDM[Double]](A)
    val Bb = sc.broadcast[BDM[Double]](B)

    math.sqrt(X.rows.map { row =>
      sum(row.vector.toBreeze.asInstanceOf[BDV[Double]].mapActivePairs((j, jVal) =>
        math.pow(jVal - (Ab.value(row.index.toInt,::) * Bb.value(j,::).t), 2)))
    }.mean())
  }

  def main(args: Array[String]) {
    printf("Running with M=%d, U=%d, rank=%d, iters=%d, reg=%f\n", M, U, rank, ITERATIONS, REG)

    val sparkConf = new SparkConf().setAppName("SparkFastALS")
    val sc = new SparkContext(sparkConf)

    // Create data
    val entries = sc.parallelize(0 until M, NUMCHUNKS).map { i =>
      IndexedRow(i, Vectors.dense((0 until U).map(j => math.sin(i*j+i+j)).toArray))
    }.cache()

    val entries_t = sc.parallelize(0 until U, NUMCHUNKS).map { i =>
      IndexedRow(i, Vectors.dense((0 until M).map(j => math.sin(i*j+i+j)).toArray))
    }.cache()

    val R = new IndexedRowMatrix(entries, M, U)
    val Rt = new IndexedRowMatrix(entries_t, U, M)

    // Initialize m and u
    var ms = BDM.ones[Double](M, rank) / (M.toDouble * M)
    var us = BDM.ones[Double](U, rank) / (U.toDouble * U)

    val errs = Array.ofDim[Double](ITERATIONS)

    // Iteratively update movies then users
    for (iter <- 1 to ITERATIONS) {
      println("Iteration " + iter + ":")

      // Update ms
      println("Computing new ms")
      ms = multByXstar(R, ms, us, minimizer(us))

      // Update us
      println("Computing new us")
      us = multByXstarTranspose(Rt, ms, us, minimizer(ms))

      // Comment this out for large runs to avoid an extra pass
      errs(iter - 1) = computeLoss(ms, us, R)
    }

    println("RMSEs: " + errs.mkString(", "))

    sc.stop()
  }
}
