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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, DenseMatrix => BDM}
import breeze.linalg.inv
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Fast ALS for Spark
 */
object SparkFastALS {
  // Number of movies
  var M = 0
  // Number of users
  var U = 0
  // Number of nonzeros
  var NNZ = 0
  // Number of features
  var rank = 0
  // Number of iterations
  var ITERATIONS = 0
  // Regularization parameter
  var REG = 0

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
      val v = row.vector.toBreeze.asInstanceOf[BSV[Double]].mapActivePairs((j, jVal) =>
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
      val v = row.vector.toBreeze.asInstanceOf[BSV[Double]].mapActivePairs((j, jVal) =>
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
      row.vector.toBreeze.asInstanceOf[BSV[Double]].mapActivePairs((j, jVal) =>
        math.pow(jVal - (Ab.value(row.index.toInt,::) * Bb.value(j,::).t), 2)).sum
    }.mean())
  }

  def main(args: Array[String]) {
    val options = (0 to 5).map(i => if (i < args.length) Some(args(i)) else None)

    options.toArray match {
      case Array(m, u, nn, trank, iters, reg) =>
        M = m.getOrElse("10").toInt
        U = u.getOrElse("5").toInt
        NNZ = nn.getOrElse("23").toInt
        rank = trank.getOrElse("2").toInt
        ITERATIONS = iters.getOrElse("20").toInt
        REG = reg.getOrElse("1").toInt

      case _ =>
        System.err.println("Usage: SparkFastALS [M] [U] [nnz] [rank] [iters] [regularization]")
        System.exit(1)
    }

    printf("Running with M=%d, U=%d, nnz=%d, rank=%d, iters=%d, reg=%d\n", M, U, NNZ, rank, ITERATIONS, REG)

    val sparkConf = new SparkConf().setAppName("SparkFastALS")
    val sc = new SparkContext(sparkConf)

    // Create data
    val entries = sc.parallelize(1 to NNZ).map{x =>
      val i = math.round(math.random * (M - 1)).toInt
      val j = math.round(math.random * (U - 1)).toInt
      ((math.abs(i), math.abs(j)), math.random)
    }.reduceByKey(_ + _).map{case (a, b) => MatrixEntry(a._1, a._2, b)}

    val R = new CoordinateMatrix(entries, M, U).toIndexedRowMatrix()
    val Rt = new CoordinateMatrix(entries.map(a => MatrixEntry(a.j, a.i, a.value)), U, M).toIndexedRowMatrix()

    // Initialize m and u
    var ms = BDM.ones[Double](M, rank) / (M.toDouble * M)
    var us = BDM.ones[Double](U, rank) / (U.toDouble * U)

    // Iteratively update movies then users
    for (iter <- 1 to ITERATIONS) {
      println("Iteration " + iter + ":")

      // Update ms
      println("Computing new ms")
      ms = multByXstar(R, ms, us, minimizer(us))

      // Update us
      println("Computing new us")
      us = multByXstarTranspose(Rt, ms, us, minimizer(ms))

      println("error = " + computeLoss(ms, us, R))
      println()
    }

    sc.stop()
  }
}
