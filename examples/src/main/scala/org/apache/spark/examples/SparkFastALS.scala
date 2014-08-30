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

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM}
import breeze.linalg._
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Generalized Low Rank Models for Spark
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


  def minimizer(B: BDM[Double]) : BDM[Double] = {
    val btb =  B * inv(B.t * B + BDM.eye[Double](rank) * REG)
  }

  def multByXstar(X: RowMatrix, A: BDM[Double],
                  B: BDM[Double], C: BDM[Double]) : BDM[Double] = {
    val xminusab = X.rows.map(row => )


  }

  def main(args: Array[String]) {
    val options = (0 to 5).map(i => if (i < args.length) Some(args(i)) else None)

    options.toArray match {
      case Array(m, u, nn, trank, iters, reg) =>
        M = m.getOrElse("10").toInt
        U = u.getOrElse("5").toInt
        NNZ = nn.getOrElse("23").toInt
        rank = trank.getOrElse("2").toInt
        ITERATIONS = iters.getOrElse("5").toInt
        REG = reg.getOrElse("100").toInt

      case _ =>
        System.err.println("Usage: SparkFastALS [M] [U] [nnz] [rank] [iters] [regularization]")
        System.exit(1)
    }

    printf("Running with M=%d, U=%d, nnz=%d, rank=%d, iters=%d, reg=%d\n", M, U, NNZ, rank, ITERATIONS, REG)

    val sparkConf = new SparkConf().setAppName("SparkFastALS")
    val sc = new SparkContext(sparkConf)

    // Create data
    val R = sc.parallelize(1 to NNZ).map{x =>
      val i = math.round(math.random * (M - 1)).toInt
      val j = math.round(math.random * (U - 1)).toInt
      ((i, j), math.random)
    }.reduceByKey(_ + _).map{case (a, b) => (a._1, a._2, b)}

    // Transpose data
    val RT = R.map { case (i, j, rij) => (j, i, rij) }

    // Initialize m and u
    var ms = BDM.ones[Double](M, rank) / (M.toDouble * M)
    var us = BDM.ones[Double](U, rank) / (U.toDouble * U)

    // Iteratively update movies then users
    var msb = sc.broadcast(ms)
    var usb = sc.broadcast(us)

    for (iter <- 1 to ITERATIONS) {
      println("Iteration " + iter + ":")

      // Update ms
      println("Computing gradient losses")
      var lg = computeLossGrads(msb, usb, R)
      println("Updating M factors")
      ms = update(usb, msb, lg, 1.0/iter)
      msb = sc.broadcast(ms) // Re-broadcast ms because it was updated

      // Update us
      println("Computing gradient losses")
      lg = computeLossGrads(usb, msb, RT)
      println("Updating U factors")
      us = update(msb, usb, lg, 1.0/iter)
      usb = sc.broadcast(us) // Re-broadcast us because it was updated

      println("error = " + computeLoss(msb, usb, R).map { case (_, _, lij) => lij}.mean())
      //println(us.mkString(", "))
      //println(ms.mkString(", "))
      println()
    }

    sc.stop()
  }
}
