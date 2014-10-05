# SparkFastALS

SparkFastALS is a Spark package for modeling and fitting matrix factorization.
For more information on FastALS, see [our paper](http://www.stanford.edu/~rezab/papers/fastals.pdf).
This implementation is of Algorithm 4.1 in the paper:

[alt text][fastals.png]

## Compilation

To compile and run, run the following from the Spark root directory. Compilation:
```
sbt/sbt assembly
```
To run with 4GB of ram:
```
./bin/spark-submit --class org.apache.spark.mllib.examples.SparkFastALS  \
  ./examples/target/scala-2.10/spark-examples-1.1.0-SNAPSHOT-hadoop1.0.4.jar \
  --executor-memory 4G \
  --driver-memory 4G
```

# FastALS

For example, the following code fits a model using squared error loss and quadratic
regularization with `rank=5` on the matrix `A`:

    // Iteratively update movies then users
    for (iter <- 1 to ITERATIONS) {
      println("Iteration " + iter + ":")

      // Update ms
      println("Computing new ms")
      ms = multByXstar(R, ms, us, minimizer(us))

      // Update us
      println("Computing new us")
      us = multByXstarTranspose(Rt, ms, us, minimizer(ms))
    }
       

## Missing data

The input is a [RowMatrix](http://spark.apache.org/docs/1.1.0/mllib-data-types.html#rowmatrix), 
which can handle sparse matrices, so the unobserved entries are 
simply not included in the sparse representation.