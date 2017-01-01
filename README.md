# spark-vlbfgs
This package is an implementation of the [Vector-free L-BFGS](https://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce.pdf)
solver and some scalable machine learning algorithms for Apache Spark.

Apache Spark MLlib provides scalable implementation of popular machine learning algorithms,
which lets users train models from big dataset and iterate fast.
The existing implementations assume that the number of parameters is small enough to fit in the memory of a single machine.
However, many applications require solving problems with billions of parameters on a huge amount of data
such as Ads CTR prediction and deep neural network.
This requirement far exceeds the capacity of exisiting MLlib algorithms many of which use L-BFGS as the underlying solver.
In order to fill this gap, we developed Vector-free L-BFGS for MLlib.
Vector-free L-BFGS avoids the expensive dot product operations in the two loop recursion
and greatly improves computation efficiency with a great degree of parallelism.
It can solve optimization problems with billions of parameters in the Spark SQL framework where the training data are often generated.
The algorithm scales very well and enables a variety of MLlib algorithms to handle a massive number of parameters over large datasets.

## Supported algorithms

spark-vlbfgs currently supports the following algorithms:

* Logistic Regression

with regularization:

* L1
* L2
* Elastic Net

To be supported:

* Linear Regression
* Softmax Regression
* Multilayer Perceptron Classifier

## Build and run spark-vlbfgs

spark-vlbfgs is built using [Apache Maven](http://maven.apache.org/).
To build spark-vlbfgs and its example programs, run:
    
    mvn clean package -DskipTests

by default this project will be built against spark-2.0.0 with scala-2.11,
if you want to specify other version, use maven `-D` parameter such as:

    mvn clean package -Dscala.binary.version=2.10 -Dspark.version=2.0.0

then run example:

    spark-submit
       --master yarn
       --num-executors 10
       --executor-cores 2
       --class org.apache.spark.ml.example.VLORExample
       /path/to/spark-vlbfgs-0.1-SNAPSHOT.jar [paramlist]

## Example

You can train a logistic regression model via spark-vlbfgs API which is consistent with Apache Spark MLlib:

    val dataset: Dataset[_] = spark.read.format("libsvm").load("data/a9a")
    val trainer = new VLogisticRegression()
      .setColsPerBlock(100)
      .setRowsPerBlock(10)
      .setColPartitions(3)
      .setRowPartitions(3)
      .setRegParam(0.5)
    val model = trainer.fit(dataset)

    println(s"Vector-free logistic regression coefficients: ${model.coefficients}")

## Reference

* [Large-scale L-BFGS using MapReduce](https://papers.nips.cc/paper/5333-large-scale-l-bfgs-using-mapreduce.pdf)
* https://github.com/mengxr/spark-vl-bfgs

## Contact & Acknowledgements

If you have any questions or encounter bugs, feel free to submit an issue or contact:

* [Yanbo Liang](https://github.com/yanboliang) (ybliang8@gmail.com)
* [Weichen Xu](https://github.com/WeichenXu123) (WeichenXu123@outlook.com)

We are immensely grateful to [Xiangrui Meng](https://github.com/mengxr) for the initial work and guidance during the design and development of spark-vlbfgs.
