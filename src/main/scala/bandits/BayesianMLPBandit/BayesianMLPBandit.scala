package bandits.BayesianMLPBandit

import org.apache.spark.ml.MLPpredictor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.mllib.linalg.Vector
import scala.util.Random

object MLPBanditApp {
  def main(args: Array[String]): Unit = {
    val appName = "BayesianMLPBanitTest"
    val conf = new SparkConf().setAppName(appName).setMaster("local[16]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val data = sqlContext.read.format("libsvm").load("/Users/rcoleman/spark/data/mllib/sample_multiclass_classification_data.txt")
    val layers = Array[Int](4, 5, 4, 3)

    val splits: Array[DataFrame] = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train: DataFrame = splits(0)
    val test: DataFrame = splits(1)

    // train the MLP bandit model
    val banditModel: BayesianMLPBandit = BayesianMLPBandit.train(train, layers)
    val recommendations = banditModel.recommendBatch(test)
    recommendations.collect().foreach(println)
  }
}


object BayesianMLPBandit {

  val MAX_ITERATIONS: Int = 100
  val SEED: Long = 1234L
  val BLOCK_SIZE: Int = 128
  def FLATPAYOFF(classes: Int): Array[Double] = Array.fill[Double](classes)(0)



  def train(trainingData: DataFrame, layers: Array[Int]): BayesianMLPBandit = {
    val mlp = this.trainMLP(trainingData,layers)
    new BayesianMLPBandit(mlp)
  }

  def trainValidate(trainingData: DataFrame): BayesianMLPBandit = {
    ???
  }

  def trainMLP(trainingData: DataFrame, layers: Array[Int]): MultilayerPerceptronClassificationModel = {
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
    trainer.fit(trainingData)
  }

}


class BayesianMLPBandit(val trainedMLP: MultilayerPerceptronClassificationModel,
                        payoffPerClass: Option[Array[Double]] = None,
                        val seed: Long = BayesianMLPBandit.SEED) {

  val payoff = payoffPerClass match {
      case None => BayesianMLPBandit.FLATPAYOFF(trainedMLP.layers.head)
      case Some(nonUniformPayoff) => nonUniformPayoff
  }
  val predictor: MLPpredictor = MLPpredictor(trainedMLP,payoff)

  def computePosteriorProb(features: Vector): Array[Double] = this.predictor.predict(features).toArray

  def computePosteriorPayoff(features: Vector, normalized: Boolean = true): Array[Double] = {
    val postPayoff = this.computePosteriorProb(features).zip(payoff).map{ t => t._1 * t._2 }
    if (normalized) {
      val sum: Double = postPayoff.toArray.sum
      payoff.map{ x =>
        if (x == 0.0) {
          x
        } else {
          x / sum
        }
      }
    } else {
      postPayoff
    }
  }

  def samplePull(posteriorPayoff: Array[Double]): Int = {
    val cumSum: Array[(Double, Int)] = posteriorPayoff.scanLeft(0.0)(_ + _).zipWithIndex
    val r: Double = Random.nextDouble()
    cumSum.find{case (cp,idx) => cp > r }.get._2
  }

  def recommend(features: Vector): Int = this.predictor.recommend(features)

  def recommendBatch(data: DataFrame): RDD[MLPBanditRecommendation] = {
    val predictor = this.predictor
    data.map(MLPpredictor.reportRecommendation(predictor))
  }

}


case class MLPBanditRecommendation(label: Double, features: Vector, posteriorPayoff: Array[Double], recommendation: Double)