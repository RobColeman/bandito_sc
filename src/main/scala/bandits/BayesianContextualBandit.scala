package bandits

import classifiers.multiclass.MulticlassClassifier
import org.apache.spark.ml.MLPClassifier
import org.apache.spark.sql.Row
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.mllib.linalg.Vector
import scala.util.Random


object MLPBanditApp {
  def main(args: Array[String]): Unit = {
    val appName = "BayesianMLPBanitTest"
    val conf = new SparkConf().setAppName(appName).setMaster("local[16]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val data = sqlContext.read
      .format("libsvm")
      .load("/Users/rcoleman/spark/data/mllib/sample_multiclass_classification_data.txt")

    val params: Map[String, Any] = BayesianContextualBandit.PARAMS

    val splits: Array[DataFrame] = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train: DataFrame = splits(0)
    val test: DataFrame = splits(1)

    // train the MLP bandit model
    val banditModel: BayesianContextualBandit = BayesianContextualBandit.train(train, "mlp", params)
    val recommendations = test.map(BayesianContextualBandit.reportRecommendation(banditModel))
    recommendations.collect().foreach(println)
  }
}

object BayesianContextualBandit {

  val CLASSIFIER = "mlp"
  val PARAMS: Map[String,Any] = Map("layers" -> Array[Int](4, 5, 4, 3),
                                    "num_classes" -> 3)
  val MLP_DEFAULT = true
  val MAX_ITERATIONS: Int = 100
  val SEED: Long = 1234L
  val BLOCK_SIZE: Int = 128
  def FLATPAYOFF(classes: Int): Array[Double] = Array.fill[Double](classes)(1.0 / classes)

  def train(trainingData: DataFrame,
            classifierType: String = this.CLASSIFIER,
            params: Map[String,Any] = this.PARAMS,
            payoff: Option[Array[Double]] = None,
            seed: Long = this.SEED): BayesianContextualBandit = {

    val classifier = classifierType match {
      case "mlp" => this.trainMLP(trainingData, params)
      case _ => this.trainMLP(trainingData, params)
    }

    val p: Array[Double] = payoff match {
      case None => BayesianContextualBandit.FLATPAYOFF(params("num_classes").asInstanceOf[Int])
      case Some(nonUniformPayoff) => nonUniformPayoff
    }

    new BayesianContextualBandit(classifier, p)
  }

  def trainValidate(trainingData: DataFrame): BayesianContextualBandit = {
    ???
  }

  private def trainMLP(trainingData: DataFrame, params: Map[String,Any]): MLPClassifier = {
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(params("layers").asInstanceOf[Array[Int]])
      .setBlockSize(params.getOrElse("block_size", this.BLOCK_SIZE).asInstanceOf[Int])
      .setSeed(params.getOrElse("seed",this.SEED).asInstanceOf[Long])
      .setMaxIter(params.getOrElse("max_iterations",this.MAX_ITERATIONS).asInstanceOf[Int])
    new MLPClassifier(trainer.fit(trainingData))
  }

  def reportRecommendation(banditModel: BayesianContextualBandit)(row: Row): MLPBanditRecommendation = {
    val label: Double = row.getAs[Double](0)
    val features: Vector = row.getAs[Vector](1)
    val posteriorProb: Array[Double] = banditModel.classifier.predict(features)
    val posteriorPayoff: Array[Double] = banditModel.computeExpectedPayoff(features)
    val recommendation: Int = banditModel.recommend(row)
    MLPBanditRecommendation(label, recommendation, features.toArray.toSeq, posteriorProb.toSeq,posteriorPayoff.toSeq)
  }

}

class BayesianContextualBandit(val classifier: MulticlassClassifier, val payoff: Array[Double]) extends Serializable {

  def computeExpectedPayoff(features: Vector, normalized: Boolean = true): Array[Double] = {
    val unNomredPostPayoff: Array[Double] = this.classifier.predict(features).zip(payoff).map{ t => t._1 * t._2 }
    if (normalized) {
      val sum: Double = unNomredPostPayoff.toArray.sum
      unNomredPostPayoff.map{ x => if (x == 0.0) x else x / sum }
    } else {
      unNomredPostPayoff
    }
  }

  def recommend(features: Vector): Int = {
    val cumSum: Array[(Double, Int)] = this.computeExpectedPayoff(features).scanLeft(0.0)(_ + _).tail.zipWithIndex
    val r: Double = Random.nextDouble()
    cumSum.find{ case (cp,idx) => cp > r } match {
      case Some(cumPrIdx) => cumPrIdx._2
      case None => -1
    }
  }

  def recommend(row: Row): Int = this.recommend(row.getAs[Vector](1))

  /**
    *
    * @param features
    * @param exclude indicies to exclude from recommendations (adds recently seen)
    * @return
    */
  def recommend(features: Vector, exclude: Array[Int]): Int = {
    val (expPayoff, idxMap): (Seq[Double], Seq[(Int, Int)]) = this.computeExpectedPayoff(features)
      .zipWithIndex
      .filterNot{ case (p, idx) => exclude.contains(idx) }
      .zipWithIndex
      .map{ case ((p,origIdx),newIdx) =>  (p,(newIdx,origIdx)) }.unzip

    val newToOldIdx: Map[Int, Int] = idxMap.toMap

    val cumSum: Seq[(Double, Int)] = expPayoff.scanLeft(0.0)(_ + _).tail.zipWithIndex
    val r: Double = Random.nextDouble()

    val recIdx = cumSum.find{ case (cp,idx) => cp > r } match {
      case Some(cumPrIdx) => cumPrIdx._2
      case None => -1
    }
    newToOldIdx(recIdx)
  }

  def recommend(row: Row, exclude: Array[Int]): Int = this.recommend(row.getAs[Vector](1), exclude)

}


case class MLPBanditRecommendation(label: Double,
                                   recommendation: Int,
                                   features: Seq[Double],
                                   posteriorProb: Seq[Double],
                                   posteriorPayoff: Seq[Double])