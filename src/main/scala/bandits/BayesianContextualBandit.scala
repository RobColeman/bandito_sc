package bandits

import classifiers.multiclass.MulticlassClassifier
import org.apache.spark.ml.MLPClassifier
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{DataFrame, Row}
import scala.util.Random



object BayesianContextualBandit {

  val MLP_DEFAULT = true
  val MAX_ITERATIONS: Int = 100
  val SEED: Long = 1234L
  val BLOCK_SIZE: Int = 128
  def FLATPAYOFF(classes: Int): Array[Double] = Array.fill[Double](classes)(1.0 / classes)

  def train(trainingData: DataFrame, layers: Array[Int], payoff: Option[Array[Double]] = None, seed: Long = this.SEED): BayesianContextualBandit = {
    val mlp = this.trainMLP(trainingData,layers)
    val p: Array[Double] = payoff match {
      case None => BayesianContextualBandit.FLATPAYOFF(layers.last)
      case Some(nonUniformPayoff) => nonUniformPayoff
    }
    new BayesianContextualBandit(mlp, p)
  }

  def trainValidate(trainingData: DataFrame): BayesianContextualBandit = {
    ???
  }

  private def trainMLP(trainingData: DataFrame, layers: Array[Int]): MLPClassifier = {
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
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

}


case class MLPBanditRecommendation(label: Double,
                                   recommendation: Int,
                                   features: Seq[Double],
                                   posteriorProb: Seq[Double],
                                   posteriorPayoff: Seq[Double])