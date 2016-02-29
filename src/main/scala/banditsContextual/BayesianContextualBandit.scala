package banditsContextual

import classifiers.multiclass.MulticlassClassifier
import org.apache.spark.ml.{MCLogisticRegression, MLPClassifier}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{DataFrame, Row}
import scala.util.Random




object BayesianContextualBandit {

  val CLASSIFIER = "mlp"
  val PARAMS: Map[String,Any] = Map("layers" -> Array[Int](4, 5, 4, 3),
                                    "num_classes" -> 3)
  val MLP_DEFAULT = true
  val MAX_ITERATIONS: Int = 100
  val SEED: Long = 1234L
  val BLOCK_SIZE: Int = 128
  def FLATPAYOFF(classes: Int): Array[Double] = Array.fill[Double](classes)(1.0 / classes)

  def apply(classifier: MulticlassClassifier,
            params: Map[String,Any] = this.PARAMS,
            payoff: Option[Array[Double]] = None): BayesianContextualBandit = {

    val p: Array[Double] = payoff match {
      case None => BayesianContextualBandit.FLATPAYOFF(params("num_classes").asInstanceOf[Int])
      case Some(nonUniformPayoff) => nonUniformPayoff
    }
    new BayesianContextualBandit(classifier, p)
  }

  def train(trainingData: DataFrame,
            classifierType: String = this.CLASSIFIER,
            params: Map[String,Any] = this.PARAMS,
            payoff: Option[Array[Double]] = None,
            seed: Long = this.SEED): BayesianContextualBandit = {

    val classifier: MulticlassClassifier = classifierType match {
      case "logistic" => MCLogisticRegression.train(trainingData, params)
      case "mlp" => MLPClassifier.train(trainingData, params)
      case _ => MLPClassifier.train(trainingData, params)
    }

    val p: Array[Double] = payoff match {
      case None => BayesianContextualBandit.FLATPAYOFF(params("num_classes").asInstanceOf[Int])
      case Some(nonUniformPayoff) => nonUniformPayoff
    }

    new BayesianContextualBandit(classifier, p)
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