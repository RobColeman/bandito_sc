package banditsContextual.bayesianBandits

import banditsContextual.ContextualBanditRecommendation
import classifiers.multiclass.MulticlassClassifier
import org.apache.spark.ml.{MCLogisticRegression, MCNaiveBayes}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.util.Random


object BayesianContextualBandit {

  val CLASSIFIER = "mlp"
  val PAYOFF: Option[Array[Double]] = None
  val NUM_CLASSES: Int = 3
  val SEED: Long = 1234L
  def FLATPAYOFF(classes: Int): Array[Double] = Array.fill[Double](classes)(1.0 / classes)

  val PARAMS: Map[String,Any] = Map(
    "classifier" -> this.CLASSIFIER,
    "num_classes" -> this.NUM_CLASSES,
    "payoff" -> this.PAYOFF
  )

  def apply(classifier: MulticlassClassifier,
            params: Map[String,Any] = this.PARAMS): BayesianContextualBandit = {
    new BayesianContextualBandit(classifier, this.getPayoff(params))
  }

  def train(trainingData: RDD[LabeledPoint], params: Map[String,Any] = this.PARAMS): BayesianContextualBandit = {

    val classifier: MulticlassClassifier = params.getOrElse("classifier", this.CLASSIFIER).asInstanceOf[String] match {
      case "logistic" => MCLogisticRegression.train(trainingData, params)
      // case "mlp" => MLPClassifier.train(trainingData, params, sqlContext) // MLLIB has an inconsistent API
      case "naive_bayes" => MCNaiveBayes.train(trainingData, params)
      case _ => MCNaiveBayes.train(trainingData, params)
    }

    new BayesianContextualBandit(classifier, this.getPayoff(params))
  }

  private def getPayoff(params: Map[String,Any]): Array[Double] = {
    params.getOrElse("payoff", this.PAYOFF).asInstanceOf[Option[Array[Double]]] match {
      case None => BayesianContextualBandit.FLATPAYOFF(params("num_classes").asInstanceOf[Int])
      case Some(nonUniformPayoff) => nonUniformPayoff
    }
  }

  def reportRecommendation(banditModel: BayesianContextualBandit)(lp: LabeledPoint): ContextualBanditRecommendation = {
    val label: Double = lp.label
    val features: Vector = lp.features
    val posteriorProb: Array[Double] = banditModel.classifier.predict(features)
    val posteriorPayoff: Array[Double] = banditModel.computeExpectedPayoff(features)
    val recommendation: Int = banditModel.recommend(lp)
    ContextualBanditRecommendation(label, recommendation, features.toArray.toSeq, posteriorProb.toSeq,posteriorPayoff.toSeq)
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
  def recommend(lp: LabeledPoint): Int = this.recommend(lp.features)

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


