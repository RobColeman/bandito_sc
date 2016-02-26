package org.apache.spark.ml

import bandits.BayesianMLPBandit.MLPBanditRecommendation
import org.apache.spark.ml.ann.FeedForwardTopology
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.Row
import scala.util.Random


object MLPpredictor {

  def apply(mlp: MultilayerPerceptronClassificationModel, payoff: Array[Double]): MLPpredictor =
    new MLPpredictor(mlp, payoff: Array[Double])

  def reportRecommendation(mlp: MLPpredictor)(row: Row): MLPBanditRecommendation = {
    val features = row.getAs[Vector](1)
    val posteriorPayoff: Array[Double] = mlp.computePosteriorPayoff(features)
    val recommendation = mlp.recommend(features)
    MLPBanditRecommendation(row.getAs[Double](0), features, posteriorPayoff, recommendation)
  }

}

class MLPpredictor(mlp: MultilayerPerceptronClassificationModel, payoff: Array[Double]) extends Serializable {
  def predict(features: Vector): Vector = {
    val model = FeedForwardTopology.multiLayerPerceptron(this.mlp.layers, true).getInstance(this.mlp.weights)
    model.predict(features)
  }
  def computePosteriorProb(features: Vector): Array[Double] = {
    val response = this.predict(features).toArray
    val min = response.min
    val shifted = if (min < 0) {
      response.map{ r => r - min}
    } else {
      response
    }
    val sum = shifted.sum
    shifted.map{ r =>
      if (sum == 0) {
        0.0
      } else {
        r / sum
      }
    }
  }

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

  def recommend(features: Vector): Int = {
    val posteriorPayoff = this.computePosteriorPayoff(features)
    val cumSum: Array[(Double, Int)] = posteriorPayoff.scanLeft(0.0)(_ + _).zipWithIndex
    val r: Double = Random.nextDouble()
    cumSum.find{case (cp,idx) => cp > r }.get._2
  }

}