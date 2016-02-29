package org.apache.spark.ml

import banditsContextual.BayesianContextualBandit
import classifiers.multiclass.MulticlassClassifier
import org.apache.spark.ml.ann.FeedForwardTopology
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame

object MLPClassifier {

  val CLASSIFIER = "mlp"
  val PARAMS: Map[String,Any] = Map("layers" -> Array[Int](4, 5, 4, 3),
    "num_classes" -> 3)
  val MLP_DEFAULT = true
  val MAX_ITERATIONS: Int = 100
  val SEED: Long = 1234L
  val BLOCK_SIZE: Int = 128

  def train(trainingData: DataFrame, params: Map[String,Any]): MLPClassifier = {
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(params("layers").asInstanceOf[Array[Int]])
      .setBlockSize(params.getOrElse("block_size", this.BLOCK_SIZE).asInstanceOf[Int])
      .setSeed(params.getOrElse("seed",this.SEED).asInstanceOf[Long])
      .setMaxIter(params.getOrElse("max_iterations",this.MAX_ITERATIONS).asInstanceOf[Int])
    new MLPClassifier(trainer.fit(trainingData))
  }

}

class MLPClassifier(val mlp: MultilayerPerceptronClassificationModel) extends MulticlassClassifier {
  def response(features: Vector): Array[Double] = {
    val model = FeedForwardTopology.multiLayerPerceptron(this.mlp.layers, BayesianContextualBandit.MLP_DEFAULT).getInstance(this.mlp.weights)
    model.predict(features).toArray
  }
}
