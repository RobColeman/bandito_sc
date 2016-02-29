package org.apache.spark.ml

import bandits.BayesianContextualBandit
import classifiers.multiclass.MulticlassClassifier
import org.apache.spark.ml.ann.FeedForwardTopology
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel
import org.apache.spark.mllib.linalg.Vector


class MLPClassifier(val mlp: MultilayerPerceptronClassificationModel) extends MulticlassClassifier {
  def response(features: Vector): Array[Double] = {
    val model = FeedForwardTopology.multiLayerPerceptron(this.mlp.layers, BayesianContextualBandit.MLP_DEFAULT).getInstance(this.mlp.weights)
    model.predict(features).toArray
  }
}
