package org.apache.spark.ml

import classifiers.multiclass.MulticlassClassifier
import org.apache.spark.mllib.classification.{NaiveBayesModel, NaiveBayes}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


object MCNaiveBayes {
  def train(trainingData: RDD[LabeledPoint], params: Map[String,Any]): MCNaiveBayes = {

    val model: NaiveBayesModel = new NaiveBayes()
      .setLambda(params("lambda").asInstanceOf[Double])
      .setModelType("multinomial")
      .run(trainingData)

    new MCNaiveBayes(model)
  }

}

class MCNaiveBayes(val model: NaiveBayesModel) extends MulticlassClassifier {
  def response(features: Vector): Array[Double] = model.predictProbabilities(features).toArray
}
