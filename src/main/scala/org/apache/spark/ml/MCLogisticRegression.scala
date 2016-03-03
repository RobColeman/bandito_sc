package org.apache.spark.ml

import classifiers.multiclass.MulticlassClassifier
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint


object MCLogisticRegression {
  def train(trainingData: RDD[LabeledPoint], params: Map[String,Any]): MCLogisticRegression = {

    val model: LogisticRegressionModel = new LogisticRegressionWithLBFGS()
      .setNumClasses(params("num_classes").asInstanceOf[Int])
      .run(trainingData)

    new MCLogisticRegression(model)
  }
}

class MCLogisticRegression(val model: LogisticRegressionModel) extends MulticlassClassifier {
  def response(features: Vector): Array[Double] = ???
  //TODO, must get multi-class response from LogisticRegressionModel
}
