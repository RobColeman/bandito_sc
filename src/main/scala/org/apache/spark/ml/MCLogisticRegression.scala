package org.apache.spark.ml

import classifiers.multiclass.MulticlassClassifier
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.regression.LabeledPoint

/**
  * Created by rcoleman on 2/29/16.
  */
object MCLogisticRegression {
  def train(trainingData: DataFrame, params: Map[String,Any]): MCLogisticRegression = {
    val training = trainingData.map{ row => LabeledPoint(row.getAs[Double](0), row.getAs[Vector](1)) }

    val model: LogisticRegressionModel = new LogisticRegressionWithLBFGS()
      .setNumClasses(params("num_classes").asInstanceOf[Int])
      .run(training)

    new MCLogisticRegression(model)
  }
}

class MCLogisticRegression(val model: LogisticRegressionModel) extends MulticlassClassifier {
  def response(features: Vector): Array[Double] = ???
}
