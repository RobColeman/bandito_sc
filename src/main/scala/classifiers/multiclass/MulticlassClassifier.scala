package classifiers.multiclass

import org.apache.spark.mllib.linalg.Vector


trait MulticlassClassifier extends Serializable {
  def response(features: Vector): Array[Double]
  def predict(features: Vector): Array[Double] = {
    val response = this.response(features)
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
}
