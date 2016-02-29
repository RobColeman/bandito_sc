package classifiers.multiclass

import org.apache.spark.mllib.linalg.Vector

/**
  * Created by rcoleman on 2/29/16.
  */
trait MulticlassClassifier extends Serializable {
  def predict(features: Vector): Array[Double]
}
