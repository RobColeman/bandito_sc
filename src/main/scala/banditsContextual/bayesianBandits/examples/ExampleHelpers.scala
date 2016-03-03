package banditsContextual.examples

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
  * Created by rcoleman on 3/3/16.
  */
object ExampleHelpers {

  def loadData(sqlContext: SQLContext): DataFrame = {
    sqlContext.read
      .format("libsvm")
      .load("/Users/rcoleman/spark/data/mllib/sample_multiclass_classification_data.txt")
  }

  def dfToLabeledPoints(df: DataFrame): RDD[LabeledPoint] = {
    df.map{ row => LabeledPoint(row.getAs[Double](0),row.getAs[Vector](1)) }
  }

  def toNonNegativeValues(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    data.map{ lp => LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map{ _ + 1.0 })) }
  }

}
