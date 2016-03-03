package banditsContextual.examples

import banditsContextual.ContextualBanditRecommendation
import banditsContextual.bayesianBandits.BayesianContextualBandit
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by rcoleman on 2/29/16.
  */
object LogRegBanditExample {
  import ExampleHelpers._
  def main(args: Array[String]): Unit = {
    val appName = "BayesianMLPBanitTest"
    val conf = new SparkConf().setAppName(appName).setMaster("local[16]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val data = dfToLabeledPoints(loadData(sqlContext))

    val params: Map[String, Any] = Map(
      "num_classes" -> 3,
      "classifier" -> "logistic"
    )

    val splits: Array[RDD[LabeledPoint]] = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train: RDD[LabeledPoint] = splits(0)
    val test: RDD[LabeledPoint] = splits(1)

    // train the MLP bandit model
    val banditModel: BayesianContextualBandit = BayesianContextualBandit.train(train, params)

    test.map(BayesianContextualBandit.reportRecommendation(banditModel)).collect().foreach(println)
  }
}
