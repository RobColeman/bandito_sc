package banditsContextual.examples

import banditsContextual.ContextualBanditRecommendation
import banditsContextual.bayesianBandits.BayesianContextualBandit
import org.apache.spark.ml.MLPClassifier
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}


object MLPBanditExample {
  import ExampleHelpers._

  def main(args: Array[String]): Unit = {
    val appName = "BayesianMLPBanitTest"
    val conf = new SparkConf().setAppName(appName).setMaster("local[16]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val data = loadData(sqlContext)

    val params: Map[String, Any] = BayesianContextualBandit.PARAMS

    val splits: Array[DataFrame] = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train: DataFrame = splits(0)
    val test: RDD[LabeledPoint] = dfToLabeledPoints(splits(1))

    // train the MLP bandit model
    val mlpModel = MLPClassifier.train(train, params)
    val banditModel: BayesianContextualBandit = BayesianContextualBandit(mlpModel, params)

    test.map(BayesianContextualBandit.reportRecommendation(banditModel)).collect().foreach(println)
  }
}
