package banditsContextual.examples

import banditsContextual.bayesianBandits.BayesianContextualBandit
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}


/**
  * Created by rcoleman on 3/3/16.
  */
object NaiveBayesBanditExample {
  import ExampleHelpers._

  def main(args: Array[String]): Unit = {

    val appName = "BayesianMLPBanitTest"
    val conf = new SparkConf().setAppName(appName).setMaster("local[16]")
    val sc = new SparkContext(conf)
    val sqlContext: SQLContext = new SQLContext(sc)

    val data = toNonNegativeValues(dfToLabeledPoints(loadData(sqlContext)))

    val params: Map[String, Any] = Map(
      "classifier" -> "naive_bayes",
      "lambda" -> 0.5,
      "num_classes" -> 3
    )

    val splits: Array[RDD[LabeledPoint]] = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train: RDD[LabeledPoint] = splits(0).cache()
    val test: RDD[LabeledPoint] = splits(1).cache()

    // train the MLP bandit model
    val banditModel: BayesianContextualBandit = BayesianContextualBandit.train(train, params)

    test.map(BayesianContextualBandit.reportRecommendation(banditModel)).collect().foreach(println)
  }

}
