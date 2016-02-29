package banditsContextual

import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by rcoleman on 2/29/16.
  */
object MLPBanditExample {
  def main(args: Array[String]): Unit = {
    val appName = "BayesianMLPBanitTest"
    val conf = new SparkConf().setAppName(appName).setMaster("local[16]")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val data = sqlContext.read
      .format("libsvm")
      .load("/Users/rcoleman/spark/data/mllib/sample_multiclass_classification_data.txt")

    val params: Map[String, Any] = BayesianContextualBandit.PARAMS

    val splits: Array[DataFrame] = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train: DataFrame = splits(0)
    val test: DataFrame = splits(1)

    // train the MLP bandit model
    val banditModel: BayesianContextualBandit = BayesianContextualBandit.train(train, "mlp", params)
    val recommendations = test.map(BayesianContextualBandit.reportRecommendation(banditModel))
    recommendations.collect().foreach(println)
  }
}
