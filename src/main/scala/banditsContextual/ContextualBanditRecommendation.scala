package banditsContextual

/**
  * Created by rcoleman on 2/29/16.
  */
case class ContextualBanditRecommendation(label: Double,
                                          recommendation: Int,
                                          features: Seq[Double],
                                          posteriorProb: Seq[Double],
                                          posteriorPayoff: Seq[Double])
