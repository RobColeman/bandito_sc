package learners.online_algorithms.stochastic.bayesian

import banditModels.BetaBinomialBanditModel
import bandits.BernoulliBandit

object BayesianUCBLearner {

  val defaultCredibleInterval: Double = 0.95

  def apply(bandits: Vector[BernoulliBandit], credibleInterval: Double = defaultCredibleInterval): BayesianUCBLearner = {
    new BayesianUCBLearner(bandits, credibleInterval)
  }

}

class BayesianUCBLearner(val bandits: Vector[BernoulliBandit],
                         val credibleInterval: Double = BayesianBanditsLearner.defaultCredibleInterval) extends BaseBayesianBandit {

  def getScores: Vector[Double] = banditModels.map{ _.upperConfidenceBound }

}


object BayesianUCBLearnerApp {
  def main(args: Array[String]): Unit = {

    // val probs = Vector(0.01,0.05,0.1,0.2,0.4,0.5)
    // val probs = Vector(0.001,0.005,0.01,0.05,0.1)
    //val probs = Vector(0.0005,0.001,0.005,0.01,0.05)
    val probs = Vector(0.0001,0.0005,0.001,0.005,0.01)

    val bandits: Vector[BernoulliBandit] = probs.map( BernoulliBandit )

    val learner: BayesianUCBLearner = BayesianUCBLearner(bandits)

    learner.play(100)

    learner.play(500 - learner.trials)

    learner.play(1000 - learner.trials)

    learner.play(5000 - learner.trials)

    learner.play(10000 - learner.trials)

    learner.play(20000 - learner.trials)
  }
}
