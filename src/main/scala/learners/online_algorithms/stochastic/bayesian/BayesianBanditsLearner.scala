package learners.online_algorithms.stochastic.bayesian

import bandits.BernoulliBandit

object BayesianBanditsLearner {

  val defaultCredibleInterval: Double = 0.95

  def apply(bandits: Vector[BernoulliBandit]): BayesianBanditsLearner = new BayesianBanditsLearner(bandits)

}

class BayesianBanditsLearner(val bandits: Vector[BernoulliBandit],
                             val credibleInterval: Double = BayesianBanditsLearner.defaultCredibleInterval) extends BaseBayesianBanditLearner {

  def getScores: Vector[Double] = banditModels.map{ _.samplePosterior }

}


object BayesianBanditsLearnerApp {
  def main(args: Array[String]): Unit = {

    // val probs = Vector(0.01,0.05,0.1,0.2,0.4,0.5)
    // val probs = Vector(0.001,0.005,0.01,0.05,0.1)
    // val probs = Vector(0.0005,0.001,0.005,0.01,0.05)
    val probs = Vector(0.0001,0.0005,0.001,0.005,0.01)

    val bandits: Vector[BernoulliBandit] = probs.map( BernoulliBandit )

    val learner: BayesianBanditsLearner = BayesianBanditsLearner(bandits)

    learner.play(100)

    learner.play(500 - learner.trials)

    learner.play(1000 - learner.trials)

    learner.play(5000 - learner.trials)

    learner.play(10000 - learner.trials)

    learner.play(50000 - learner.trials)
  }
}

