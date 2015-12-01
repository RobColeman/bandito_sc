package learners.online_algorithms.stochastic.bayesian

import banditModels.BetaBinomialBanditModel
import bandits.BernoulliBandit

abstract class BaseBayesianBanditLearner {

  val bandits: Vector[BernoulliBandit]
  val credibleInterval: Double

  val nBandits: Int = bandits.length

  val banditModels: Vector[BetaBinomialBanditModel] = bandits.zipWithIndex.map{ t =>
    BetaBinomialBanditModel(t._2, t._1, credibleInterval)
  }

  def trials: Int = this.banditModels.map{ _.trials }.sum
  def successes: Int = this.banditModels.map{ _.successes }.sum

  def getScores: Vector[Double]

  private def selectNextBanditToPlay: BetaBinomialBanditModel = {
    val banditToPlay: Int = util.selectHighestScore(this.getScores)
    this.banditModels(banditToPlay)
  }

  def play: Unit = selectNextBanditToPlay.pull
  def play(n: Int): Unit = {
    var i = 0
    for (i <- 0 until n) {
      this.play
    }
    println(s"")
    this.banditModels.foreach{ _.print }
  }

  // TODO: better toString
  // TODO: compute regret

}