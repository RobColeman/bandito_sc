package banditModels

import bandits.BernoulliBandit
import breeze.stats.distributions.Binomial
import org.apache.commons.math3.distribution.BetaDistribution

abstract class BaseBetaBinomialBanditModel {
  val id: Int
  val name: String = s"Bandit $id"
  val bandit: BernoulliBandit
  val credibleInterval: Double = 0.95

  def trials: Int
  def successes: Int

  def beta_a_scale: Int = 1 + this.successes
  def beta_b_scale: Int = 1 + this.trials - this.successes
  def betaPrior: BetaDistribution = new BetaDistribution(this.beta_a_scale,this.beta_b_scale)

  def binomialLikelihood: Binomial = new Binomial(this.trials, this.successes.toDouble / this.trials.toDouble)

  def samplePosterior: Double = this.betaPrior.sample()

  private def computePosteriorHPDBounds: (Double, Double) = {
    // TODO: fix this, numerically compute HPD, not central region
    val posterior: BetaDistribution = this.betaPrior
    val alpha = (1.0 - credibleInterval) / 2.0
    val lower: Double = posterior.inverseCumulativeProbability( alpha )
    val upper: Double = posterior.inverseCumulativeProbability( 1 - alpha )
    (lower,upper)
  }
  def lowerConfidenceBound: Double = this.computePosteriorHPDBounds._1
  def upperConfidenceBound: Double = this.computePosteriorHPDBounds._2

  def pull: Boolean
  def paramString: String = {
    this.name + s" trials=${this.trials}, params: a_scale=${this.beta_a_scale}, b_scale=${this.beta_b_scale} ."
  }
  def print: Unit = println(this.paramString)
}

object BetaBinomialBanditModel {
  def apply(id: Int, bandit: BernoulliBandit, credibleInterval: Double = 0.95): BetaBinomialBanditModel =
    new BetaBinomialBanditModel(id = id, bandit = bandit, credibleInterval = credibleInterval)
}
class BetaBinomialBanditModel(val id: Int, val bandit: BernoulliBandit, override val credibleInterval: Double = 0.95)
  extends BaseBetaBinomialBanditModel {

  private var nTrials: Int = 0
  def trials: Int = this.nTrials
  private var nSuccesses: Int = 0
  def successes: Int = this.nSuccesses

  def pull: Boolean = {
    this.nTrials += 1
    this.bandit.pull match {
      case true =>
        this.nSuccesses += 1
        true
      case false =>
        false
    }
  }

}

object BetaBinomialBanditModelLimitedHistory {
  def apply(id: Int,
            bandit: BernoulliBandit,
            credibleInterval: Double = 0.95,
            maxMemory: Int = 10000): BetaBinomialBanditModelLimitedHistory =
        new BetaBinomialBanditModelLimitedHistory(id = id, bandit = bandit,
          credibleInterval = credibleInterval, maxMemory = maxMemory)
}

class BetaBinomialBanditModelLimitedHistory(val id: Int, val bandit: BernoulliBandit, override val credibleInterval: Double = 0.95,
                                            maxMemory: Int = 10000) extends BaseBetaBinomialBanditModel {

  val trialHistory: BernoulliTrialsLimitedHistory = BernoulliTrialsLimitedHistory(maxMemory = maxMemory)

  def totalTrials: Long = this.trialHistory.totalTrials
  def totalSuccesses: Long = this.trialHistory.totalSuccesses

  def trials: Int = this.trialHistory.trials
  def successes: Int = this.trialHistory.successes

  def computePosteriorHPDBounds: (Double, Double) = {
    val posterior: BetaDistribution = this.betaPrior
    val alpha = (1.0 - credibleInterval) / 2.0
    val lower: Double = posterior.inverseCumulativeProbability( alpha )
    val upper: Double = posterior.inverseCumulativeProbability( 1 - alpha )
    (lower,upper)
  }

  def computeLCB: Double = this.computePosteriorHPDBounds._1
  def computeUCB: Double = this.computePosteriorHPDBounds._2

  def pull: Boolean = {
    val trialResult: Boolean = this.bandit.pull
    this.trialHistory.update(trialResult)
    trialResult
  }

}