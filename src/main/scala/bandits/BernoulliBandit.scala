package bandits

import breeze.stats.distributions.Bernoulli

case class BernoulliBandit(p: Double = 0.25) {
  require((p >= 0.0) &&  (p <= 1.0), "Probabilities of success must be between 0.0 and 1.0")

  var N: Int = 0
  var X: Int = 0
  val dist = new Bernoulli(p)

  def pull: Boolean = {
    this.N += 1
    dist.draw() match {
      case true =>
        this.X += 1
        true
      case false => false
    }
  }

}
