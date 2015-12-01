package metrics

import bandits.BernoulliBandit

trait BernoulliRegret {

  val bandits: Vector[BernoulliBandit]

  private val bestProb: Double = this.bandits.map{ _.p }.max

  def computeRegret: Double = this.bandits.map{ b =>  (bestProb * b.N) - (b.p * b.N) }.sum

}
