package math

import scala.annotation.tailrec
import breeze.stats.distributions.Beta
import org.apache.commons.math3.distribution.BetaDistribution

object HighestPosteriorDensity {

  def linearSymmetric(dist: BetaDistribution, interval: Double = 0.95, stepSize: Double = 0.00001): (Double,Double) = {
    val brzDist: Beta = new Beta(dist.getAlpha, dist.getBeta)
    val mode: Double = brzDist.mode
    linearSymmetricSearch(mode - stepSize, mode + stepSize, dist=brzDist, interval=interval, stepSize=stepSize)
  }

  private def computeDensity(intMin: Double, intMax: Double, dist: Beta): Double = dist.cdf(intMax) - dist.cdf(intMin)

  // linear search, symetric interval, still not correct
  @tailrec
  private def linearSymmetricSearch(mode: Double, intervalWidth: Double, dist: Beta, interval: Double, stepSize: Double = 0.00001): (Double,Double) = {
    val nextMi = if ((mode - intervalWidth) >= 0.0) mode - intervalWidth else 0.0
    val nextMa = if ((mode + intervalWidth) <= 1.0) mode + intervalWidth else 1.0
    if (computeDensity(nextMi,nextMa,dist) >= interval) {
      (nextMi,nextMa)
    } else {
      linearSymmetricSearch(mode=mode, intervalWidth=intervalWidth+stepSize, dist=dist, interval=interval, stepSize=stepSize)
    }
  }

  // bisection
  //@tailrec
  def bisection(mode: Double, offset: Double, dist: Beta, interval: Double, stepSize: Double = 0.00001): (Double,Double) = {
    ???
  }

}

object CentralDensity {
  def CentralDensityRegion(posterior: BetaDistribution, interval: Double = 0.95): (Double,Double) = {
    val alpha = (1.0 - interval) / 2.0
    val lower: Double = posterior.inverseCumulativeProbability( alpha )
    val upper: Double = posterior.inverseCumulativeProbability( 1 - alpha )
    (lower,upper)
  }
}
