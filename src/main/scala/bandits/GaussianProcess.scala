package bandits

import gov.sandia.cognition.statistics.distribution.NormalInverseWishartDistribution
import breeze.stats.distributions.MultivariateGaussian
import breeze.linalg._

object GaussianProcess {
  def apply(startValues: DenseVector[Double], deltaMeanVec: DenseVector[Double], deltaCovMatrix: DenseMatrix[Double]): GaussianProcess =
    new GaussianProcess(startValues,deltaMeanVec,deltaCovMatrix)
}
class GaussianProcess(private val startValues: DenseVector[Double], private val deltaMeanVec: DenseVector[Double], private val deltaCovMatrix: DenseMatrix[Double]) {

  val D = startValues.length

  def getStartValues: DenseVector[Double] = this.startValues
  def getDeltaMeanVec: DenseVector[Double] = this.deltaMeanVec
  def getDeltaCovMatrix: DenseMatrix[Double] = this.deltaCovMatrix

  private var nSamples = 0
  def getNSamples: Int = this.nSamples

  private var currentValue: DenseVector[Double] = this.startValues.copy
  def getCurrentValue: DenseVector[Double] = this.currentValue

  private var historicValues: DenseMatrix[Double] = this.startValues.copy.toDenseMatrix.t
  def getHistoricValues: DenseMatrix[Double] = this.historicValues

  def getHistoricDeltas: DenseMatrix[Double] = {
    val shifted = DenseMatrix.horzcat(DenseMatrix.zeros[Double](this.D, 1), this.historicValues(::,0 to -2))
    val diff = this.historicValues - shifted
    diff(::,1 to -1)
  }
  def getTotalDelta: DenseVector[Double] = this.currentValue - this.startValues

  val deltaDistribution: MultivariateGaussian = new MultivariateGaussian(this.deltaMeanVec, this.deltaCovMatrix)

  def step: Unit = {
    this.currentValue += deltaDistribution.draw()
    this.historicValues = DenseMatrix.horzcat(this.historicValues,this.currentValue.toDenseMatrix.t)
    this.nSamples += 1
  }
  def step(n: Int): Unit = {
    (0 until n).foreach{ i => this.step}
  }

}



object GaussianProcessApp {
  def main(args: Array[String]): Unit = {
    val dims: Int = 4
    val svRange = 10.0
    val svMean = 50.0
    val deltaMeanRange = 2.0
    val covRange = 2.0
    val startValues = DenseVector.rand[Double](dims).map{ x => x * svRange + svMean }
    val deltaMeanVec = DenseVector.rand[Double](dims).map{ x => ( x - 0.5) * deltaMeanRange }
    val deltaCovMatrix = {
      val x: DenseMatrix[Double] = DenseMatrix.rand[Double](dims,dims).map{ _ * covRange }
      val C: DenseMatrix[Double] = x * x.t
      C
    }

    val GPBandit: GaussianProcess = new GaussianProcess(startValues,deltaMeanVec,deltaCovMatrix)
    GPBandit.step(100)

    val deltas = GPBandit.getHistoricDeltas
    val delta = GPBandit.getTotalDelta

  }
}