package banditModels

import scala.collection.mutable

case class BernoulliTrialsLimitedHistory(maxMemory: Int = 10000) {

  private var nTrials: Int = 0
  private var nSuccesses: Int = 0
  private var nTotalTrials: Long = 0
  private var nTotalSuccesses: Long = 0

  private val trialHistory: mutable.Queue[Boolean] = new mutable.Queue[Boolean]

  def trials: Int = this.nTrials
  def successes: Int = this.nSuccesses
  def totalTrials: Long = this.nTotalTrials
  def totalSuccesses: Long = this.nTotalSuccesses

  private def updateFull(trailResult: Boolean): Unit = {
    val oldestTrial: Boolean = this.trialHistory.dequeue()
    if (oldestTrial) this.nSuccesses -= 1

    this.nTotalTrials += 1
    if (trailResult) this.nSuccesses += 1

  }

  private def updateNotFull(trailResult: Boolean): Unit = {
    this.nTotalTrials += 1
    this.nTrials += 1
    if (trailResult) {
      this.nSuccesses += 1
      this.nTotalSuccesses += 1
    }
    this.trialHistory.enqueue(trailResult)
  }

  def update(trailResult: Boolean): Unit = {
    nTotalTrials < maxMemory match {
      case true => this.updateNotFull(trailResult)
      case false => this.updateFull(trailResult)
    }
  }

}