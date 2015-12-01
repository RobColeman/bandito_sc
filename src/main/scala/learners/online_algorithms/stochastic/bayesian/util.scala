package learners.online_algorithms.stochastic.bayesian

import scala.util.Random


object util {

  def selectHighestScore(scores: Vector[Double]): Int = {
    val ma = scores.max
    val highest = scores.zipWithIndex.filter( _._1 == ma )
    val l = highest.length
    l match {
      case 1 => highest.head._2
      case _ => highest(Random.nextInt(l))._2 // tie-breaker is uniform random
    }
  }

}
