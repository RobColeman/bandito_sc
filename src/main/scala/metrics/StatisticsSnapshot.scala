package metrics

case class StatisticsSnapshot(trials: Int, successes: Int, banditsModelsStats: Vector[String])
