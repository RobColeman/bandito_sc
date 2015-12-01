name := "bandito_sc"

version := "1.0"

scalaVersion := "2.10.4"

val scalaTestVersion = "2.2.4"
val breezeVersion = "0.11.2"

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest"          % scalaTestVersion  % "test",
  "org.scalanlp"  %% "breeze"             % breezeVersion,
  "org.scalanlp"  %% "breeze-natives"     % breezeVersion,
  "org.apache.commons" % "commons-math3"  % "3.3"
)