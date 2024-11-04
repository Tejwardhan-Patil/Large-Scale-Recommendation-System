name := "RecommendationEngine"

version := "1.0"

scalaVersion := "2.13.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.3.0",
  "org.apache.spark" %% "spark-mllib" % "3.3.0",
  "org.scalatest" %% "scalatest" % "3.2.14" % Test,
  "com.typesafe.akka" %% "akka-stream" % "2.6.20",
  "com.typesafe.akka" %% "akka-http" % "10.2.10",
  "com.typesafe" % "config" % "1.4.2"
)

resolvers += "Akka Repository" at "https://repo.akka.io/releases/"

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

scalacOptions ++= Seq(
  "-deprecation",
  "-encoding", "utf8",
  "-feature",
  "-unchecked",
  "-Xfatal-warnings",
  "-Xlint:_",
  "-Ywarn-dead-code"
)

test in assembly := {}