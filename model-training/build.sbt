name := "model-training"

version := "1.0"

scalaVersion := "2.13.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.0",
  "org.apache.spark" %% "spark-mllib" % "3.2.0",
  "org.scalatest" %% "scalatest" % "3.2.9" % Test,
  "com.typesafe" % "config" % "1.4.1"
)

resolvers += "Apache Repository" at "https://repository.apache.org/content/repositories/releases/"

scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked")

testOptions in Test += Tests.Argument("-oD")