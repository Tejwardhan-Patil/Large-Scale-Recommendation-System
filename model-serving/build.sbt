name := "Model-Serving"

version := "1.0"

scalaVersion := "2.13.6"

libraryDependencies ++= Seq(
  "org.scalatra" %% "scalatra" % "2.7.1",
  "com.typesafe.akka" %% "akka-http" % "10.2.6",
  "org.json4s" %% "json4s-native" % "3.7.0-M5",
  "org.slf4j" % "slf4j-api" % "1.7.30",
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "com.typesafe.akka" %% "akka-stream" % "2.6.14",
  "org.scalatest" %% "scalatest" % "3.2.9" % "test"
)

assemblyJarName in assembly := "model-serving.jar"

mainClass in Compile := Some("com.website.modelserving.Main")

enablePlugins(JavaAppPackaging)

scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-encoding", "utf8")