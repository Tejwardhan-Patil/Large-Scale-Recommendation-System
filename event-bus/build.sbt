name := "event-bus"

version := "1.0"

scalaVersion := "2.13.8"

organization := "com.mycompany"

libraryDependencies ++= Seq(
  // Kafka dependencies for producers and consumers
  "org.apache.kafka" %% "kafka" % "2.8.0",
  "org.apache.kafka" %% "kafka-streams" % "2.8.0",
  "org.apache.kafka" %% "kafka-clients" % "2.8.0",

  // JSON library for event serialization/deserialization
  "org.json" % "json" % "20210307",

  // Logging dependencies (Logback for Kafka logging)
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "ch.qos.logback" % "logback-core" % "1.2.3",

  // Postgres JDBC driver for database interaction
  "org.postgresql" % "postgresql" % "42.2.20",

  // Testing libraries
  "org.scalatest" %% "scalatest" % "3.2.9" % Test,
  "org.mockito" %% "mockito-scala" % "1.16.42" % Test
)

// Enable parallel execution of tests
parallelExecution in Test := true

// Add repositories (Confluent)
resolvers ++= Seq(
  "Confluent" at "https://packages.confluent.io/maven/"
)

// Enable forked JVM for running tests
fork in Test := true

// Package settings
enablePlugins(sbtassembly.AssemblyPlugin)

// Assembly plugin configuration for creating a fat JAR
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

// Set the main class for the assembly (packaged fat JAR)
mainClass in assembly := Some("consumers.FeedbackConsumer")

// Test settings
testOptions in Test += Tests.Argument(TestFrameworks.ScalaTest, "-oD")