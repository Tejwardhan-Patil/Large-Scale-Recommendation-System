name := "DataIngestion"

version := "1.0.0"

scalaVersion := "2.13.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.0",
  "org.apache.spark" %% "spark-sql" % "3.2.0",
  "com.typesafe" % "config" % "1.4.1",
  "org.scalatest" %% "scalatest" % "3.2.9" % Test,
  "mysql" % "mysql-connector-java" % "8.0.26",
  "org.apache.kafka" %% "kafka" % "2.8.0"
)

resolvers += "Confluent" at "https://packages.confluent.io/maven/"

scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked",
  "-Xfatal-warnings"
)

assemblyJarName in assembly := "data-ingestion-assembly.jar"

mainClass in assembly := Some("com.website.data.ingestion.Main")