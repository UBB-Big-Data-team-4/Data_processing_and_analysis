from pyspark.sql import SparkSession, DataFrame
import datetime

def getProcessedData() -> DataFrame:
    spark = SparkSession.builder.appName("BigDataProject").getOrCreate()

    mockData = []
    t = datetime.datetime.now()

    for i in range (2000):
        mockData.append([t, i // 500])
        t = t + datetime.timedelta(seconds=1)

    df = spark.createDataFrame(
        mockData
        , schema='timestamp TIMESTAMP, people INTEGER'
    )

    return df