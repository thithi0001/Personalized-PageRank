from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("TestSpark").setMaster("local")
sc = SparkContext(conf=conf)

rdd = sc.parallelize(["Hello", "from", "Spark"])
print("RDD:",rdd.collect())

sc.stop()
