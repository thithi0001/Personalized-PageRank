import os
import time

# Ẩn log INFO/DEBUG của Spark
os.environ["PYSPARK_SUBMIT_ARGS"] = "--conf spark.ui.showConsoleProgress=false pyspark-shell"
import logging
logging.getLogger("py4j").setLevel(logging.WARN)

# Dữ liệu cạnh của đồ thị
nodes_list = ["P1", "P2", "P3", "P4", "P5", "P6"]
edges = [
    ("P1", "P2"), ("P1", "P3"),
    ("P3", "P1"), ("P3", "P2"), ("P3", "P5"),
    ("P5", "P4"), ("P5", "P6"),
    ("P4", "P5"), ("P4", "P6"),
    ("P6", "P4"),
]

damping = 0.85
max_iteration = 5
source = "P1"
EPSILON = 1e-6

# Cấu hình Spark
from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName("Personalized PageRank").setMaster("local[*]")
conf.set("spark.driver.memory", "4g")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
spark = SparkSession.builder.config(conf=conf).getOrCreate()


start_time = time.time()
print("Start Personalized PageRank...")

edges_df = spark.createDataFrame(edges, ["node", "dst"]).cache()
# edges_df.show()

all_nodes_df = spark.createDataFrame([(n,) for n in nodes_list], ["node"]).cache()
out_links_df = edges_df.groupBy("node").agg(F.count("dst").alias("out_links"))
out_links_df = all_nodes_df.join(out_links_df, "node", "left") \
    .withColumn("out_links", F.coalesce(F.col("out_links"), F.lit(0))).cache()
# out_links_df.show()

s_df = spark.createDataFrame([(n, (1 - damping) if n == source else 0.0) for n in nodes_list], ["node", "s"]).cache()
# s_df.show()

ranks_df = spark.createDataFrame([(n, 1.0 if n == source else 0.0) for n in nodes_list], ["node", "rank"]).cache()
# ranks_df.show()

# Chạy thuật toán Personalized PageRank
for _ in range(max_iteration):
    print(f"\n--- Iteration {_+1} ---")
    r = ranks_df.join(out_links_df, on="node", how="left")
    # r.show()
    
    contribs_df = edges_df.join(r, "node", "left") \
        .withColumn("contrib",F.when(F.col("out_links") > 0, F.col("rank") / F.col("out_links")).otherwise(0.0))
    contribs_df = contribs_df.select(F.col("dst").alias("node"), "contrib")
    
    contribs_df = contribs_df.groupBy("node").agg(F.sum("contrib").alias("contrib_sum"))
    # contribs_df.show()

    ranks_df = all_nodes_df.join(contribs_df, "node", "left") \
        .join(s_df, "node", "left") \
        .fillna(0, subset=["contrib_sum"]) \
        .withColumn("rank", F.col("s") + damping * F.col("contrib_sum")) \
        .select("node", "rank")

    diff_df = ranks_df.alias("new").join(r.alias("old"), "node", "left") \
        .withColumn("old_rank", F.coalesce(F.col("old.rank"), F.lit(0.0))) \
        .withColumn("new_rank", F.coalesce(F.col("new.rank"), F.lit(0.0))) \
        .withColumn("diff", F.abs(F.col("new.rank") - F.col("old.rank")))
    # diff_df.show()
    
    ranks_df = ranks_df.cache()
    # ranks_df.show()
    
    total_diff = diff_df.agg(F.sum("diff").alias("total_diff")).first()["total_diff"] or 0.0
    
    if total_diff < EPSILON:
        print(f"Iteration {_+1}, total diff = {total_diff:.8f}")
        break

print("Final ranks:")
ranks_df.select("node", F.format_number("rank", 4).alias("rank")).show()

end_time = time.time()
print(f"TIME: {end_time - start_time:.6f} seconds")

print("Done.")
sc.stop()