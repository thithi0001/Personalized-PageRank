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

start_time = time.time()

# Tạo RDD từ danh sách cạnh
links = sc.parallelize(edges).distinct().groupByKey().mapValues(list).cache()
# print("RDD:",links.collect())

# Lấy toàn bộ node
all_nodes = sc.parallelize(nodes_list)
# all_nodes = links.flatMap(lambda x: [x[0]] + x[1]).distinct().cache()
# print("RDD:",all_nodes.collect())

# Khởi tạo Personalized PageRank
s = all_nodes.map(lambda node: (node, (1 - damping) if node == source else 0.0)).cache()
# print("RDD:",s.collect())
ranks = all_nodes.map(lambda node: (node, 1.0 if node == source else 0.0)).cache()
# print("RDD:",ranks.collect())

# Chạy thuật toán Personalized PageRank
for _ in range(max_iteration):
    prev_ranks = ranks

    contribs = links.join(ranks).flatMap(
        lambda x: [(dest, x[1][1] / len(x[1][0])) for dest in x[1][0]]
    )

    ranks = contribs.reduceByKey(lambda x, y: x + y)

    ranks = ranks.rightOuterJoin(s).mapValues(
        lambda pair: (pair[1] if pair[0] is None else damping * pair[0] + pair[1])
    )

    ranks = ranks.cache()

    diff = ranks.join(prev_ranks).mapValues(lambda x: abs(x[0] - x[1]))
    total_diff = diff.values().sum()

    if total_diff < EPSILON:
        print(f"Iteration {_+1}, total diff = {total_diff:.8f}")
        break

print("Final ranks after max_iteration:")
# # In kết quả cuối cùng
for node, rank in ranks.collect():
    print(f"{node}: {rank:.6f}")

end_time = time.time()
print(f"TIME: {end_time - start_time:.6f} seconds")

print("Done.")
sc.stop()