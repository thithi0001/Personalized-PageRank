from collections import defaultdict

out_links = defaultdict(list)
in_links = defaultdict(list)
nodes = set()

edges = [
    ("P1", "P2"), ("P1", "P3"),
    ("P3", "P1"), ("P3", "P2"), ("P3", "P5"),
    ("P5", "P4"), ("P5", "P6"),
    ("P4", "P5"), ("P4", "P6"),
    ("P6", "P4"),
]

damping = 0.85
max_iteration = 1000
source = "P1"
EPSILON = 1e-6

for src, dst in edges:
    out_links[src].append(dst)
    in_links[dst].append(src)
    nodes.update([src, dst])

ranks = {node: (1.0 if node == source else 0.0) for node in nodes}

f = open('result.txt',"w")
for i in range(max_iteration):
    print(f"\n--- Iteration {i+1} ---")
    f.write(f"\n--- Iteration {i+1} ---\n")
    new_ranks = {}
    for node in nodes:
        rank_sum = 0.0
        for src in in_links[node]:
            if len(out_links[src]) > 0:
                rank_sum += ranks[src] / len(out_links[src])

        s = 1.0 if node == source else 0.0
        new_ranks[node] = (1 - damping) * s + damping * rank_sum

    diff = sum(abs(ranks[node] - new_ranks[node]) for node in nodes)
    ranks = new_ranks
    for node, rank in ranks.items():
        f.write(f"{node}: {rank:.6f}\n")
    if diff < EPSILON:
        print(f"Total diff = {diff:.8f}")
        f.write(f"Total diff = {diff:.8f}\n")
        break


for node, rank in ranks.items():
    print(f"{node}: {rank:.6f}")

f.close()