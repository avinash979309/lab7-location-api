import csv
import heapq
import math

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from scipy.spatial import cKDTree

app = FastAPI()

all_ids = []
all_coords = []
all_cats = []

try:
    with open("locations.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_ids.append(row["ID"].strip())
            all_coords.append([float(row["Latitude"]), float(row["Longitude"])])
            all_cats.append(row["Category"].strip())
except FileNotFoundError:
    print("locations.csv not found!")

coords = np.array(all_coords)
tree = cKDTree(coords)


@app.post("/search/")
async def search(
    lat: float = Form(...),
    long: float = Form(...),
    cat: str = Form(...),
    rad: float = Form(...),
    roads: UploadFile = File(...),
):
    text = (await roads.read()).decode(errors="ignore")

    raw = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        try:
            raw.append((float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            continue

    graph = {}
    if raw:
        pts1 = np.array([[r[0], r[1]] for r in raw])
        pts2 = np.array([[r[2], r[3]] for r in raw])
        _, idx1 = tree.query(pts1)
        _, idx2 = tree.query(pts2)

        seen = set()
        for a, b in zip(idx1, idx2):
            a, b = int(a), int(b)
            if a == b:
                continue
            edge = (min(a, b), max(a, b))
            if edge in seen:
                continue
            seen.add(edge)
            d = math.dist(coords[a], coords[b])
            graph.setdefault(a, []).append((b, d))
            graph.setdefault(b, []).append((a, d))

    _, src = tree.query([lat, long])
    src = int(src)

    nearby = tree.query_ball_point([lat, long], rad)
    targets = set(i for i in nearby if all_cats[i] == cat)

    visited = {}
    heap = [(0.0, src)]
    found = []

    while heap and len(found) < 10:
        cost, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited[node] = cost
        if node in targets:
            found.append(all_ids[node])
        for nbr, w in graph.get(node, []):
            if nbr not in visited:
                heapq.heappush(heap, (cost + w, nbr))

    return {"ids": found}
