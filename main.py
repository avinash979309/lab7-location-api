import csv, heapq, math
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from scipy.spatial import cKDTree

app = FastAPI()

pts = []
ids = []

with open("locations.csv", newline="") as f:
    for row in csv.reader(f):
        if len(row) < 4:
            continue
        try:
            id_ = row[0].strip()
            la, lo, ca = float(row[1]), float(row[2]), row[3].strip()
            ids.append(id_)
            pts.append([la, lo, ca])
        except ValueError:
            pass

coords = np.array([[p[0], p[1]] for p in pts])
cats = [p[2] for p in pts]
tree = cKDTree(coords)


@app.post("/search/")
async def search(
    lat: float = Form(...),
    long: float = Form(...),
    cat: str = Form(...),
    rad: float = Form(...),
    link: UploadFile = File(...),
):
    content = (await link.read()).decode(errors="ignore")

    # parse all valid lines first, then batch-query KD-tree
    endpoints = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        try:
            endpoints.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue

    graph = {}
    if endpoints:
        ep = np.array(endpoints)
        _, ai_all = tree.query(ep[:, :2])
        _, bi_all = tree.query(ep[:, 2:])
        seen = set()
        for ai, bi in zip(ai_all, bi_all):
            if ai == bi:
                continue
            key = (min(ai, bi), max(ai, bi))
            if key in seen:
                continue
            seen.add(key)
            w = math.dist(coords[ai], coords[bi])
            graph.setdefault(int(ai), []).append((int(bi), w))
            graph.setdefault(int(bi), []).append((int(ai), w))

    _, snap = tree.query([lat, long])

    candidates = set(
        i for i in tree.query_ball_point([lat, long], rad) if cats[i] == cat
    )

    dist = {snap: 0.0}
    heap = [(0.0, snap)]
    result = []

    while heap and len(result) < 10:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u in candidates:
            result.append(ids[u])
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return {"ids": result}
