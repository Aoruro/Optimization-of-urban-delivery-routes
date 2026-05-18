import numpy as np
import heapq
from collections import deque
import time
from functools import lru_cache
# ========== 辅助函数：MST 下界（Prim算法） ==========
def prim_mst(mask, current, dist_matrix):
    n = dist_matrix.shape[0]
    # 生成节点列表：当前节点 + 所有未访问节点
    nodes = [current]
    for i in range(n):
        if not (mask & (1 << i)) and i != current:
            nodes.append(i)
    k = len(nodes)
    if k <= 1:
        return 0.0
    # Prim算法，直接在原距离矩阵上查询，不构建子矩阵
    visited = [False] * k
    min_edge = [float('inf')] * k
    min_edge[0] = 0.0
    total = 0.0
    for _ in range(k):
        u = -1
        for i in range(k):
            if not visited[i] and (u == -1 or min_edge[i] < min_edge[u]):
                u = i
        visited[u] = True
        total += min_edge[u]
        # 更新邻接边
        for v in range(k):
            if not visited[v]:
                d = dist_matrix[nodes[u]][nodes[v]]
                if d < min_edge[v]:
                    min_edge[v] = d
    return total

# ========== 状态表示 ==========
def state_key(visited_mask, current):
    return (visited_mask, current)

# ========== BFS（按步数，不考虑距离） ==========
def bfs_tsp(dist_matrix):
    n = len(dist_matrix)
    start_mask = 1 << 0
    start = 0
    queue = deque()
    queue.append((start_mask, start, [start]))
    visited_states = set()
    visited_states.add((start_mask, start))
    
    nodes_expanded = 0
    while queue:
        mask, curr, path = queue.popleft()
        nodes_expanded += 1
        if mask == (1<<n) - 1:
            # 返回起点
            total_dist = sum(dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
            total_dist += dist_matrix[path[-1]][path[0]]
            return path + [path[0]], total_dist, nodes_expanded
        # 扩展
        for nxt in range(n):
            if mask & (1<<nxt):
                continue
            new_mask = mask | (1<<nxt)
            if (new_mask, nxt) not in visited_states:
                visited_states.add((new_mask, nxt))
                queue.append((new_mask, nxt, path + [nxt]))
    return None, float('inf'), nodes_expanded

# ========== UCS（Dijkstra） ==========
def ucs_tsp(dist_matrix):
    n = len(dist_matrix)
    start_mask = 1 << 0
    start = 0

    # 优先队列元素不再包含路径
    pq = [(0.0, start_mask, start)]
    best_cost = {}
    parent = {}
    nodes_expanded = 0

    best_cost[(start_mask, start)] = 0.0
    parent[(start_mask, start)] = None   # 回溯终止标记

    while pq:
        cost, mask, curr = heapq.heappop(pq)

        # 跳过已经被更优成本访问过的状态
        if cost > best_cost.get((mask, curr), float('inf')):
            continue

        nodes_expanded += 1

        # 所有城市均已访问，准备返回起点
        if mask == (1 << n) - 1:
            total_cost = cost + dist_matrix[curr][0]

            # 回溯重建路径
            path = []
            cur_mask, cur_node = mask, curr
            while cur_mask != start_mask or cur_node != start:
                path.append(cur_node)
                prev_node = parent[(cur_mask, cur_node)]
                cur_mask ^= (1 << cur_node)
                cur_node = prev_node
            path.append(start)
            path.reverse()
            path.append(0)      # 返回起点
            return path, total_cost, nodes_expanded

        # 扩展邻居
        for nxt in range(n):
            if mask & (1 << nxt):
                continue
            new_mask = mask | (1 << nxt)
            new_cost = cost + dist_matrix[curr][nxt]
            old_cost = best_cost.get((new_mask, nxt))

            if old_cost is None or new_cost < old_cost:
                best_cost[(new_mask, nxt)] = new_cost
                parent[(new_mask, nxt)] = curr
                heapq.heappush(pq, (new_cost, new_mask, nxt))

    return None, float('inf'), nodes_expanded

# ========== A*（启发式 = MST下界） ==========
def a_star_tsp(dist_matrix):
    n = len(dist_matrix)
    start_mask = 1 << 0
    start = 0
    full_mask = (1 << n) - 1
    
    @lru_cache(maxsize=None)
    def heuristic(mask, curr):
        """可采纳启发式：当前节点+未访问节点的MST长度（不含返回起点）"""
        if mask == full_mask:
            return dist_matrix[curr][0]  # 直接返回起点的距离
        # 计算MST（包括当前节点和所有未访问节点）
        return prim_mst(mask, curr, dist_matrix)
    
    # 编码状态：key = (mask << 5) | curr  (n<=32)
    g = {}
    f = {}
    start_key = (start_mask << 5) | start
    g[start_key] = 0.0
    f[start_key] = heuristic(start_mask, start)
    pq = [(f[start_key], start_mask, start, [start])]
    
    nodes_expanded = 0
    while pq:
        f_val, mask, curr, path = heapq.heappop(pq)
        key = (mask << 5) | curr
        if f_val != f.get(key, None):
            continue
        nodes_expanded += 1
        if mask == full_mask:
            total = g[key] + dist_matrix[curr][0]
            return path + [0], total, nodes_expanded
        # 生成后继
        for nxt in range(n):
            if mask & (1 << nxt):
                continue
            new_mask = mask | (1 << nxt)
            new_g = g[key] + dist_matrix[curr][nxt]
            new_key = (new_mask << 5) | nxt
            if new_key not in g or new_g < g[new_key]:
                g[new_key] = new_g
                new_f = new_g + heuristic(new_mask, nxt)
                f[new_key] = new_f
                heapq.heappush(pq, (new_f, new_mask, nxt, path + [nxt]))
    return None, float('inf'), nodes_expanded

if __name__ == "__main__":
    from 数据处理 import load_single_tsp_instance_no_header 
    instance = load_single_tsp_instance_no_header(r"C:\Users\20682\Documents\dataset.xlsx")
    dist_matrix = instance['dist_matrix']
    n = instance['num_cities']
    print(f"Testing with {n} cities")
    
    # BFS
    start = time.time()
    path, dist, expanded = bfs_tsp(dist_matrix)
    t_bfs = time.time() - start
    print(f"BFS: expanded={expanded}, dist={dist:.2f}, time={t_bfs:.4f}s")
    
    # UCS
    start = time.time()
    path, dist, expanded = ucs_tsp(dist_matrix)
    t_ucs = time.time() - start
    print(f"UCS: expanded={expanded}, dist={dist:.2f}, time={t_ucs:.4f}s")
    
    # A*
    start = time.time()
    path, dist, expanded = a_star_tsp(dist_matrix)
    t_astar = time.time() - start
    print(f"A*: expanded={expanded}, dist={dist:.2f}, time={t_astar:.4f}s")
