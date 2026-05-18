import numpy as np
import heapq
from collections import deque
import time

# ========== 辅助函数：MST 下界（Prim算法） ==========
def mst_lower_bound(unvisited_set, current_city, dist_matrix, coords=None):
    """
    计算从当前城市出发，经过所有未访问城市的MST下界。
    参数：
        unvisited_set: list 或 set 未访问的城市索引
        current_city: 当前城市索引
        dist_matrix: 距离矩阵
    返回：MST总长度（下界）
    """
    if not unvisited_set:
        return 0.0
    nodes = [current_city] + list(unvisited_set)
    n = len(nodes)
    # 构建子图距离矩阵
    sub_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = dist_matrix[nodes[i]][nodes[j]]
            sub_dist[i][j] = d
            sub_dist[j][i] = d
    # Prim算法求MST
    visited = [False]*n
    min_edge = [float('inf')]*n
    min_edge[0] = 0
    total = 0.0
    for _ in range(n):
        u = -1
        for i in range(n):
            if not visited[i] and (u == -1 or min_edge[i] < min_edge[u]):
                u = i
        visited[u] = True
        total += min_edge[u]
        for v in range(n):
            if not visited[v] and sub_dist[u][v] < min_edge[v]:
                min_edge[v] = sub_dist[u][v]
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
    # f = g + h
    def heuristic(mask, curr):
        unvisited = [i for i in range(n) if not (mask & (1<<i)) and i != curr]
        if not unvisited:
            return dist_matrix[curr][0]  # 回到起点的距离
        # 包含当前节点和未访问节点的MST
        nodes = [curr] + unvisited
        # 快速计算MST
        return mst_lower_bound(unvisited, curr, dist_matrix)
    
    g_score = {(start_mask, start): 0.0}
    f_score = {(start_mask, start): heuristic(start_mask, start)}
    pq = [(f_score[(start_mask, start)], start_mask, start, [start])]
    nodes_expanded = 0
    
    while pq:
        f, mask, curr, path = heapq.heappop(pq)
        if f != f_score.get((mask, curr), None):
            continue
        nodes_expanded += 1
        if mask == (1<<n) - 1:
            total = g_score[(mask, curr)] + dist_matrix[curr][0]
            return path + [0], total, nodes_expanded
        for nxt in range(n):
            if mask & (1<<nxt):
                continue
            new_mask = mask | (1<<nxt)
            new_g = g_score[(mask, curr)] + dist_matrix[curr][nxt]
            if (new_mask, nxt) not in g_score or new_g < g_score[(new_mask, nxt)]:
                g_score[(new_mask, nxt)] = new_g
                new_f = new_g + heuristic(new_mask, nxt)
                f_score[(new_mask, nxt)] = new_f
                heapq.heappush(pq, (new_f, new_mask, nxt, path + [nxt]))
    return None, float('inf'), nodes_expanded

# ========== 测试 ==========
if __name__ == "__main__":
    # 加载数据（使用你之前的函数）
    from 数据处理 import load_single_tsp_instance_no_header  # 替换为你的实际导入
    instance = load_single_tsp_instance_no_header(r"C:\Users\20682\Documents\数据集.xlsx")
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