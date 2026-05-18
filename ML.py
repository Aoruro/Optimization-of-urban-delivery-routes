import pandas as pd
import numpy as np
import heapq
import time
import random
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ==================== 数据加载（支持多实例） ====================
def load_all_tsp_instances(file_path):
    """
    读取包含多个TSP实例的Excel文件（无表头），每行一个实例。
    返回实例列表，每个实例为 dict。
    """
    df = pd.read_excel(file_path, header=None)
    instances = []
    for idx, row in df.iterrows():
        inst_id = str(row[0])
        n = int(row[1])
        total_dist = float(row[2]) if pd.notna(row[2]) else None
        best_cat = int(row[3]) if pd.notna(row[3]) else None
        
        coords = []
        for i in range(n):
            x = row[4 + 2*i]
            y = row[4 + 2*i + 1]
            coords.append([x, y])
        coords = np.array(coords, dtype=float)
        
        # 欧氏距离矩阵
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d
        
        instances.append({
            'id': inst_id,
            'num_cities': n,
            'coords': coords,
            'dist_matrix': dist_matrix,
            'total_distance': total_dist,
            'best_route_category': best_cat
        })
    return instances

# ==================== A* 算法（优化版） ====================
def prim_mst(mask, current, dist_matrix):
    n = dist_matrix.shape[0]
    nodes = [current] + [i for i in range(n) if not (mask & (1 << i)) and i != current]
    k = len(nodes)
    if k <= 1:
        return 0.0
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
        for v in range(k):
            if not visited[v]:
                d = dist_matrix[nodes[u]][nodes[v]]
                if d < min_edge[v]:
                    min_edge[v] = d
    return total

def a_star_tsp(dist_matrix, time_limit=60):
    n = len(dist_matrix)
    start_mask = 1 << 0
    start = 0
    full_mask = (1 << n) - 1
    
    @lru_cache(maxsize=None)
    def heuristic(mask, curr):
        if mask == full_mask:
            return dist_matrix[curr][0]
        return prim_mst(mask, curr, dist_matrix)
    
    g = {}
    f = {}
    start_key = (start_mask << 5) | start
    g[start_key] = 0.0
    f[start_key] = heuristic(start_mask, start)
    pq = [(f[start_key], start_mask, start, [start])]
    
    nodes_expanded = 0
    start_time = time.time()
    while pq:
        if time.time() - start_time > time_limit:
            print(f"  A* 超时（{time_limit}s），放弃该实例")
            return None, float('inf'), nodes_expanded
        f_val, mask, curr, path = heapq.heappop(pq)
        key = (mask << 5) | curr
        if f_val != f.get(key, None):
            continue
        nodes_expanded += 1
        if mask == full_mask:
            total = g[key] + dist_matrix[curr][0]
            return path + [0], total, nodes_expanded
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

# ==================== 样本提取 ====================
def extract_samples_from_path(dist_matrix, path, num_negatives=2):
    n = len(dist_matrix)
    full_path = path[:-1]  # 去掉末尾重复的起点
    num_cities = len(full_path)
    samples = []
    visited_mask = 0
    g = 0.0
    for idx, current in enumerate(full_path):
        visited_mask |= (1 << current)
        next_city = full_path[(idx + 1) % num_cities]
        # 正样本
        samples.append({
            'visited_mask': visited_mask,
            'current': current,
            'candidate': next_city,
            'label': 1,
            'g': g
        })
        # 负样本
        unvisited = [c for c in range(n) if not (visited_mask & (1 << c)) and c != current]
        if next_city in unvisited:
            unvisited.remove(next_city)
        neg_candidates = random.sample(unvisited, min(num_negatives, len(unvisited)))
        for neg in neg_candidates:
            samples.append({
                'visited_mask': visited_mask,
                'current': current,
                'candidate': neg,
                'label': 0,
                'g': g
            })
        # 更新 g
        if idx + 1 < num_cities:
            g += dist_matrix[current][full_path[idx+1]]
        else:
            g += dist_matrix[current][full_path[0]]
    return samples

# ==================== 特征提取 ====================
def compute_features(sample, dist_matrix, coords):
    mask = sample['visited_mask']
    curr = sample['current']
    cand = sample['candidate']
    g = sample['g']
    n = dist_matrix.shape[0]
    
    unvisited = [c for c in range(n) if not (mask & (1 << c)) and c != curr]
    if cand in unvisited:
        other_unvisited = [c for c in unvisited if c != cand]
    else:
        other_unvisited = unvisited
    
    dist_curr_cand = dist_matrix[curr][cand]
    min_dist_cand_to_unv = min([dist_matrix[cand][u] for u in other_unvisited]) if other_unvisited else 0.0
    avg_dist_cand_to_unv = np.mean([dist_matrix[cand][u] for u in other_unvisited]) if other_unvisited else 0.0
    unvisited_ratio = len(unvisited) / n
    avg_all_dist = np.mean(dist_matrix)
    g_norm = g / avg_all_dist if avg_all_dist > 0 else g
    avg_dist_cand_to_all = np.mean([dist_matrix[cand][c] for c in range(n) if c != cand])
    
    return np.array([
        dist_curr_cand,
        min_dist_cand_to_unv,
        avg_dist_cand_to_unv,
        unvisited_ratio,
        g_norm,
        avg_dist_cand_to_all
    ])

def extract_feature_matrix(samples, dist_matrix, coords):
    X, y = [], []
    for s in samples:
        X.append(compute_features(s, dist_matrix, coords))
        y.append(s['label'])
    return np.array(X), np.array(y)

# ==================== 主流程：多实例收集样本 ====================
def collect_samples_with_features(instances, max_instances=5, time_limit_per_instance=60):
    X_all = []
    y_all = []
    for idx, inst in enumerate(instances[:max_instances]):
        print(f"\n处理实例 {idx+1}: ID={inst['id']}, 城市数={inst['num_cities']}")
        if inst['num_cities'] > 25:
            print("  跳过（城市数>25）")
            continue
        dist_matrix = inst['dist_matrix']
        coords = inst['coords']
        
        path, dist, expanded = a_star_tsp(dist_matrix, time_limit_per_instance)
        if path is None:
            print("  A*未完成，跳过")
            continue
        print(f"  A*完成: 长度={dist:.2f}, 扩展节点={expanded}")
        
        samples = extract_samples_from_path(dist_matrix, path, num_negatives=2)
        # 立即计算特征
        for s in samples:
            feat = compute_features(s, dist_matrix, coords)
            X_all.append(feat)
            y_all.append(s['label'])
        print(f"  本实例贡献 {len(samples)} 个样本，累计特征 {len(X_all)}")
    return np.array(X_all), np.array(y_all)

# ==================== 训练与评估 ====================
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n模型准确率: {acc:.4f}")
    # 特征重要性
    importances = clf.feature_importances_
    feature_names = ['dist_curr_cand', 'min_dist_to_unv', 'avg_dist_to_unv', 'unvisited_ratio', 'g_norm', 'avg_dist_to_all']
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.4f}")
    # 可选：绘图
    plt.figure(figsize=(8,4))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()
    return clf

# ==================== 主程序 ====================
if __name__ == "__main__":
    file_path = r"C:\Users\20682\Documents\dataset.xlsx"
    instances = load_all_tsp_instances(file_path)
    print(f"加载 {len(instances)} 个实例")
    
    X, y = collect_samples_with_features(instances, max_instances=5, time_limit_per_instance=120)
    if len(X) == 0:
        print("没有有效样本，请检查实例规模或增大时间限制")
        exit()
    
    print(f"\n总样本数: {len(X)}, 正样本比例: {np.mean(y):.3f}")
    model = train_and_evaluate(X, y)
    
    # 保存模型供后续学习型A*使用
    import joblib
    joblib.dump(model, "tsp_model.pkl")
    print("模型已保存为 tsp_model.pkl")