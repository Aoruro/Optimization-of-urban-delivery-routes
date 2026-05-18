import pandas as pd
import numpy as np

def load_single_tsp_instance_no_header(file_path):
    """
    读取没有列名的 Excel 文件（单行数据）。
    数据格式：
        实例ID, 城市数, 总距离, 路线类别,
        City1_X, City1_Y, City2_X, City2_Y, ..., City20_X, City20_Y
    """
    df = pd.read_excel(file_path, header=None)  # 没有标题行
    if df.shape[0] != 1:
        print(f"警告：文件包含 {df.shape[0]} 行，将只使用第一行。")
    row = df.iloc[0]
    
    inst_id = str(row[0])          # 实例ID（可能是字符串）
    n = int(row[1])                # 城市数量
    total_dist = float(row[2]) if pd.notna(row[2]) else None
    best_cat = int(row[3]) if pd.notna(row[3]) else None
    
    # 提取坐标：从第4列（索引4）开始，每两列一组
    coords = []
    for i in range(n):
        x = row[4 + 2*i]
        y = row[4 + 2*i + 1]
        coords.append([x, y])
    coords = np.array(coords, dtype=float)
    
    # 计算欧氏距离矩阵
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i][j] = d
            dist_matrix[j][i] = d
    
    return {
        'id': inst_id,
        'num_cities': n,
        'coords': coords,
        'dist_matrix': dist_matrix,
        'total_distance': total_dist,
        'best_route_category': best_cat
    }

if __name__ == "__main__":
    instance = load_single_tsp_instance_no_header(r"C:\Users\20682\Documents\工作簿1.xlsx")
    print(f"Instance ID: {instance['id']}")
    print(f"Number of cities: {instance['num_cities']}")
    print(f"Total distance (given): {instance['total_distance']}")
    print(f"Coordinates (first 3 cities):\n{instance['coords'][:3]}")
    print(f"Distance matrix shape: {instance['dist_matrix'].shape}")
    print(f"Example distance between city0 and city1: {instance['dist_matrix'][0][1]:.4f}")