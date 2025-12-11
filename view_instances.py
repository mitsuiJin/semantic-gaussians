# python view_instances.py \
#   --ply_path "/data/shjin1019/repos/semantic-gaussians/output/scene0073_03_rgb_train/point_cloud/iteration_7000/point_cloud.ply" \
#   --npy_path "instance_ids_7k.npy" \
#   --json_path "scene_graph_nodes_7k_2.json"


import viser
import numpy as np
import argparse
import os
import json
import time
from plyfile import PlyData

def load_ply_xyz(path):
    print(f"Loading PLY from {path}...")
    plydata = PlyData.read(path)
    xyz = np.stack((plydata['vertex']['x'], 
                    plydata['vertex']['y'], 
                    plydata['vertex']['z']), axis=1)
    return xyz

def get_random_colors(n):
    np.random.seed(42)
    return np.random.randint(0, 255, size=(n, 3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="Path to point_cloud.ply")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to instance_ids.npy")
    # JSON 경로는 필수가 아닌 선택으로 둡니다 (없으면 점만 보여줌)
    parser.add_argument("--json_path", type=str, default=None, help="Path to scene_graph_nodes.json")
    args = parser.parse_args()

    server = viser.ViserServer()
    print("Viser Server started!")

    # 1. Geometry & Instance IDs 로드 (점군 시각화용)
    xyz = load_ply_xyz(args.ply_path)
    instance_ids = np.load(args.npy_path)
    
    print(f"Total points: {len(xyz)}")
    print(f"Clustered points: {np.sum(instance_ids != -1)}")

    # 색상 입히기 (배경: 회색)
    colors = np.full_like(xyz, 200, dtype=np.uint8)
    
    max_id = instance_ids.max()
    if max_id >= 0:
        palette = get_random_colors(max_id + 1)
        mask = (instance_ids != -1)
        colors[mask] = palette[instance_ids[mask]]

    # 다운샘플링 (뷰어 성능 최적화)
    if len(xyz) > 1000000:
        print("Downsampling for viewer performance...")
        choice = np.random.choice(len(xyz), 1000000, replace=False)
        xyz = xyz[choice]
        colors = colors[choice]

    # [Viser] 점군 추가
    server.add_point_cloud(
        "/colored_instances",
        points=xyz,
        colors=colors,
        point_size=0.01
    )
    
    # 2. Bounding Box 로드 및 시각화 (JSON이 제공된 경우)
    if args.json_path and os.path.exists(args.json_path):
        print(f"Loading BBoxes from {args.json_path}...")
        with open(args.json_path, 'r') as f:
            scene_nodes = json.load(f)
            
        # 박스 색상 (라벨별)
        label_colors = {}
        base_colors = get_random_colors(20) # 20개 정도 랜덤 컬러 미리 생성

        for i, node in enumerate(scene_nodes):
            label = node['label']
            if label not in label_colors:
                label_colors[label] = base_colors[len(label_colors) % 20]
            
            # AABB 계산 (Min/Max -> Center/Dims)
            min_pt = np.array(node['bbox_min'])
            max_pt = np.array(node['bbox_max'])
            center = (min_pt + max_pt) / 2
            dimensions = max_pt - min_pt
            
            # [Viser] 박스 추가 (가장 단순하고 안전한 API 사용)
            server.add_box(
                name=f"/objects/{node['id']}_{label}_box",
                position=center,
                dimensions=dimensions,
                color=label_colors[label],
                visible=True
                # wireframe=True 옵션이 있다면 좋지만, 에러가 나면 solid box로 보임
            )
            
            # [Viser] 라벨 추가
            server.add_label(
                name=f"/objects/{node['id']}_{label}_text",
                text=f"{label} ({node['id']})",
                position=max_pt
            )
        print("BBoxes added successfully.")
    else:
        print("No JSON path provided or file not found. Skipping BBox visualization.")
    
    print("Visualization Ready! Open the URL provided above.")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Closing...")

if __name__ == "__main__":
    main()