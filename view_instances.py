# python view_instances.py \
#   --ply_path "/data/shjin1019/repos/semantic-gaussians/output/scene0073_03_rgb_train/point_cloud/iteration_7000/point_cloud.ply" \
#   --npy_path "instance_ids_convexhull.npy" \
#   --json_path "scene_graph_nodes_convexhull.json"


import viser
import numpy as np
import argparse
import os
import json
import time
from plyfile import PlyData
from scipy.spatial import ConvexHull

def load_ply_xyz(path):
    print(f"Loading PLY from {path}...")
    plydata = PlyData.read(path)
    xyz = np.stack((plydata['vertex']['x'], 
                    plydata['vertex']['y'], 
                    plydata['vertex']['z']), axis=1)
    return xyz

def get_random_colors(n):
    np.random.seed(42)
    return np.random.randint(50, 255, size=(n, 3)) # 너무 어두운 색 제외

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="Path to point_cloud.ply")
    parser.add_argument("--npy_path", type=str, required=True, help="Path to instance_ids.npy")
    parser.add_argument("--json_path", type=str, default=None, help="Path to scene_graph_nodes.json")
    args = parser.parse_args()

    # Viser 서버 시작
    server = viser.ViserServer()
    print("Viser Server started!")

    # 1. 원본 데이터 로드 (Convex Hull 계산용)
    xyz_full = load_ply_xyz(args.ply_path)
    instance_ids = np.load(args.npy_path)
    
    print(f"Total points: {len(xyz_full)}")
    print(f"Clustered points: {np.sum(instance_ids != -1)}")

    # 색상 배열 생성
    colors_full = np.full_like(xyz_full, 200, dtype=np.uint8) 
    
    max_id = instance_ids.max()
    palette = None
    
    if max_id >= 0:
        palette = get_random_colors(max_id + 1)
        mask = (instance_ids != -1)
        colors_full[mask] = palette[instance_ids[mask]]

    # 2. 시각화용 다운샘플링
    xyz_vis = xyz_full
    colors_vis = colors_full
    
    MAX_SHOW_POINTS = 1000000
    if len(xyz_full) > MAX_SHOW_POINTS:
        print(f"Downsampling for viewer performance ({len(xyz_full)} -> {MAX_SHOW_POINTS})...")
        choice = np.random.choice(len(xyz_full), MAX_SHOW_POINTS, replace=False)
        xyz_vis = xyz_full[choice]
        colors_vis = colors_full[choice]

    # [Viser] 점군 추가
    server.scene.add_point_cloud(
        "/colored_instances",
        points=xyz_vis,
        colors=colors_vis,
        point_size=0.01
    )
    
    # 3. 3D Convex Hull (Real Shape) 계산 및 시각화
    if args.json_path and os.path.exists(args.json_path):
        print("Calculating 3D Convex Hulls from FULL point cloud...")
        with open(args.json_path, 'r') as f:
            scene_nodes = json.load(f)
            
        for node in scene_nodes:
            node_id = node['id']
            label = node['label']
            
            # 해당 인스턴스의 점들 추출 (다운샘플링 안 된 'xyz_full' 사용)
            instance_mask = (instance_ids == node_id)
            points_3d = xyz_full[instance_mask]
            
            if len(points_3d) < 4:
                continue
            
            # 인스턴스 색상 가져오기
            if palette is not None:
                instance_color = palette[node_id]
            else:
                instance_color = np.array([255, 255, 255])

            try:
                # 3D Convex Hull 계산
                hull = ConvexHull(points_3d)
                
                # Hull의 엣지(선분) 추출
                segments = []
                for simplex in hull.simplices:
                    # 삼각형의 세 변을 선분으로 추가
                    p0 = points_3d[simplex[0]]
                    p1 = points_3d[simplex[1]]
                    p2 = points_3d[simplex[2]]
                    
                    # (Start, End) 쌍으로 저장
                    segments.append([p0, p1])
                    segments.append([p1, p2])
                    segments.append([p2, p0])
                
                segments = np.array(segments) # Shape: (N_edges, 2, 3)
                
                # [수정됨] 단일 색상 전달 (배열 생성 X)
                # colors 인자에 (3,) 형태를 넣으면 모든 선분에 적용됨
                server.scene.add_line_segments(
                    name=f"/objects/{node_id}_{label}_hull",
                    points=segments,
                    colors=instance_color, 
                    line_width=2.0, 
                )
                
                # 라벨 추가
                center = points_3d.mean(axis=0)
                server.scene.add_label(
                    name=f"/objects/{node_id}_{label}_text",
                    text=f"{label}",
                    position=center
                )
                
            except Exception as e:
                # 에러 발생 시 원인 출력 (디버깅용)
                print(f"[Error] Failed to draw hull for {label} ({node_id}): {e}")
                continue
            
        print("3D Convex Hulls added successfully.")
    else:
        print("No JSON path provided. Skipping Hull visualization.")
    
    print("Visualization Ready! Open the URL provided above.")
    
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Closing...")

if __name__ == "__main__":
    main()