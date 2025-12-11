# python generate_scenegraph.py \
#     --ply_path "/data/shjin1019/repos/semantic-gaussians/output/scene0073_03_rgb_train/point_cloud/iteration_7000/point_cloud.ply" \
#     --fusion_path "/data/shjin1019/repos/datasets/scannet_fused/scene0073_03_7k/0.pt"

import torch
import numpy as np
import os
import argparse
import gc
from plyfile import PlyData
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull  # [추가] 실제 모양 계산용
import json
from tqdm import tqdm

# [설정] Scannet 라벨 정의 (작은 객체 제외하고 주요 가구 위주로 구성)
LABELS = (
    "wall", "floor", "bed", "chair", "table", "door", "cabinet", "curtain", "clothes"
)

# [설정] 메모리 보호를 위한 클래스당 최대 점 개수 제한
MAX_POINTS_PER_CLASS = 200000

def load_ply_xyz(path):
    """PLY 파일에서 XYZ 좌표만 읽어옵니다."""
    plydata = PlyData.read(path)
    xyz = np.stack((plydata['vertex']['x'], 
                    plydata['vertex']['y'], 
                    plydata['vertex']['z']), axis=1)
    return xyz.astype(np.float32) # 메모리 절약을 위해 float32 변환

def load_semantic_features(path):
    """Fusion 결과(.pt)에서 시맨틱 특징 벡터를 로드합니다."""
    data = torch.load(path, map_location='cpu')
    return data['feat'].detach().cpu().numpy().astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="Path to point_cloud.ply")
    parser.add_argument("--fusion_path", type=str, required=True, help="Path to fusion .pt file")
    args = parser.parse_args()

    # 1. 데이터 로드
    print(f"Loading Geometry from: {args.ply_path}")
    xyz = load_ply_xyz(args.ply_path)
    
    print(f"Loading Semantics from: {args.fusion_path}")
    point_feats = load_semantic_features(args.fusion_path)
    
    # 2. 2D 모델(LSeg) 로드 및 텍스트 임베딩 추출
    print("Loading 2D Model for Text Embeddings...")
    from model.lseg_predictor import LSeg
    
    # 체크포인트 경로 확인 (없으면 기본 경로 사용)
    ckpt_path = "./weights/lseg/demo_e200.ckpt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/data/shjin1019/repos/semantic-gaussians/weights/lseg/demo_e200.ckpt"
        
    model_2d = LSeg(ckpt_path)
    
    # 텍스트 임베딩 추출 (Gradient 끊기 필수)
    text_features = model_2d.extract_text_feature(list(LABELS)).float().detach().cpu().numpy()
    
    # 3. 유사도 계산 및 라벨 할당
    print("Calculating Similarity...")
    # (점 개수, 512) x (512, 라벨 수) -> (점 개수, 라벨 수)
    sim = np.dot(point_feats, text_features.T)
    point_labels = np.argmax(sim, axis=1)
    
    # 메모리 확보를 위해 불필요한 변수 삭제
    del point_feats
    del text_features
    gc.collect() 
    
    # 4. 인스턴스 클러스터링 (DBSCAN)
    scene_nodes = []
    global_id_counter = 0
    
    # [추가됨] 전체 점에 대한 인스턴스 ID 맵 초기화 (-1: 배경/노이즈)
    num_total_points = len(xyz)
    global_instance_ids = np.full(num_total_points, -1, dtype=np.int32)

    print("Clustering Instances...")
    unique_labels = np.unique(point_labels)
    
    for label_idx in tqdm(unique_labels):
        class_name = LABELS[label_idx]
        
        # (선택 사항) 배경은 인스턴스 분할에서 제외하고 싶다면 주석 해제
        if class_name in ["wall", "floor"]: continue
            
        # 현재 클래스에 해당하는 점들의 인덱스 마스크
        mask = (point_labels == label_idx)
        
        # [중요] 원본 데이터에서의 인덱스를 기억해야 나중에 ID 맵에 기록할 수 있음
        original_indices = np.where(mask)[0] 
        class_xyz = xyz[mask]
        class_scores = sim[mask, label_idx]
        
        num_points = len(class_xyz)
        if num_points < 50: continue # 너무 적은 점은 무시
            
        # === 다운샘플링 ===
        # 점이 너무 많으면 OOM 방지를 위해 랜덤 샘플링하여 클러스터링 수행
        # 주의: 샘플링을 하면 ID 맵 시각화 시 샘플링되지 않은 점은 빈 구멍으로 보일 수 있음
        if num_points > MAX_POINTS_PER_CLASS:
            # 랜덤 인덱스 선택
            sample_indices = np.random.choice(num_points, MAX_POINTS_PER_CLASS, replace=False)
            clustering_xyz = class_xyz[sample_indices]
            clustering_scores = class_scores[sample_indices]
            # 샘플링된 점들의 원본 인덱스도 업데이트
            current_indices = original_indices[sample_indices]
        else:
            clustering_xyz = class_xyz
            clustering_scores = class_scores
            current_indices = original_indices
            
        # DBSCAN 수행
        try:
            # eps=0.1 (10cm), min_samples=1000 (노이즈 강하게 제거)
            clustering = DBSCAN(eps=0.1, min_samples=1000, n_jobs=-1).fit(clustering_xyz)
        except Exception as e:
            print(f"Skipping {class_name} due to error: {e}")
            continue
        
        cluster_ids = clustering.labels_
        unique_clusters = np.unique(cluster_ids)
        
        for cid in unique_clusters:
            if cid == -1: continue # 노이즈 점 무시
            
            # 현재 클러스터에 속하는 점들 골라내기
            instance_mask = (cluster_ids == cid)
            instance_points = clustering_xyz[instance_mask]
            instance_score_arr = clustering_scores[instance_mask]
            
            # [후처리] 점 개수가 너무 적으면 진짜 객체가 아닐 확률 높음
            if len(instance_points) < 9000:
                continue
                
            # [추가됨] ID 맵에 기록 (시각화용)
            # 현재 클러스터에 속한 점들의 '원본 인덱스' 위치에 ID 부여
            valid_indices = current_indices[instance_mask]
            global_instance_ids[valid_indices] = global_id_counter
            
            # [핵심 추가] 2D Footprint (Convex Hull) 계산
            # XY 평면 투영 점들
            points_2d = instance_points[:, :2]
            footprint = []
            try:
                # 점이 3개 이상이어야 Hull 계산 가능
                if len(points_2d) >= 3:
                    hull = ConvexHull(points_2d)
                    # Hull의 외곽선 좌표 순서대로 저장
                    footprint = points_2d[hull.vertices].tolist()
                else:
                    # 점이 너무 적으면 그냥 BBox 사용
                    min_xy = points_2d.min(axis=0)
                    max_xy = points_2d.max(axis=0)
                    footprint = [
                        [min_xy[0], min_xy[1]],
                        [max_xy[0], min_xy[1]],
                        [max_xy[0], max_xy[1]],
                        [min_xy[0], max_xy[1]]
                    ]
            except Exception:
                 # 일직선 등 예외 발생 시 BBox로 대체
                 min_xy = points_2d.min(axis=0)
                 max_xy = points_2d.max(axis=0)
                 footprint = [[min_xy[0], min_xy[1]], [max_xy[0], max_xy[1]]]

            node = {
                "id": global_id_counter,
                "label": class_name,
                "score": float(np.mean(instance_score_arr)), 
                "centroid": instance_points.mean(axis=0).tolist(),
                "bbox_min": instance_points.min(axis=0).tolist(),
                "bbox_max": instance_points.max(axis=0).tolist(),
                "point_count": int(len(instance_points)),
                "footprint": footprint  # [추가됨] 실제 바닥 모양 다각형
            }
            scene_nodes.append(node)
            global_id_counter += 1

    # 결과 저장
    output_json = "scene_graph_nodes_convexhull.json"
    output_npy = "instance_ids_convexhull.npy"
    
    with open(output_json, "w") as f:
        json.dump(scene_nodes, f, indent=4)
    np.save(output_npy, global_instance_ids)
        
    print(f"Done! Saved with Footprint data to {output_json}")

if __name__ == "__main__":
    main()