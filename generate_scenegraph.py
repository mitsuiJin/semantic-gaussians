import torch
import numpy as np
import os
import argparse
import gc  # 가비지 컬렉터 추가
from plyfile import PlyData
from sklearn.cluster import DBSCAN
import json
from tqdm import tqdm

# Scannet 라벨 (scannet_constants.py 내용 반영 + 사용자 추가 라벨)
# 주의: view_viser.py 등에서 사용한 라벨 순서와 정확히 일치해야 합니다.
LABELS = (
    "wall", "floor", "bed", "chair", "table", "door", "cabinet", "curtain", "clothes"
    #"window", 
    # 여기에 추가했던 라벨들을 적어주세요 (dumbell, radiator 등)
    #"dumbell", "radiator", "trash can" 
)

# [설정] 클러스터링 시 한 클래스당 최대 점 개수 제한 (메모리 보호)
MAX_POINTS_PER_CLASS = 200000

def load_ply_xyz(path):
    """ply 파일에서 XYZ 좌표만 로드"""
    plydata = PlyData.read(path)
    xyz = np.stack((plydata['vertex']['x'], 
                    plydata['vertex']['y'], 
                    plydata['vertex']['z']), axis=1)
    return xyz

def load_semantic_features(path):
    """Fusion 결과(.pt)에서 피처 로드"""
    data = torch.load(path, map_location='cpu')
    return data['feat'].detach().cpu().numpy() # (N, 512)

def generate_scene_graph(ply_path, fusion_path, eps=0.1, min_samples=10):
    print(f"Loading Geometry from: {ply_path}")
    xyz = load_ply_xyz(ply_path)
    
    print(f"Loading Semantics from: {fusion_path}")
    feats = load_semantic_features(fusion_path)
    
    # 1. 텍스트 임베딩과 유사도 계산하여 라벨 결정 (여기서는 간단히 LSeg 피처 자체 클러스터링 대신 Argmax 가정을 사용)
    # 실제로는 LSeg Text Embedding과 내적을 해야 정확한 라벨 문자열을 알 수 있습니다.
    # 하지만 여기서는 '같은 특징을 가진 점들'을 그룹화하는 것이 목적이므로,
    # 텍스트 임베딩 매칭 코드는 생략하고, 
    # **Fusion Feature 자체를 이용하거나, view_viser 처럼 텍스트 임베딩을 가져와야 합니다.**
    # 편의상 여기서는 코드 복잡도를 줄이기 위해 'view_viser' 로직을 일부 차용해야 합니다.
    # (아래 main 함수에서 텍스트 임베딩 생성 부분 추가)
    
    return xyz, feats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--fusion_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading Geometry from: {args.ply_path}")
    xyz = load_ply_xyz(args.ply_path)
    
    print(f"Loading Semantics from: {args.fusion_path}")
    point_feats = load_semantic_features(args.fusion_path)
    
    # 2D Model 로드 (LSeg)
    print("Loading 2D Model for Text Embeddings...")
    from model.lseg_predictor import LSeg
    # 경로가 없다면 기본값 사용, 있다면 해당 경로 사용
    ckpt_path = "./weights/lseg/demo_e200.ckpt"
    if not os.path.exists(ckpt_path):
        # 만약 weights 폴더 위치가 다르다면 이 부분을 수정하세요
        ckpt_path = "/data/shjin1019/repos/semantic-gaussians/weights/lseg/demo_e200.ckpt"
        
    model_2d = LSeg(ckpt_path)
    
    # 텍스트 임베딩 추출
    # [수정됨] detach().cpu() 적용
    text_features = model_2d.extract_text_feature(list(LABELS)).float().detach().cpu().numpy()
    
    print("Calculating Similarity...")
    # (N, 512) @ (512, C) -> (N, C)
    # 메모리 절약을 위해 chunk 단위로 계산하거나 float32 사용
    sim = np.dot(point_feats, text_features.T)
    point_labels = np.argmax(sim, axis=1)
    
    # [중요] 피처 벡터는 이제 필요 없으므로 메모리에서 해제
    del point_feats
    del text_features
    gc.collect() 
    
    scene_nodes = []
    global_id_counter = 0

    print("Clustering Instances...")
    unique_labels = np.unique(point_labels)
    
    for label_idx in tqdm(unique_labels):
        class_name = LABELS[label_idx]
        
        # 배경 제외
        # if class_name in ["wall", "floor", "ceiling"]:
        #     continue
            
        # 해당 클래스 마스크
        mask = (point_labels == label_idx)
        class_xyz = xyz[mask]
        class_scores = sim[mask, label_idx] # 해당 클래스에 대한 점수만 가져옴
        
        num_points = len(class_xyz)
        if num_points < 50: 
            continue
            
        # === [핵심 수정] 다운샘플링 ===
        # 점이 너무 많으면 DBSCAN이 OOM을 일으키므로 랜덤 샘플링
        if num_points > MAX_POINTS_PER_CLASS:
            indices = np.random.choice(num_points, MAX_POINTS_PER_CLASS, replace=False)
            clustering_xyz = class_xyz[indices]
            clustering_scores = class_scores[indices]
        else:
            clustering_xyz = class_xyz
            clustering_scores = class_scores
            
        # DBSCAN 수행
        try:
            clustering = DBSCAN(eps=0.1, min_samples=1000, n_jobs=-1).fit(clustering_xyz)
        except Exception as e:
            print(f"Skipping {class_name} due to error: {e}")
            continue
        
        cluster_ids = clustering.labels_
        unique_clusters = np.unique(cluster_ids)
        
        for cid in unique_clusters:
            if cid == -1: continue 
            
            instance_mask = (cluster_ids == cid)
            instance_points = clustering_xyz[instance_mask]
            instance_score_arr = clustering_scores[instance_mask]
            
            # [추가] 강력한 후처리 필터링
            # 1. 점 개수가 너무 적으면 버림 (예: 500개 미만)
            if len(instance_points) < 5000:
                continue
                
            # 2. 신뢰도가 너무 낮으면 버림 (선택 사항)
            # if np.mean(instance_score_arr) < 0.6:
            #     continue
            
            node = {
                "id": global_id_counter,
                "label": class_name,
                "score": float(np.mean(instance_score_arr)), 
                "centroid": instance_points.mean(axis=0).tolist(),
                "bbox_min": instance_points.min(axis=0).tolist(),
                "bbox_max": instance_points.max(axis=0).tolist(),
                "point_count": int(len(instance_points))
            }
            scene_nodes.append(node)
            global_id_counter += 1

    # 결과 저장
    output_json = "scene_graph_nodes_7k.json"
    with open(output_json, "w") as f:
        json.dump(scene_nodes, f, indent=4)
        
    print(f"Done! Found {len(scene_nodes)} objects. Saved to {output_json}")

if __name__ == "__main__":
    main()