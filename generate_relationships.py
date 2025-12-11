#python generate_relationships.py --input_json scene_graph_nodes_7k_2.json

import json
import numpy as np
import argparse
from itertools import permutations

def compute_iou_2d(bbox1, bbox2):
    """XY 평면에서의 겹침 정도(Intersection over Union) 계산 (for 'on' relationship)"""
    # bbox = [min_x, min_y, min_z, max_x, max_y, max_z]
    
    # 1. XY 평면 좌표 추출
    x_min1, y_min1 = bbox1[0], bbox1[1]
    x_max1, y_max1 = bbox1[3], bbox1[4]
    
    x_min2, y_min2 = bbox2[0], bbox2[1]
    x_max2, y_max2 = bbox2[3], bbox2[4]
    
    # 2. 교차 영역 계산
    inter_x_min = max(x_min1, x_min2)
    inter_y_min = max(y_min1, y_min2)
    inter_x_max = min(x_max1, x_max2)
    inter_y_max = min(y_max1, y_max2)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
        
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 3. 합집합 영역 계산
    area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def check_proximity(center1, center2, threshold=1.5):
    """두 객체 간의 거리(Euclidean Distance) 확인"""
    dist = np.linalg.norm(np.array(center1) - np.array(center2))
    return dist < threshold, dist

def check_vertical(bbox_subject, bbox_object):
    """
    Subject가 Object 위에 있는지 확인 ('on')
    조건 1: Subject의 바닥면이 Object의 윗면과 높이가 비슷하거나 높아야 함
    조건 2: XY 평면상에서 겹쳐야 함 (수직으로 위에 있음)
    """
    # bbox: [min_x, min_y, min_z, max_x, max_y, max_z]
    subj_min_z = bbox_subject[2]
    obj_max_z = bbox_object[5]
    
    # 높이 차이 (Subject가 Object 위에 살짝 떠있거나 닿아있는 경우)
    # ScanNet 데이터 특성상 약간의 오차 허용 (예: -0.2m ~ +0.5m)
    z_diff = subj_min_z - obj_max_z
    
    is_above = -0.2 < z_diff < 0.5
    
    # XY 평면 겹침 확인 (IoU가 0보다 크면 겹침)
    iou_xy = compute_iou_2d(bbox_subject, bbox_object)
    is_xy_overlap = iou_xy > 0.0
    
    return is_above and is_xy_overlap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="scene_graph_nodes.json", help="Input nodes JSON")
    parser.add_argument("--output_json", type=str, default="scene_graph_final.json", help="Output full graph JSON")
    args = parser.parse_args()

    print(f"Loading nodes from {args.input_json}...")
    with open(args.input_json, 'r') as f:
        nodes = json.load(f)
        
    relationships = []
    
    # 모든 노드 쌍에 대해 관계 검사 (Permutations: 순서 중요 A->B != B->A)
    # 자기 자신과의 관계는 제외
    for subj, obj in permutations(nodes, 2):
        
        # BBox 리스트 평탄화: [min_x, min_y, min_z, max_x, max_y, max_z]
        subj_bbox = subj['bbox_min'] + subj['bbox_max']
        obj_bbox = obj['bbox_min'] + obj['bbox_max']
        
        # 1. Proximity (Near)
        is_near, dist = check_proximity(subj['centroid'], obj['centroid'], threshold=1.2) # 1.2m 이내
        
        # 2. Vertical (On / Supported by)
        is_on = check_vertical(subj_bbox, obj_bbox)
        
        # === 관계 생성 ===
        
        # Rule 1: 'on' 관계가 성립하면 'near'보다 우선순위를 둠
        if is_on:
            rel = {
                "subject_id": subj['id'],
                "subject_label": subj['label'],
                "predicate": "on",
                "object_id": obj['id'],
                "object_label": obj['label']
            }
            relationships.append(rel)
            
        # Rule 2: 'on'이 아니면서 가까우면 'near'
        elif is_near:
            rel = {
                "subject_id": subj['id'],
                "subject_label": subj['label'],
                "predicate": "near",
                "object_id": obj['id'],
                "object_label": obj['label']
            }
            relationships.append(rel)

    # 최종 결과 구조
    full_scene_graph = {
        "nodes": nodes,
        "relationships": relationships
    }
    
    # 저장
    with open(args.output_json, 'w') as f:
        json.dump(full_scene_graph, f, indent=4)
        
    print(f"Scene Graph Generated!")
    print(f"Nodes: {len(nodes)}")
    print(f"Relationships: {len(relationships)}")
    print(f"Saved to {args.output_json}")

if __name__ == "__main__":
    main()