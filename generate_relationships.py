#python generate_relationships.py --input_json scene_graph_nodes_convexhull.json

import json
import numpy as np
import argparse
import math
from shapely.geometry import Polygon

# [설정] 관계의 목적어(Target)가 될 수 없는 배경 객체들
IGNORE_AS_TARGET = ["floor", "wall", "ceiling"]

# 관계 역관계 매핑 (중복 제거용)
INVERSE_MAP = {
    "left_of": "right_of",
    "right_of": "left_of",
    "front_of": "behind",
    "behind": "front_of",
    "on": "under",
    "under": "on",
    "near": "near"
}

# 우선순위 (높을수록 좋음)
PRIORITY_MAP = {
    "on": 3, 
    "under": 3, 
    "front_of": 2, 
    "behind": 2, 
    "left_of": 2, 
    "right_of": 2, 
    "near": 1
}

def get_bbox_dims(node):
    """BBox의 너비(W), 깊이(D), 높이(H) 및 대각선 길이(Scale) 반환"""
    min_p = np.array(node['bbox_min'])
    max_p = np.array(node['bbox_max'])
    dims = max_p - min_p # [x, y, z]
    scale = np.linalg.norm(dims) # 대각선 길이
    return dims, scale

def get_polygon(node):
    """Footprint 다각형 반환"""
    if 'footprint' in node and len(node['footprint']) > 2:
        return Polygon(node['footprint'])
    else:
        min_p, max_p = node['bbox_min'], node['bbox_max']
        return Polygon([(min_p[0], min_p[1]), (max_p[0], min_p[1]), (max_p[0], max_p[1]), (min_p[0], max_p[1])])

def calculate_direction_score(p1, p2):
    """
    두 점 사이의 각도를 계산하여 방향과 '정렬 점수(Alignment Score)' 반환
    점수가 높을수록 축(0, 90, 180, 270도)에 가까움
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    
    # 4방위 판정
    if -45 <= angle <= 45:
        pred = "right_of"
        ideal = 0
    elif 45 < angle <= 135:
        pred = "behind"
        ideal = 90
    elif -135 <= angle < -45:
        pred = "front_of"
        ideal = -90
    else:
        pred = "left_of"
        ideal = 180 if angle > 0 else -180
    
    # 정렬 점수: 이상적인 각도와의 차이가 작을수록 1.0에 가까움 (Cosine 유사도 활용)
    # diff가 0이면 score 1.0, diff가 45면 score 0.0 근처가 되도록 조정
    angle_diff = abs(angle - ideal)
    if angle_diff > 180: angle_diff = 360 - angle_diff
    
    # 45도 차이면 0점, 0도 차이면 1점 (선형 보간)
    align_score = max(0.0, 1.0 - (angle_diff / 45.0))
    
    return pred, align_score

def analyze_relationship_adaptive(subj, obj):
    """
    개선된 관계 분석 로직 (Adaptive Thresholding)
    """
    subj_c = np.array(subj['centroid'])
    obj_c = np.array(obj['centroid'])
    
    # 1. 크기(Scale) 계산
    subj_dims, subj_scale = get_bbox_dims(subj)
    obj_dims, obj_scale = get_bbox_dims(obj)
    avg_scale = (subj_scale + obj_scale) / 2.0
    
    # [피드백 1] 적응형 거리 Threshold
    dist = np.linalg.norm(subj_c - obj_c)
    dist_threshold = max(1.5, avg_scale * 2.5) # 최소 1.5m, 혹은 객체 크기의 2.5배
    
    if dist > dist_threshold:
        return None, 0.0

    # 2. 기하학적 겹침 분석 (IoU & IoS)
    subj_poly = get_polygon(subj)
    obj_poly = get_polygon(obj)
    if not subj_poly.is_valid: subj_poly = subj_poly.buffer(0)
    if not obj_poly.is_valid: obj_poly = obj_poly.buffer(0)
    
    intersection = subj_poly.intersection(obj_poly).area
    subj_area = subj_poly.area
    obj_area = obj_poly.area
    union_area = subj_area + obj_area - intersection
    
    iou = intersection / union_area if union_area > 0 else 0.0
    ios = intersection / subj_area if subj_area > 0 else 0.0 # Intersection over Subject

    # 3. [피드백 3] 정규화된 높이 분석
    # 주체와 객체의 높이(H)를 고려하여 허용 범위를 설정
    subj_min_z, subj_max_z = subj['bbox_min'][2], subj['bbox_max'][2]
    obj_min_z, obj_max_z = obj['bbox_min'][2], obj['bbox_max'][2]
    
    # Z-Gap: 주체 바닥 - 객체 천장
    z_gap_on = subj_min_z - obj_max_z 
    
    # 허용 오차: 절대값 20cm 혹은 객체 높이의 20% 중 큰 값 (유연함)
    tolerance_on = max(0.2, obj_dims[2] * 0.2) 
    
    # [ON 판단]
    # IoS가 높아야 함(작은게 큰거 위에 올라감) + Z-Gap이 허용오차 내에 있어야 함
    if ios > 0.3 and abs(z_gap_on) < tolerance_on:
        # 점수: 겹침 정도 + 높이 정확도
        score = ios * 10.0 + (1.0 - abs(z_gap_on)/tolerance_on)
        return "on", score

    # [UNDER 판단]
    z_gap_under = obj_min_z - subj_max_z
    tolerance_under = max(0.2, subj_dims[2] * 0.2)
    
    if ios > 0.3 and abs(z_gap_under) < tolerance_under:
        score = ios * 10.0 + (1.0 - abs(z_gap_under)/tolerance_under)
        return "under", score

    # 4. [피드백 2] 방향성 판단 (연속 각도 기반)
    # 겹침이 적을 때만 방향성 부여 (IoU 기준)
    if iou < 0.1:
        pred, align_score = calculate_direction_score(obj_c, subj_c)
        # 점수: 정렬도(0~1) * 5.0
        return pred, align_score * 5.0

    # 5. [피드백 5] Near 판단 (정규화된 거리)
    # 거리가 가까울수록 1.0, 멀수록 0.0 (적응형 임계값 기준)
    normalized_dist_score = max(0.0, 1.0 - (dist / dist_threshold))
    
    if normalized_dist_score > 0.2: # 하위 20%는 버림
        return "near", normalized_dist_score # 최대 1.0

    return None, 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="scene_graph_nodes_convexhull.json")
    parser.add_argument("--output_json", type=str, default="scene_graph_final.json")
    args = parser.parse_args()

    print(f"Loading nodes from {args.input_json}...")
    with open(args.input_json, 'r') as f:
        nodes = json.load(f)
        
    all_candidates = []
    
    # Step 1: 모든 관계 후보 생성
    for subj in nodes:
        for obj in nodes:
            if subj['id'] == obj['id']: continue
            if obj['label'] in IGNORE_AS_TARGET: continue 
            
            pred, score = analyze_relationship_adaptive(subj, obj)
            
            if pred:
                all_candidates.append({
                    "subject_id": subj['id'],
                    "subject_label": subj['label'],
                    "predicate": pred,
                    "object_id": obj['id'],
                    "object_label": obj['label'],
                    "score": score
                })

    # Step 2: 정렬 (우선순위 -> 점수)
    all_candidates.sort(key=lambda x: (PRIORITY_MAP.get(x['predicate'], 0), x['score']), reverse=True)
    
    # Step 3: [피드백 6] 스마트 중복 제거
    # 규칙: 두 객체 A, B 사이에는 "가장 강력한 관계 하나"만 남긴다.
    # 단, 방향성(left)과 근접(near)처럼 정보량이 다른 경우, 상위 정보를 선택.
    
    final_relationships = []
    processed_pairs = set() # (A, B) 혹은 (B, A)가 처리되었는지 확인
    
    for rel in all_candidates:
        s_id = rel['subject_id']
        o_id = rel['object_id']
        
        # 정렬된 키를 사용하여 A->B와 B->A를 같은 쌍으로 취급
        pair_key = tuple(sorted((s_id, o_id)))
        
        if pair_key in processed_pairs:
            continue # 이미 더 높은 점수/우선순위의 관계가 이 쌍에 존재함
            
        # 선택된 관계 저장
        final_rel = {k: v for k, v in rel.items() if k != 'score'}
        final_relationships.append(final_rel)
        processed_pairs.add(pair_key)

    # 결과 저장
    output_data = {
        "nodes": nodes,
        "relationships": final_relationships
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f" Generated {len(final_relationships)} adaptive relationships.")
    print(f" Eliminated redundant edges using Smart Deduplication.")
    print(f" Saved to {args.output_json}")

if __name__ == "__main__":
    main()