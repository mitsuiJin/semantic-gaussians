#python generate_relationships.py --input_json scene_graph_nodes_convexhull.json

import json
import numpy as np
import argparse
import math
from shapely.geometry import Polygon

IGNORE_AS_TARGET = ["floor", "wall", "ceiling"]

# 관계 우선순위
PRIORITY_MAP = {
    "on": 5,        # 물리적 접촉 (가장 중요)
    "under": 5, 
    "front_of": 2, 
    "behind": 2, 
    "left_of": 2, 
    "right_of": 2, 
    "near": 1
}

def get_polygon(node):
    if 'footprint' in node and len(node['footprint']) > 2:
        return Polygon(node['footprint'])
    else:
        min_p, max_p = node['bbox_min'], node['bbox_max']
        return Polygon([(min_p[0], min_p[1]), (max_p[0], min_p[1]), (max_p[0], max_p[1]), (min_p[0], max_p[1])])

def calculate_direction_score(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    
    if -45 <= angle <= 45: pred = "right_of"; ideal = 0
    elif 45 < angle <= 135: pred = "behind"; ideal = 90
    elif -135 <= angle < -45: pred = "front_of"; ideal = -90
    else: pred = "left_of"; ideal = 180 if angle > 0 else -180
    
    angle_diff = abs(angle - ideal)
    if angle_diff > 180: angle_diff = 360 - angle_diff
    align_score = max(0.0, 1.0 - (angle_diff / 45.0)) # 0~1 점수
    
    return pred, align_score

def analyze_relationship_sparse(subj, obj):
    subj_c = np.array(subj['centroid'])
    obj_c = np.array(obj['centroid'])
    
    # 1. 거리 계산
    dist = np.linalg.norm(subj_c - obj_c)
    
    # [핵심] 거리 컷오프를 엄격하게 적용 (on/under가 아니면 1.5m 넘어가면 무시)
    # 단, 객체가 매우 큰 경우(침대 등)는 예외적으로 허용
    dims = np.array(subj['bbox_max']) - np.array(subj['bbox_min'])
    max_dim = np.max(dims)
    dist_limit = max(1.5, max_dim * 1.5) 

    if dist > dist_limit: return None, 0.0

    # 2. Vertical (ON/UNDER) - 거리 상관없이 물리적 겹침이면 인정
    subj_min_z, subj_max_z = subj['bbox_min'][2], subj['bbox_max'][2]
    obj_max_z = obj['bbox_max'][2]
    
    subj_poly = get_polygon(subj)
    obj_poly = get_polygon(obj)
    if not subj_poly.is_valid: subj_poly = subj_poly.buffer(0)
    if not obj_poly.is_valid: obj_poly = obj_poly.buffer(0)
    
    inter_area = subj_poly.intersection(obj_poly).area
    subj_area = subj_poly.area
    ios = inter_area / subj_area if subj_area > 0 else 0.0
    
    if ios > 0.3 and -0.2 < (subj_min_z - obj_max_z) < 0.5:
        # 가중치: 겹침 정도가 클수록 높음
        return "on", 10.0 + ios 

    # 3. Horizontal & Near (거리가 멀면 점수 대폭 깎음)
    if ios < 0.1:
        # 거리 페널티: 거리가 0에 가까우면 1.0, limit에 가까우면 0.0
        dist_score = max(0.0, 1.0 - (dist / dist_limit))
        
        # 방향성 계산
        pred, align_score = calculate_direction_score(obj_c, subj_c)
        
        # 최종 점수 = 방향정확도 * 거리점수 * 5.0
        # 거리가 멀면 방향이 정확해도 점수가 낮아져서 탈락됨
        final_score = align_score * dist_score * 5.0
        
        if final_score > 1.0: # 일정 점수 이상만 인정 (노이즈 제거)
            return pred, final_score
        elif dist_score > 0.5: # 방향은 애매하지만 매우 가까우면 near
            return "near", dist_score * 2.0

    return None, 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="scene_graph_nodes_convexhull.json")
    parser.add_argument("--output_json", type=str, default="scene_graph_sparse.json")
    args = parser.parse_args()

    print(f"Loading nodes from {args.input_json}...")
    with open(args.input_json, 'r') as f:
        nodes = json.load(f)
        
    final_relationships = []
    
    # [Step 1] 각 Subject별로 가장 중요한 관계 Top-K개만 선택
    for subj in nodes:
        candidates = []
        
        for obj in nodes:
            if subj['id'] == obj['id']: continue
            if obj['label'] in IGNORE_AS_TARGET: continue
            
            pred, score = analyze_relationship_sparse(subj, obj)
            if pred:
                candidates.append({
                    "subject_id": subj['id'],
                    "subject_label": subj['label'],
                    "predicate": pred,
                    "object_id": obj['id'],
                    "object_label": obj['label'],
                    "score": score
                })
        
        if not candidates: continue
        
        # [핵심 필터링]
        # 1. 'on' 관계는 무조건 포함 (최대 1개)
        on_rels = [r for r in candidates if r['predicate'] == "on"]
        if on_rels:
            best_on = max(on_rels, key=lambda x: x['score'])
            final_relationships.append(best_on)
            # on이 있으면 다른 관계는 굳이 필요 없음 (Exclusivity)
            continue 

        # 2. 나머지 관계는 점수순 정렬 후 상위 2개(Top-2)만 유지
        # -> 이렇게 하면 'table'이 10개 물체와 관계를 맺는 것을 방지함
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 상위 2개만 선택하되, 중복 방향 방지 (예: 왼쪽1, 왼쪽2 -> 왼쪽1만)
        used_directions = set()
        count = 0
        for r in candidates:
            if r['predicate'] in used_directions: continue
            
            final_relationships.append(r)
            used_directions.add(r['predicate'])
            count += 1
            if count >= 2: break # 최대 2개까지만 연결

    # 결과 저장
    output_data = {
        "nodes": nodes,
        "relationships": final_relationships
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Generated Sparse Scene Graph: {len(final_relationships)} edges.")
    print(f"Applied: Distance Decay & Top-2 Neighbor Limit.")
    print(f"Saved to {args.output_json}")

if __name__ == "__main__":
    main()