import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="scene_graph_sparse.json")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 색상 팔레트 (파스텔톤)
    color_map = {
        "bed": "#8dd3c7", "chair": "#ffffb3", "table": "#bebada", 
        "door": "#fb8072", "cabinet": "#80b1d3", "curtain": "#fdb462", 
        "clothes": "#b3de69", "floor": "#d9d9d9", "wall": "#bc80bd"
    }

    # 1. 객체 그리기 (Footprint & Centroid)
    centroids = {} # 화살표를 그리기 위해 좌표 저장

    for node in data['nodes']:
        nid = node['id']
        label = node['label']
        cx, cy = node['centroid'][0], node['centroid'][1]
        centroids[nid] = (cx, cy)
        
        color = color_map.get(label, "#cccccc")

        # Footprint(다각형) 그리기
        if 'footprint' in node and len(node['footprint']) > 2:
            # 2D 좌표 리스트
            poly_coords = node['footprint']
            # Matplotlib Polygon 생성
            poly = patches.Polygon(poly_coords, closed=True, facecolor=color, edgecolor='black', alpha=0.6)
            ax.add_patch(poly)
        else:
            # Footprint 없으면 BBox로 대체
            min_p = node['bbox_min']
            max_p = node['bbox_max']
            rect = patches.Rectangle(
                (min_p[0], min_p[1]), 
                max_p[0] - min_p[0], 
                max_p[1] - min_p[1],
                facecolor=color, edgecolor='black', alpha=0.6
            )
            ax.add_patch(rect)

        # 중심점에 ID와 라벨 텍스트 표시
        ax.text(cx, cy, f"{label}\n({nid})", ha='center', va='center', fontsize=9, fontweight='bold', color='black')
        
        # 중심점 점 찍기
        ax.plot(cx, cy, 'o', color='black', markersize=3)

    # 2. 관계 화살표 그리기
    # 너무 복잡하면 화살표는 일부만 그리거나 옵션으로 뺄 수 있음
    for rel in data['relationships']:
        s_id = rel['subject_id']
        o_id = rel['object_id']
        pred = rel['predicate']
        
        # 시작점, 끝점
        start = centroids[s_id]
        end = centroids[o_id]
        
        # 화살표 그리기 (annotate 사용)
        # 겹침 방지를 위해 화살표를 약간 휘게 만듦 (connectionstyle)
        ax.annotate(
            "",
            xy=end, xycoords='data',
            xytext=start, textcoords='data',
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, connectionstyle="arc3,rad=0.2", alpha=0.7)
        )
        
        # 화살표 중간에 관계 텍스트 표시
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # 텍스트가 겹칠 수 있으므로 약간 이동 (간단한 휴리스틱)
        offset = 0.1 if s_id % 2 == 0 else -0.1
        
        ax.text(mid_x + offset, mid_y + offset, pred, fontsize=8, color='blue', alpha=0.8, 
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # 그래프 설정
    ax.set_aspect('equal') # 비율 유지 (찌그러짐 방지)
    ax.set_title("3D Scene Graph - Physical Layout Map", fontsize=15)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    # 축 범위 자동 설정 (약간의 여백 포함)
    all_x = [p[0] for p in centroids.values()]
    all_y = [p[1] for p in centroids.values()]
    margin = 0.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 저장 및 출력
    plt.savefig("scene_graph_map.png", dpi=150)
    print("Saved physical map visualization to 'scene_graph_map.png'")
    plt.show()

if __name__ == "__main__":
    main()