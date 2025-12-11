import json
import argparse
import numpy as np

def get_location_description(centroid, room_min, room_max):
    """
    객체의 중심점을 방 전체 크기 대비 상대적 위치(예: Back-Left)로 변환
    """
    x, y = centroid[0], centroid[1]
    min_x, min_y = room_min[0], room_min[1]
    max_x, max_y = room_max[0], room_max[1]
    
    # 0.0 ~ 1.0 정규화
    norm_x = (x - min_x) / (max_x - min_x + 1e-6)
    norm_y = (y - min_y) / (max_y - min_y + 1e-6)
    
    # X축 (좌우)
    if norm_x < 0.33: x_desc = "Left"
    elif norm_x < 0.66: x_desc = "Center"
    else: x_desc = "Right"
    
    # Y축 (앞뒤 - 깊이)
    if norm_y < 0.33: y_desc = "Front" # 입구 쪽
    elif norm_y < 0.66: y_desc = "Center"
    else: y_desc = "Back" # 안쪽
    
    if x_desc == "Center" and y_desc == "Center":
        return "Center of the room"
    return f"{y_desc}-{x_desc} side"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="scene_graph_final.json")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    nodes = data['nodes']
    rels = data['relationships']

    # 1. 방 전체 크기 계산 (Room Extents)
    all_min = []
    all_max = []
    for n in nodes:
        all_min.append(n['bbox_min'])
        all_max.append(n['bbox_max'])
    
    room_min = np.min(all_min, axis=0)
    room_max = np.max(all_max, axis=0)
    room_dims = room_max - room_min
    
    # 2. 객체 상세 정보 생성 (Rich Attributes)
    objects_desc = []
    for n in nodes:
        # 크기 계산 (W x D x H)
        dims = np.array(n['bbox_max']) - np.array(n['bbox_min'])
        width, depth, height = dims[0], dims[1], dims[2]
        
        # 위치 설명 (좌표 -> 언어)
        loc_desc = get_location_description(n['centroid'], room_min, room_max)
        
        # 부피/면적에 따른 묘사 힌트
        size_hint = ""
        if height > 1.5: size_hint = "(Tall)"
        elif height < 0.5: size_hint = "(Low/Short)"
        
        obj_str = (
            f"- **{n['label']} (ID: {n['id']})**\n"
            f"  - Dimensions: {width:.2f}m(W) x {depth:.2f}m(D) x {height:.2f}m(H) {size_hint}\n"
            f"  - Location: Placed at the {loc_desc} of the room."
        )
        objects_desc.append(obj_str)

    # 3. 관계 정보 그룹화 (Subject 중심)
    rel_map = {}
    for r in rels:
        subj = f"{r['subject_label']}({r['subject_id']})"
        pred = r['predicate'].upper()
        obj = f"{r['object_label']}({r['object_id']})"
        
        if subj not in rel_map:
            rel_map[subj] = []
        rel_map[subj].append(f"{pred} -> {obj}")

    relations_desc = []
    for subj, rel_list in rel_map.items():
        joined = ", ".join(rel_list)
        relations_desc.append(f"- {subj} is {joined}.")

    # 4. 최종 프롬프트 조합
    prompt = f"""
# SYSTEM ROLE
You are an expert Spatial AI capable of reconstructing 3D scenes from data.
Your task is to describe the room layout vividly and logically, as if explaining it to a visually impaired person or a potential buyer.

# SCENE METADATA
- **Room Size**: Approx. {room_dims[0]:.1f}m (Width) x {room_dims[1]:.1f}m (Depth) x {room_dims[2]:.1f}m (Height)
- **Total Objects**: {len(nodes)}

# DETAILED OBJECT LIST (Geometry & Location)
{chr(10).join(objects_desc)}

# SPATIAL RELATIONSHIPS (Topology)
{chr(10).join(relations_desc)}

# INSTRUCTION
Based on the data above, generate a natural language description of this room.
1. **Start with an overview**: Mention the probable room type (e.g., Bedroom, Living room) based on the furniture.
2. **Describe the layout flow**: Start from the likely entrance (Front side) and move inwards (Back side).
3. **Use the specific attributes**: Mention if objects are tall, large, or placed in corners (using the Location data).
4. **Group nearby objects**: Instead of listing relations one by one, combine them (e.g., "A table with a chair next to it").
5. **Reasoning**: If something seems unusual (e.g., clothes on a cabinet), mention it naturally.

# YOUR DESCRIPTION
"""
    
    print(prompt)

if __name__ == "__main__":
    main()