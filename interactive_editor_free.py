import json
import argparse
import numpy as np
import sys
from plyfile import PlyData, PlyElement

def generate_prompt_for_manual_llm(data):
    """LLM에게 줄 프롬프트 텍스트 생성"""
    nodes_info = [f"- {n['label']} (ID: {n['id']})" for n in data['nodes']]
    context = [f"- {r['subject_label']}({r['subject_id']}) is {r['predicate']} {r['object_label']}({r['object_id']})" for r in data['relationships']]
    
    return f"""
You are a Spatial AI. Analyze this Scene Graph and suggest edits.

[Objects]
{chr(10).join(nodes_info)}

[Relationships]
{chr(10).join(context)}

[Task]
Suggest 2-3 edits (REMOVE clutter, MOVE obstacles).
Return ONLY a valid JSON list like:
[
    {{"id": 0, "type": "REMOVE", "target_id": 8, "desc": "Clean up clothes."}},
    {{"id": 1, "type": "MOVE", "target_id": 2, "params": [0.5, 0.0, 0.0], "desc": "Move chair blocking door."}}
]
"""

def get_rule_based_recommendations(data):
    """
    [대안] LLM 없이 코드로 간단한 규칙을 적용해 추천 목록 생성
    규칙 1: 'clothes'나 'trash'가 어디 위에 있으면 -> REMOVE
    규칙 2: 'door' 앞에 무언가 있으면 -> MOVE
    """
    recommendations = []
    count = 0
    
    # 1. Clutter Removal Rule
    clutter_items = ["clothes", "trash", "pillow", "box"]
    for rel in data['relationships']:
        if rel['subject_label'] in clutter_items and rel['predicate'] == "on":
            recommendations.append({
                "id": count,
                "type": "REMOVE",
                "target_id": rel['subject_id'],
                "desc": f"Rule: Remove {rel['subject_label']} found on {rel['object_label']} (Clutter)"
            })
            count += 1
            
    # 2. Obstacle Clearing Rule
    for rel in data['relationships']:
        if rel['object_label'] == "door" and rel['predicate'] in ["front_of", "near", "left_of", "right_of"]:
            # 문 근처에 있는 의자나 테이블 등
            if rel['subject_label'] in ["chair", "table", "cabinet"]:
                recommendations.append({
                    "id": count,
                    "type": "MOVE",
                    "target_id": rel['subject_id'],
                    "params": [0.5, 0.0, 0.0], # 기본적으로 X축으로 0.5m 이동
                    "desc": f"Rule: Move {rel['subject_label']} away from {rel['object_label']} (Obstacle)"
                })
                count += 1
                
    return recommendations

def apply_edits(input_ply, instance_ids_path, output_ply, selected_commands):
    """PLY 파일 수정 (기존과 동일)"""
    print(f"\n📂 Loading {input_ply}...")
    plydata = PlyData.read(input_ply)
    vertex = plydata['vertex']
    
    x = np.array(vertex['x']).copy()
    y = np.array(vertex['y']).copy()
    z = np.array(vertex['z']).copy()
    opacity = np.array(vertex['opacity']).copy()
    
    print(f"📂 Loading {instance_ids_path}...")
    instance_ids = np.load(instance_ids_path)

    for cmd in selected_commands:
        tid = cmd['target_id']
        action = cmd['type']
        mask = (instance_ids == tid)
        
        if np.sum(mask) == 0:
            print(f"⚠️ Target ID {tid} empty.")
            continue
            
        print(f"🔨 Action: {action} ID {tid} ({cmd['desc']})")
        if action == "REMOVE":
            opacity[mask] = -10000.0
        elif action == "MOVE":
            p = cmd.get('params', [0.5, 0.0, 0.0])
            x[mask] += p[0]; y[mask] += p[1]; z[mask] += p[2]

    vertex.data['x'] = x; vertex.data['y'] = y; vertex.data['z'] = z
    vertex.data['opacity'] = opacity
    PlyData([plydata.elements[0]]).write(output_ply)
    print(f"✅ Done! Saved to {output_ply}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_graph", default="scene_graph_sparse.json")
    parser.add_argument("--input_ply", default="point_cloud.ply")
    parser.add_argument("--instance_ids", default="instance_ids.npy")
    parser.add_argument("--output_ply", default="scene_edited_free.ply")
    args = parser.parse_args()

    with open(args.scene_graph, 'r') as f:
        data = json.load(f)

    print("="*50)
    print(" 🛠️  Scene Auto-Editor (Free Mode)")
    print("="*50)
    print("1. Manual Mode (Copy prompt to ChatGPT)")
    print("2. Rule-based Mode (Auto-suggest without LLM)")
    mode = input("\nSelect Mode (1 or 2): ").strip()

    recommendations = []

    if mode == "1":
        print("\n📋 Copy this prompt to ChatGPT:")
        print("-" * 40)
        print(generate_prompt_for_manual_llm(data))
        print("-" * 40)
        print("👉 Paste ChatGPT's JSON response below (Press Enter, then Ctrl+D or Ctrl+Z to finish):")
        try:
            lines = sys.stdin.read()
            recommendations = json.loads(lines)
        except Exception as e:
            print(f"❌ Input Error: {e}")
            return
            
    elif mode == "2":
        print("\n🤖 Running Rule-based Analysis...")
        recommendations = get_rule_based_recommendations(data)
        if not recommendations:
            print("No matching rules found in this scene.")
            return
    else:
        print("Invalid selection.")
        return

    # 추천 리스트 출력 및 선택
    print("\n" + "="*50)
    print(" 📝 Suggestions")
    print("="*50)
    for i, rec in enumerate(recommendations):
        icon = "🗑️" if rec['type'] == "REMOVE" else "📦"
        print(f"[{i+1}] {icon} {rec['desc']}")

    choice = input("\nSelect numbers to apply (e.g. '1 2' or 'all'): ").strip()
    if not choice: return

    selected = []
    if choice.lower() == 'all':
        selected = recommendations
    else:
        for idx in choice.split():
            if idx.isdigit():
                i = int(idx) - 1
                if 0 <= i < len(recommendations):
                    selected.append(recommendations[i])

    if selected:
        apply_edits(args.input_ply, args.instance_ids, args.output_ply, selected)

if __name__ == "__main__":
    main()