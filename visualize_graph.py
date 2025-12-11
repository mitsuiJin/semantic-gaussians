import json
import networkx as nx
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="scene_graph_final.json")
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    # 그래프 생성
    G = nx.DiGraph()
    
    # 노드 색상 설정 (라벨별)
    color_map = {
        "bed": "#8dd3c7", "chair": "#ffffb3", "table": "#bebada", 
        "door": "#fb8072", "cabinet": "#80b1d3", "curtain": "#fdb462", 
        "clothes": "#b3de69", "floor": "#d9d9d9", "wall": "#bc80bd"
    }
    
    node_colors = []
    labels = {}

    for node in data['nodes']:
        G.add_node(node['id'])
        labels[node['id']] = f"{node['label']}\n({node['id']})"
        color = color_map.get(node['label'], "#cccccc")
        node_colors.append(color)

    # 엣지(관계) 추가
    edge_labels = {}
    for rel in data['relationships']:
        G.add_edge(rel['subject_id'], rel['object_id'])
        edge_labels[(rel['subject_id'], rel['object_id'])] = rel['predicate']

    # 그리기
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1.5, seed=42) # 레이아웃 고정

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, edgecolors='black', alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, arrowsize=20, edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, font_color='red')

    plt.title("3D Scene Graph Topology", fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("scene_graph_visualization.png")
    print("Saved graph image to 'scene_graph_visualization.png'")
    plt.show()

if __name__ == "__main__":
    main()