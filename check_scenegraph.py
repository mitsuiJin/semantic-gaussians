import os
import torch
import viser
import json
import numpy as np
import time
from omegaconf import OmegaConf
from scene import Scene
from model import GaussianModel
from utils.system_utils import searchForMaxIteration

def main(config, json_path):
    # 1. 3D Gaussian Splatting 모델 로드
    print("Loading 3D Gaussians...")
    
    if config.model.load_iteration == -1:
        loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
    else:
        loaded_iter = config.model.load_iteration
    
    print(f"Loading iteration {loaded_iter}...")

    scene = Scene(config.scene)
    gaussians = GaussianModel(config.model.sh_degree)
    
    ply_path = os.path.join(config.model.model_dir, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    # 2. Viser 서버 시작
    server = viser.ViserServer()
    print("Viser Server started! Check the URL in the terminal.")

    # 3. Scene Graph JSON 로드
    print(f"Loading Scene Graph from {json_path}...")
    with open(json_path, 'r') as f:
        scene_nodes = json.load(f)

    # 4. Bounding Box 및 라벨 그리기
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128)
    ]
    
    label_to_color = {}
    
    for i, node in enumerate(scene_nodes):
        label = node['label']
        if label not in label_to_color:
            label_to_color[label] = colors[len(label_to_color) % len(colors)]
        
        # BBox 정보 추출
        min_pt = np.array(node['bbox_min'])
        max_pt = np.array(node['bbox_max'])
        center = (min_pt + max_pt) / 2
        dimensions = max_pt - min_pt
        
        # [수정됨] line_width 제거 (이전 에러 반영)
        server.add_box(
            name=f"/objects/{node['id']}_{label}_box",
            position=center,
            dimensions=dimensions,
            color=label_to_color[label],
            visible=True,
        )
        
        # [수정됨] scale 제거 (현재 에러 반영)
        server.add_label(
            name=f"/objects/{node['id']}_{label}_text",
            text=f"{label} (ID:{node['id']})",
            position=max_pt,
            # scale=0.1  <-- 이 부분이 에러 원인이므로 삭제했습니다.
        )

    # 5. 가우시안 점 그리기 (Point Cloud)
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    # 다운샘플링 (웹 뷰어 성능을 위해 점 개수 조절)
    # 전체 점을 다 뿌리면 느릴 수 있으므로 1/10 수준으로 샘플링
    if len(xyz) > 1000000:
        mask = np.random.rand(len(xyz)) < 0.1 
        xyz = xyz[mask]
    
    print(f"Sending {len(xyz)} points to Viser...")
    
    # [확인 필요] 만약 여기서도 에러가 나면 point_size 인자를 지워야 할 수도 있습니다.
    server.add_point_cloud(
        "/gaussian_splats",
        points=xyz,
        colors=(200, 200, 200), # 회색으로 통일 (박스가 잘 보이게)
        point_size=0.01,
    )

    # 무한 루프
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Closing server...")

if __name__ == "__main__":
    config = OmegaConf.load("./config/view_scannet.yaml")
    
    # 경로 설정
    json_file_path = "scene_graph_nodes_7k.json" 
    config.model.load_iteration = 7000 
    
    main(config, json_file_path)