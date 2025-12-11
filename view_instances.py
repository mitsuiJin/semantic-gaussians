import viser
import numpy as np
import os
from plyfile import PlyData
import time
import matplotlib.cm as cm

# [설정] 경로
PLY_PATH = "/data/shjin1019/repos/semantic-gaussians/output/scene0073_03_rgb_train/point_cloud/iteration_7000/point_cloud.ply"
INSTANCE_NPY_PATH = "instance_ids_7k.npy"

def load_ply_xyz(path):
    print("Loading PLY...")
    plydata = PlyData.read(path)
    xyz = np.stack((plydata['vertex']['x'], 
                    plydata['vertex']['y'], 
                    plydata['vertex']['z']), axis=1)
    return xyz

def get_random_colors(n):
    """인스턴스 개수만큼 랜덤 RGB 생성"""
    return np.random.randint(0, 255, size=(n, 3))

def main():
    server = viser.ViserServer()
    print("Viser Server started!")

    # 데이터 로드
    xyz = load_ply_xyz(PLY_PATH)
    instance_ids = np.load(INSTANCE_NPY_PATH)
    
    print(f"Total points: {len(xyz)}")
    print(f"Annotated points: {np.sum(instance_ids != -1)}")

    # 색상 배열 초기화 (기본 회색)
    colors = np.full_like(xyz, 200, dtype=np.uint8) # (N, 3) Light Gray
    
    # 인스턴스별 색상 할당
    unique_ids = np.unique(instance_ids)
    # 배경(-1) 제외한 ID 개수
    max_id = unique_ids.max()
    
    if max_id >= 0:
        # ID별 랜덤 색상 팔레트 생성
        palette = get_random_colors(max_id + 1)
        
        # 벡터화된 색상 할당 (빠름)
        mask = (instance_ids != -1)
        colors[mask] = palette[instance_ids[mask]]

    # 다운샘플링 (웹 뷰어 렉 방지)
    if len(xyz) > 1000000:
        print("Downsampling for smoother visualization...")
        choice = np.random.choice(len(xyz), 1000000, replace=False)
        xyz = xyz[choice]
        colors = colors[choice]

    # 포인트 클라우드 전송
    server.add_point_cloud(
        "/colored_instances",
        points=xyz,
        colors=colors,
        point_size=0.01
    )
    
    print("Done! Check your browser.")
    
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()