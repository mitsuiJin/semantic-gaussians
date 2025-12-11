#!/usr/bin/env python3
"""
Scene Graph Builder for Gaussian Grouping + Semantic Gaussians

Usage:
    python build_scene_graph.py \
        --gaussian_grouping /path/to/gaussian-grouping/output/lerf/figurines \
        --semantic_gaussians /path/to/semantic-gaussians/output/figurines \
        --output scene_graph.json
"""

import os
import json
import torch
import numpy as np
import clip
from pathlib import Path
from tqdm import tqdm
import argparse


class GaussianSceneGraph:
    def __init__(self, gg_path, sg_path):
        """
        Args:
            gg_path: Gaussian Grouping output path
            sg_path: Semantic Gaussians output path
        """
        self.gg_path = Path(gg_path)
        self.sg_path = Path(sg_path)
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-L/14@336px", device=self.device)
        
        print("Loading data...")
        self.load_gaussian_grouping()
        self.load_semantic_features()
        
    def load_gaussian_grouping(self):
        """Load Gaussian Grouping results"""
        # 1. Load point cloud
        ply_path = self.gg_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"
        print(f"Loading PLY: {ply_path}")
        
        try:
            from plyfile import PlyData
            plydata = PlyData.read(str(ply_path))
        except Exception as e:
            print(f"Error reading PLY file: {e}")
            print("Please ensure 'plyfile' is installed: pip install plyfile")
            exit(1)
            
        
        # Extract positions
        vertices = plydata['vertex']
        self.positions = np.stack([
            vertices['x'], vertices['y'], vertices['z']
        ], axis=1)
        
        # Extract object features (Identity Encoding)
        if 'obj_dc_0' in vertices:
            self.object_features = np.stack([
                vertices[f'obj_dc_{i}'] for i in range(16)
            ], axis=1)
            print(f"Loaded object features shape: {self.object_features.shape}")
        else:
            print("Warning: No object features found in PLY")
            self.object_features = None
        
        # 2. Load classifier
        classifier_path = self.gg_path / "point_cloud" / "iteration_30000" / "classifier.pth"
        print(f"Loading classifier: {classifier_path}")
        
        if classifier_path.exists():
            classifier_dict = torch.load(classifier_path, map_location=self.device)
            
            # classifier is torch.nn.Conv2d(16, num_classes, kernel_size=1)
            num_classes = classifier_dict['weight'].shape[0]
            
            self.classifier = torch.nn.Conv2d(16, num_classes, kernel_size=1)
            self.classifier.load_state_dict(classifier_dict)
            self.classifier.to(self.device)
            self.classifier.eval()
            
            print(f"Loaded classifier: 16 -> {num_classes} classes")
            
            # Predict group IDs
            if self.object_features is not None:
                # Reshape for Conv2d: [N, 16] -> [1, 16, N, 1]
                feat_tensor = torch.from_numpy(self.object_features).float()
                feat_tensor = feat_tensor.unsqueeze(0).unsqueeze(-1).permute(0, 2, 1, 3)  # [1, 16, N, 1]
                feat_tensor = feat_tensor.to(self.device)
                
                with torch.no_grad():
                    logits = self.classifier(feat_tensor)  # [1, num_classes, N, 1]
                    logits = logits.squeeze(0).squeeze(-1).permute(1, 0)  # [N, num_classes]
                    self.group_ids = logits.argmax(dim=1).cpu().numpy()
                    
                print(f"Found {self.group_ids.max() + 1} groups for {len(self.group_ids)} points")
            else:
                self.group_ids = None
        else:
            print("Warning: classifier.pth not found")
            self.classifier = None
            self.group_ids = None
    
    def load_semantic_features(self):
        """Load Semantic Gaussians results"""
        # Handle both absolute and relative paths
        if Path(self.sg_path).is_absolute():
            fusion_path = Path(self.sg_path) / "0.pt"
        else:
            fusion_path = Path.cwd() / self.sg_path / "0.pt"
        
        print(f"Loading semantic features: {fusion_path}")
        
        data = torch.load(fusion_path, map_location=self.device)
        self.semantic_features = data['feat'].to(self.device)  # [N, 768]
        
        # ⭐ 1. NaN 픽스: valid_mask를 numpy로 변환
        self.valid_mask = data['mask_full'].cpu().numpy()     # [N]
        
        print(f"Semantic features shape: {self.semantic_features.shape}")

    # def get_categories_from_masks(self):
    #     """Extract categories from the dataset source folder"""
    #     mask_dir = Path() # 변수 초기화
    #     try:
    #         # self.gg_path = /data/shjin1019/repos/gaussian-grouping/output/lerf/figurines
    #         # 목표: /data/shjin1019/repos/gaussian-grouping/data/figurines/test_mask
            
    #         # ⭐ 2. 'bbase_' 오타 수정 및 경로 로직 수정
    #         # .../figurines -> .../lerf -> .../output -> .../gaussian-grouping
    #         base_project_path = self.gg_path.parent.parent.parent
            
    #         # 2. Get dataset name: "figurines"
    #         dataset_name = self.gg_path.name
            
    #         # 3. Construct the path to the original data (train_lerf.sh 참고)
    #         data_source_path = base_project_path / "data" / dataset_name
    #         mask_dir = data_source_path / "test_mask"
            
    #         print(f"Attempting to find real categories in: {mask_dir}")

    #     except Exception as e:
    #         print(f"Warning: Could not infer mask path: {e}. Using fallback.")
        
    #     if mask_dir.exists():
    #         print("Success: Found real categories in source data folder.")
    #         categories = []
    #         # 하위 폴더까지 재귀적으로 .png 검색
    #         for mask_file in mask_dir.rglob("*.png"):
    #             cat = mask_file.stem.replace("_", " ")
    #             categories.append(cat)
            
    #         unique_categories = sorted(list(set(categories)))
    #         if unique_categories:
    #             # 씬그래프에 필요한 기본 단어 추가
    #             unique_categories.extend(["floor", "wall", "table", "desk", "background"])
    #             return sorted(list(set(unique_categories)))

    #     print("Warning: Could not find 'test_mask' folder. Using fallback list.")
    #     # Option 2: Common indoor objects as fallback
    #     return [
    #         "rabbit toy", "bear toy", "elephant toy", "dog toy",
    #         "wooden table", "chair", "lamp", "vase",
    #         "floor", "wall", "carpet"
    #     ]
    def get_categories_from_masks(self):
        """Extract categories from the dataset source folder"""
        print("Using FIXED, high-quality category list for Figurines.")
        
        # ⭐ 최종 확정된 Figurines 씬의 객체 목록 (배경 포함)
        return [
            "red apple", 
            "old camera", 
            "porcelain hand", 
            "rubber duck with red hat", 
            "green toy chair", 
            "red toy chair",
            "table", 
            "wooden table",
            "floor", 
            "wall", 
            "background"
        ]
    
    def classify_group(self, group_indices, categories):
        """Classify a group using CLIP"""
        
        # ⭐ 3. (시작) N_A_N 오류 수정
        
        # 3-1. 유효한 인덱스가 0개인지 먼저 확인
        if len(group_indices) == 0:
            return categories[0], 0.0 # 시맨틱 정보 없음

        # Get mean semantic feature
        group_feat = self.semantic_features[group_indices].mean(dim=0)
        
        # 3-2. 평균 계산 후 NaN이 되었는지 확인 (0.pt 파일에 NaN이 있을 경우)
        if torch.isnan(group_feat).any():
            return categories[0], 0.0 # 유효한 피처가 없음

        # Encode text categories
        text_tokens = clip.tokenize(categories).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 3-3. 0으로 나누기 방지
        group_feat_norm_value = group_feat.norm()
        if group_feat_norm_value == 0:
            return categories[0], 0.0 

        group_feat_norm = group_feat / group_feat_norm_value
        similarities = (group_feat_norm @ text_features.T).detach().cpu().numpy()
        
        # --- (끝) N_A_N 오류 수정 ---

        best_idx = similarities.argmax()
        confidence = similarities[best_idx]
        
        return categories[best_idx], float(confidence)
    
    def compute_bbox(self, indices):
        """Compute bounding box for a group"""
        points = self.positions[indices]
        return {
            'min': points.min(axis=0).tolist(),
            'max': points.max(axis=0).tolist(),
            'center': points.mean(axis=0).tolist()
        }
    
    def compute_spatial_relation(self, bbox1, bbox2):
        """Compute spatial relation between two objects"""
        center1 = np.array(bbox1['center'])
        center2 = np.array(bbox2['center'])
        
        min1 = np.array(bbox1['min'])
        max1 = np.array(bbox1['max'])
        min2 = np.array(bbox2['min'])
        max2 = np.array(bbox2['max'])
        
        # Y축이 위쪽이라고 가정 (dataset에 따라 다를 수 있음)
        # Check "on" relation (obj 2 is on top of obj 1)
        # 1. Y(수직) 최소값이 1의 Y(수직) 최대값보다 높거나 비슷해야 함
        # 2. XZ(수평) 평면에서 겹쳐야 함
        is_above = min2[1] >= max1[1] - 0.1 # 0.1은 약간의 오차 허용
        overlap_x = (min1[0] <= max2[0]) and (min2[0] <= max1[0])
        overlap_z = (min1[2] <= max2[2]) and (min2[2] <= max1[2])

        if is_above and overlap_x and overlap_z:
            return "on"
        
        # Check "next_to" relation
        # 1. Y(수직) 중심이 비슷해야 함
        # 2. XZ(수평) 거리가 가까워야 함
        y_diff = abs(center1[1] - center2[1])
        size1_y = max1[1] - min1[1]
        size2_y = max2[1] - min2[1]

        if y_diff < (size1_y + size2_y) / 2: # 수직으로 겹치는 부분이 있음
            xz_dist = np.linalg.norm(center1[[0, 2]] - center2[[0, 2]])
            size1_xz = np.linalg.norm(max1[[0, 2]] - min1[[0, 2]])
            size2_xz = np.linalg.norm(max2[[0, 2]] - min2[[0, 2]])
            
            if xz_dist < (size1_xz + size2_xz) * 0.7: # 임계값 기반 '가까움'
                return "next_to"
        
        return None
    
    def build_scene_graph(self):
        """Build complete scene graph"""
        if self.group_ids is None:
            print("Error: No group IDs available")
            return None
        
        categories = self.get_categories_from_masks()
        print(f"Using categories: {categories}")
        
        # Build nodes
        scene_graph = {
            'nodes': [],
            'edges': []
        }
        
        unique_groups = np.unique(self.group_ids)
        print(f"Processing {len(unique_groups)} groups...")
        
        for group_id in tqdm(unique_groups):
            group_mask = self.group_ids == group_id
            
            # ⭐ 4. NaN 픽스: 유효 마스크(valid_mask)와 AND 연산
            valid_group_mask = group_mask & self.valid_mask
            group_indices = np.where(valid_group_mask)[0]
            
            # ⭐ 5. NaN 픽스: 포인트가 0개인 그룹은 즉시 건너뛰기
            #    (classify_group 함수 내부로 이 로직이 이동했으므로,
            #     여기서는 원본 Bbox 계산용 인덱스만 체크)
            original_group_indices = np.where(group_mask)[0]
            
            if len(original_group_indices) < 100:  # Skip tiny groups (포인트 100개 미만)
                continue
            
            # Classify group
            label, confidence = self.classify_group(
                torch.from_numpy(group_indices).to(self.device), # 시맨틱 계산은 유효한 인덱스만 사용
                categories
            )
            
            # Compute bbox (Bbox는 전체 포인트로 계산)
            bbox = self.compute_bbox(original_group_indices)
            
            node = {
                'id': f"obj_{int(group_id)}",
                'group_id': int(group_id),
                'label': label,
                'confidence': confidence,
                'bbox': bbox,
                'num_points': len(original_group_indices)
            }
            scene_graph['nodes'].append(node)
        
        # Build edges (spatial relations)
        print("Computing spatial relations...")
        for i, node1 in enumerate(scene_graph['nodes']):
            for node2 in scene_graph['nodes'][i+1:]:
                relation = self.compute_spatial_relation(
                    node1['bbox'], 
                    node2['bbox']
                )
                if relation:
                    scene_graph['edges'].append({
                        'source': node1['id'],
                        'target': node2['id'],
                        'relation': relation
                    })
        
        return scene_graph
    
    def save_scene_graph(self, scene_graph, output_path):
        """Save scene graph to JSON"""
        with open(output_path, 'w') as f:
            json.dump(scene_graph, f, indent=2)
        print(f"Scene graph saved to {output_path}")
    
    def visualize_scene_graph(self, scene_graph):
        """Print scene graph in readable format"""
        print("\n" + "="*60)
        print("SCENE GRAPH")
        print("="*60)
        
        print("\nObjects:")
        for node in scene_graph['nodes']:
            print(f"  {node['id']}: {node['label']} "
                  f"(confidence: {node['confidence']:.2f}, "
                  f"points: {node['num_points']})")
        
        print("\nRelations:")
        for edge in scene_graph['edges']:
            print(f"  {edge['source']} --{edge['relation']}--> {edge['target']}")
        
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian_grouping', type=str, required=True,
                        help='Path to Gaussian Grouping output')
    parser.add_argument('--semantic_gaussians', type=str, required=True,
                        help='Path to Semantic Gaussians output')
    parser.add_argument('--output', type=str, default='scene_graph.json',
                        help='Output JSON file')
    args = parser.parse_args()
    
    # Build scene graph
    builder = GaussianSceneGraph(
        args.gaussian_grouping,
        args.semantic_gaussians
    )
    
    scene_graph = builder.build_scene_graph()
    
    if scene_graph:
        builder.visualize_scene_graph(scene_graph)
        builder.save_scene_graph(scene_graph, args.output)
        
        print(f"\n✅ Success!")
        print(f"   - Nodes: {len(scene_graph['nodes'])}")
        print(f"   - Edges: {len(scene_graph['edges'])}")


if __name__ == "__main__":
    main()