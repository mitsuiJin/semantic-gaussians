import os
import torch
import numpy as np

from torch.utils.data import Dataset
from dataset.fusion_utils import Voxelizer
from dataset.augmentation import ElasticDistortion, RandomHorizontalFlip, Compose
from utils.dataset_utils import load_gaussian_ply


class FeatureDataset(Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = (
        (-np.pi / 64, np.pi / 64),
        (-np.pi / 64, np.pi / 64),
        (-np.pi, np.pi),
    )
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = "z"

    def __init__(
        self, gaussians_dir, point_dir, gaussian_iterations=30000, voxel_size=0.02, aug=False, feature_type="all"
    ):
        self.aug = aug
        self.feature_type = feature_type
        
        # [수정 1] gaussians_dir 안에 있는 것 중 '폴더'만 가져오기 (config.yaml 같은 파일 무시)
        all_items = os.listdir(gaussians_dir)
        self.scenes = [d for d in all_items if os.path.isdir(os.path.join(gaussians_dir, d))]
        self.scenes.sort()

        self.data = []
        for scene in self.scenes:
            # [수정 2] 해당 씬의 Fusion 결과 폴더가 실제로 존재하는지 확인 (안전장치)
            scene_fusion_path = os.path.join(point_dir, scene)
            if not os.path.isdir(scene_fusion_path):
                print(f"[Warning] Skipping scene '{scene}': Fusion result not found at {scene_fusion_path}")
                continue

            features = os.listdir(scene_fusion_path)
            features.sort()
            for feature in features:
                # .pt 파일이 아닌 것이 섞여 있을 경우를 대비해 확장자 체크 (선택 사항이지만 안전함)
                if not feature.endswith('.pt'):
                    continue

                ply_path = os.path.join(
                    gaussians_dir,
                    scene,
                    "point_cloud",
                    f"iteration_{gaussian_iterations}",
                    "point_cloud.ply",
                )
                feature_path = os.path.join(point_dir, scene, feature)
                self.data.append([ply_path, feature_path, 0])

        print(f"[Dataset] Loaded {len(self.data)} data items from {len(self.scenes)} scenes.")

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=aug,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
        )

        self.prevoxel_transforms = Compose([ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)])
        self.input_transforms = Compose([RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False)])

    def __getitem__(self, index):
        with torch.no_grad():
            ply_path, feature_path, head_id = self.data[index]
            locs, features = load_gaussian_ply(ply_path, self.feature_type)
            gt = torch.load(feature_path)
            features_gt, mask_chunk = gt["feat"], gt["mask_full"]

            # numpy transforms
            if self.aug:
                locs = self.prevoxel_transforms(locs)

            locs, features, _, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs, features, None, return_ind=True
            )

            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)

            indices = index3[chunk_ind] - 1
            features_gt = features_gt[indices]

            if self.aug:
                locs, features, _ = self.input_transforms(locs, features, None)

            locs = torch.from_numpy(locs).int()
            locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
            features = torch.from_numpy(features).float()

        return locs, features, features_gt, mask, head_id

    def __len__(self):
        return len(self.data)