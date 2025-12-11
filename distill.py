import os
import uuid
import torch
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from model import GaussianModel, render
from scene import Scene
from utils.system_utils import set_seed, searchForMaxIteration

from model.mink_unet import mink_unet
from model.render_utils import get_text_features
from model.openseg_predictor import OpenSeg
from dataset.feature_dataset import FeatureDataset

from sklearn.neighbors import NearestNeighbors  # [추가] 점 개수 맞추기용
from utils.dataset_utils import load_gaussian_ply  # [추가] 원본 점 로드용
import MinkowskiEngine as ME


def collate_fn(batch):
    locs, features, features_gt, mask, head_id = list(zip(*batch))

    for i in range(len(locs)):
        locs[i][:, 0] *= i

    return torch.cat(locs), torch.cat(features), torch.cat(features_gt), torch.cat(mask), head_id[0]


def tensor_field(locs, features, voxel_size):
    return ME.TensorField(
        features=features.float(),
        coordinates=ME.utils.batched_coordinates([locs / voxel_size], dtype=torch.float32),
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
        device="cuda",
    )


def init_dir(config):
    if not config.distill.exp_name:
        unique_str = str(uuid.uuid4())
        config.distill.model_dir = os.path.join("./results_distill/", unique_str[0:10])
    else:
        config.distill.model_dir = f"./results_distill/{config.distill.exp_name}"

    print("Output folder: {}".format(config.distill.model_dir))
    os.makedirs(config.distill.model_dir, exist_ok=True)
    with open(os.path.join(config.distill.model_dir, "config.yaml"), "w") as fp:
        OmegaConf.save(config, fp)

    os.makedirs(os.path.join(config.distill.model_dir, "tb_logs"), exist_ok=True)
    writer = SummaryWriter(os.path.join(config.distill.model_dir, "tb_logs"))
    return writer


def distill(config):
    if config.distill.feature_type == "all":
        model_3d = mink_unet(in_channels=56, out_channels=768, D=3, arch=config.distill.model_3d).cuda()
    elif config.distill.feature_type == "color":
        model_3d = mink_unet(in_channels=48, out_channels=768, D=3, arch=config.distill.model_3d).cuda()
    model_2d = OpenSeg(None, "ViT-L/14@336px")

    writer = init_dir(config)

    optimizer = torch.optim.AdamW(model_3d.parameters(), lr=config.distill.lr)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, config.distill.schedule_milestones, config.distill.schedule_gamma
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.distill.epochs)

    dataset = FeatureDataset(
        config.model.model_dir,
        config.fusion.out_dir,
        config.model.load_iteration,
        config.distill.voxel_size,
        config.distill.aug,
        config.distill.feature_type,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.distill.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.distill.num_workers,
    )

    ema_loss_for_log = 0.0
    #eval(config, model_3d, model_2d, dataset.voxelizer, -1)

    progress_bar = tqdm(
        range(config.distill.epochs * len(loader)),
        desc="Distill progress",
        dynamic_ncols=True,
    )

    for i in range(config.distill.epochs):
        for j, batch in enumerate(loader):
            # [추가] 루프 시작할 때마다 GPU 캐시 비우기 (이 한 줄이면 됩니다!)
            torch.cuda.empty_cache()
            locs, features, features_gt, mask, head_id = batch
            locs[:, 1:4] += (torch.rand(3) * 100).type_as(locs)

            sinput = ME.SparseTensor(features.cuda(), locs.cuda())
            features_gt, mask = features_gt.cuda(), mask.cuda()

            output = model_3d(sinput).F[mask]

            if config.distill.loss_type == "cosine":
                norm_mask = features_gt.norm(dim=-1) > 0
                if norm_mask.sum() == 0:
                    continue
                loss = (
                    1
                    - torch.nn.CosineSimilarity()(
                        output[norm_mask][:, head_id * 768 : (head_id + 1) * 768], features_gt[norm_mask]
                    )
                ).mean()
            elif config.distill.loss_type == "l1":
                loss = torch.nn.L1Loss()(output[:, head_id * 768 : (head_id + 1) * 768], features_gt)
            elif config.distill.loss_type == "l2":
                loss = torch.nn.MSELoss()(output[:, head_id * 768 : (head_id + 1) * 768], features_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), progress_bar.n)
            writer.add_scalar("lr", optimizer.state_dict()["param_groups"][0]["lr"], progress_bar.n)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(1)

        scheduler.step()

        if (i + 1) % config.distill.test_interval == 0:
            eval(config, model_3d, model_2d, dataset.voxelizer, i)

        if (i + 1) % config.distill.save_interval == 0:
            iteration_path = os.path.join(config.distill.model_dir, f"weights/{i+1}")
            os.makedirs(iteration_path, exist_ok=True)
            torch.save(model_3d.state_dict(), os.path.join(iteration_path, "model.pth"))
    # [여기서부터 추가] ========================================================
    print("\n>>> Training Finished. Starting Inference & Devoxelization...")
    
    # 1. 저장 경로 설정 (weights/100 폴더 안에 저장)
    save_dir = os.path.join(config.distill.model_dir, "distilled_results", config.distill.exp_name, "weights", "100")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "predictions.pt")
    
    # 2. 모델 평가 모드 전환
    model_3d.eval()
    
    with torch.no_grad():
        # 3. 원본 포인트 클라우드(PLY) 읽기 (점 개수를 맞추기 위해 필수!)
        # dataset.data[0][0]에는 첫 번째 씬의 ply 경로가 들어있습니다.
        ply_path = dataset.data[0][0]
        print(f">>> Loading Original PLY from: {ply_path}")
        original_locs, _ = load_gaussian_ply(ply_path, "all") # (N_points, 3)
        
        # 4. 데이터셋에서 데이터 가져오기
        # (학습 때 썼던 dataset 변수를 그대로 쓰므로 경로 에러 없음)
        locs, feats, _, mask_chunk, _ = dataset[0]
        
        # [수정됨] 이미 Tensor이므로 from_numpy 제거 & 차원 맞춤
        # locs는 (N, 4) 형태이며 첫 컬럼이 1로 되어있음 -> 0으로 변경
        b_coords = locs.int()
        b_coords[:, 0] = 0
        feats = feats.float()
        
        # 5. 모델 추론 (결과는 압축된 복셀 단위)
        in_field = ME.SparseTensor(features=feats.cuda(), coordinates=b_coords.cuda())
        out_field = model_3d(in_field)
        
        voxel_preds = out_field.features.cpu().numpy() # (M_voxels, Dim)
        # 좌표 복원 (Voxel Size 곱해주기)
        voxel_coords = out_field.C[:, 1:].cpu().numpy() * config.distill.voxel_size
        
        print(f">>> Voxel count: {voxel_preds.shape[0]}, Target Point count: {mask_chunk.sum()}")

        # 6. [핵심] 복셀 -> 원본 포인트 매핑 (KNN)
        # 뷰어 에러(Illegal Memory Access)를 막기 위해, 복셀 값을 원본 점 개수에 맞춰 늘려줍니다.
        target_mask = mask_chunk.bool().cpu().numpy()
        target_locs = original_locs[target_mask] # 마스킹된 원본 점들
        
        print(">>> Running KNN mapping (Devoxelization)...")
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(voxel_coords)
        _, indices = nbrs.kneighbors(target_locs)
        
        # 매핑된 피처 생성 (이제 개수가 딱 맞습니다!)
        point_preds = voxel_preds[indices.flatten()]
        
        # 7. 최종 저장
        torch.save(
            {
                "feat": torch.from_numpy(point_preds).half().cuda(),
                "mask_full": mask_chunk.cuda()
            },
            save_path
        )
        print(f">>> Save Complete! File: {save_path}")
    # ==========================================================================
    progress_bar.close()


def eval(config, model_3d, model_2d, voxelizer, iter):
    print("Evaluating at iteration", iter + 1)
    eval_config = deepcopy(config)
    if config.model.dynamic:
        eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, "tennis", "50")
        eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, "tennis")
    else:
        eval_config.scene.scene_path = os.path.join(
            eval_config.scene.scene_path.replace("train", "val"), "scene0011_00"
        )
        eval_config.model.model_dir = os.path.join(eval_config.model.model_dir.replace("train", "val"), "scene0011_00")

    scene = Scene(eval_config.scene)
    gaussians = GaussianModel(eval_config.model.sh_degree)
    if config.model.dynamic:
        gaussians.load_dynamic_npz(os.path.join(eval_config.model.model_dir, "params.npz"), 50)
    else:
        loaded_iter = config.model.load_iteration
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(os.path.join(eval_config.model.model_dir, "point_cloud"))
        print(f"Loading iteration {loaded_iter}...")
        gaussians.load_ply(
            os.path.join(
                eval_config.model.model_dir,
                "point_cloud",
                f"iteration_{loaded_iter}",
                "point_cloud.ply",
            )
        )

    locs, features = gaussians.get_locs_and_features(config.distill.feature_type)
    locs, features, _, _, vox_ind = voxelizer.voxelize(locs, features, None, return_ind=True)
    locs = torch.from_numpy(locs).int()
    locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
    features = torch.from_numpy(features).float()
    vox_ind = torch.from_numpy(vox_ind).cuda()

    sinput = ME.SparseTensor(features.cuda(), locs.cuda())

    features_semantic = model_3d(sinput).F[:, :768]
    features_semantic = features_semantic / features_semantic.norm(dim=-1, keepdim=True)

    bg_color = [1, 1, 1] if eval_config.scene.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras()
    with torch.no_grad():
        palette, text_features = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
        sim = torch.einsum("cq,dq->dc", text_features, features_semantic)
        label = sim.argmax(dim=1)

        new_3d = torch.zeros((label.shape[0], 3)).cuda()
        u_index = torch.unique(label)
        for index in u_index:
            if index == 255:
                index_ = 20
            else:
                index_ = index

            new_3d[label == index] = torch.tensor(
                [
                    palette[index_ * 3] / 255.0,
                    palette[index_ * 3 + 1] / 255.0,
                    palette[index_ * 3 + 2] / 255.0,
                ]
            ).cuda()

        from utils.sh_utils import RGB2SH

        gaussians._features_dc[:] = 0
        gaussians._features_dc[vox_ind, 0] = RGB2SH(new_3d)
        gaussians._features_rest[:] = 0

        iteration_path = os.path.join(eval_config.distill.model_dir, f"semantic/{iter+1}")
        os.makedirs(iteration_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views)):
            if idx % 5 != 0:
                continue
            view.cuda()
            rendering = render(view, gaussians, eval_config.pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(iteration_path, "{0:05d}".format(idx) + ".png"))


if __name__ == "__main__":
    config = OmegaConf.load("./config/distill_scannet.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)

    distill(config)