#!/usr/bin/env python3
import torch
import sys
import os

# Fusion 파일 확인
fusion_dir = "output/teatime_result/fusion_output"

print("=" * 80)
print("Fusion Data Analysis")
print("=" * 80)

fusion_files = []
for i in range(5):
    fpath = os.path.join(fusion_dir, f"{i}.pt")
    if os.path.exists(fpath):
        fusion_files.append(fpath)

if not fusion_files:
    print(f"ERROR: No fusion files found in {fusion_dir}")
    sys.exit(1)

total_points = 0
total_memory_mb = 0

for i, fpath in enumerate(fusion_files):
    try:
        data = torch.load(fpath, map_location='cpu')
        feat = data['feat']
        mask_full = data['mask_full']
        
        # 메모리 계산
        feat_memory_mb = (feat.element_size() * feat.nelement()) / (1024**2)
        mask_memory_mb = (mask_full.element_size() * mask_full.nelement()) / (1024**2)
        file_total_mb = feat_memory_mb + mask_memory_mb
        
        num_points = feat.shape[0]
        total_points += num_points
        total_memory_mb += file_total_mb
        
        print(f"\nFile {i}.pt:")
        print(f"  Feature shape: {feat.shape}")
        print(f"  Number of points: {num_points:,}")
        print(f"  Feature dtype: {feat.dtype}")
        print(f"  Mask shape: {mask_full.shape}")
        print(f"  Mask True count: {mask_full.sum().item():,}")
        print(f"  Feature memory: {feat_memory_mb:.2f} MB")
        print(f"  Total file memory: {file_total_mb:.2f} MB")
        
        # 특징 차원 확인
        if len(feat.shape) == 2:
            print(f"  Feature dimension: {feat.shape[1]}")
        
    except Exception as e:
        print(f"\nFile {i}.pt: ERROR - {e}")

print("\n" + "=" * 80)
print("Summary:")
print("=" * 80)
print(f"Total points across all files: {total_points:,}")
print(f"Total memory: {total_memory_mb:.2f} MB")
print(f"Average points per file: {total_points / len(fusion_files):,.0f}")
print("\n" + "=" * 80)
print("Diagnosis:")
print("=" * 80)

avg_points = total_points / len(fusion_files)
if avg_points <= 90000:
    print("✅ CORRECT: n_split_points=80000 was used")
    print("   This is the proper training configuration.")
elif avg_points > 500000:
    print("❌ WRONG: n_split_points=999999999 was used")
    print("   This is the evaluation configuration and causes OOM!")
    print("   You MUST re-run fusion.py with n_split_points=80000")
else:
    print("⚠️  UNCLEAR: Unexpected number of points")
    print(f"   Expected: ~80,000 per file")
    print(f"   Got: {avg_points:,.0f} per file")

print("=" * 80)