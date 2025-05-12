#!/usr/bin/env python3
"""
This script processes a subset (or all) of ScanNet scenes by index range.
It dynamically gathers ScanNet scene directories under DATA_DIR,
allows slicing with start_idx and end_idx, creates a `gsplat` folder for each,
and runs two training stages via subprocess.
"""
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process ScanNet scenes in batches by index range."
    )
    parser.add_argument(
        "--data_dir", type=Path,
        default=Path("/home/runyi_yang/benchmark2025/dataset/Scannet_val/original_data"),
        help="Base directory containing ScanNet scenes"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0,
        help="Start index (inclusive) of scenes list"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None,
        help="End index (exclusive) of scenes list; if None, process to end"
    )
    parser.add_argument(
        "--initial_port", type=int, default=6009,
        help="Port for initial checkpoint-only run"
    )
    parser.add_argument(
        "--finetune_port", type=int, default=6019,
        help="Port for full fine-tune / render-feature run"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    src_base = data_dir

    # Gather all scene IDs (subdirectories of data_dir)
    scenes = [d.name for d in data_dir.iterdir() if d.is_dir()]
    total = len(scenes)

    # Compute slicing indices
    start = max(0, args.start_idx)
    end = args.end_idx if args.end_idx is not None else total
    end = min(end, total)
    selected = scenes[start:end]
    print(f"Processing scenes {start} to {end} (total {len(selected)}) out of {total}…")

    for scene in tqdm(selected):
        scene_dir = src_base / scene
        model_dir = scene_dir / "gsplat"
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== Processing {scene} ===")

        # 1) initial checkpoint-only run
        # subprocess.run([
        #     "python", "train.py",
        #     "-s", str(scene_dir),
        #     "--dataformat", "scannet",
        #     "--model_path", str(model_dir),
        #     "--checkpoint_iterations", "7000", "30000",
        #     "--port", str(args.initial_port)
        # ], check=True)

        # 2) full fine-tune / render-feature run
        subprocess.run([
            "python", "train.py",
            "-s", str(scene_dir),
            "--dataformat", "scannet",
            "--model_path", str(model_dir),
            "--opt_vlrenderfeat_from", "30000",
            "--test_iterations", "32000", "32500",
            "--save_iterations", "32000", "32500",
            "--iterations", "32500",
            "--checkpoint_iterations", "32000", "32500",
            "--start_checkpoint", str(model_dir / "chkpnt30000.pth"),
            "--fmap_resolution", "2",
            "--lambda_clip", "0.2",
            "--fmap_lr", "0.005",
            "--fmap_render_radiithre", "2",
            "--port", str(args.finetune_port)
        ], check=True)

        # Uncomment below to run 3D feature extraction on train set
        # subprocess.run([
        #     "python", "get_3d_features.py",
        #     "-s", str(scene_dir),
        #     "--model_path", str(model_dir),
        #     "--dataformat", "holicity",
        #     "--runon_train"
        # ], check=True)


if __name__ == "__main__":
    main()
