#!/usr/bin/env bash
SRC_BASE="/insait/GSWorld/downloaded_datasets/scannet_gs_mcmc"
DST_BASE="/home/runyi_yang/benchmark2025/dataset/ScanNet/scans"

for scene_dir in "${DST_BASE}"/*; do
  scene=$(basename "$scene_dir")
  src_ply="${SRC_BASE}/${scene}/ckpts/point_cloud_30000.ply"
  dst_ply="${scene_dir}/point3d.ply"

  if [ -f "$src_ply" ]; then
    cp "$src_ply" "$dst_ply"
    echo "Copied ${src_ply} → ${dst_ply}"
  else
    echo "Warning: source not found: ${src_ply}" >&2
  fi
done
