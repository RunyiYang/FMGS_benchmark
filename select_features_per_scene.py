import os
import json
import glob
import numpy as np


def process_scene_json(
    scene_json_path, output_json_name="lang_feat_selected_imgs.json", 
    max_num_frames=500
):
    """
    Processes a single transforms_undistorted.json file:
    - Filters frames where "is_bad" is False.
    - Downsamples by selecting every second frame.
    - Writes the result to a new JSON file.
    """

    with open(scene_json_path, "r") as f:
        data = json.load(f)

    total_num = len(data.get("frames", [])) + len(data.get("test_frames", []))
    # Combine both frame lists (preserving the original ordering)
    all_frames = data.get("frames", []) + data.get("test_frames", [])
    # Partition frames into "good" and "bad"
    good_frames = [frame for frame in all_frames if not frame.get("is_bad", True)]
    bad_frames = [frame for frame in all_frames if frame.get("is_bad", True)]

    # --- Selection Strategy ---
    # Case 1: We have enough good frames.
    if len(good_frames) >= max_num_frames:
        # Uniformly sample max_num_frames from the good frames.
        # (Using np.linspace ensures a uniform spread over the sequence.)
        indices = np.linspace(0, len(good_frames) - 1, max_num_frames, dtype=int)
        selected_frames = [good_frames[i] for i in indices]
    else:
        # Case 2: There are fewer than max_num_frames good frames.
        # Use all good frames and then supplement with bad frames.
        selected_frames = good_frames.copy()
        needed = max_num_frames - len(selected_frames)
        if len(bad_frames) >= needed:
            # Uniformly sample from the bad frames to fill the gap.
            indices = np.linspace(0, len(bad_frames) - 1, needed, dtype=int)
            additional = [bad_frames[i] for i in indices]
        else:
            # If even all bad frames aren’t enough, take them all.
            additional = bad_frames
        selected_frames.extend(additional)
        # (If the overall number is still less than max_num_frames, we simply use what we have.)

    # Optional: If for some reason you ended up with more than max_num_frames (e.g. after concatenation),
    # you can uniformly downsample.
    if len(selected_frames) > max_num_frames:
        indices = np.linspace(0, len(selected_frames) - 1, max_num_frames, dtype=int)
        selected_frames = [selected_frames[i] for i in indices]

    new_data = data.copy()
    new_data["frames"] = selected_frames
    new_data["num_frames"] = len(selected_frames)
    new_data["frames_list"] = [frame["file_path"] for frame in selected_frames]
    # remove test_frames from the json
    new_data.pop("test_frames", None)

    scene_folder = os.path.dirname(scene_json_path)
    output_json_path = os.path.join(scene_folder, output_json_name)

    with open(output_json_path, "w") as f:
        json.dump(new_data, f, indent=4)

    print(
        f"Processed and created {output_json_path} with {len(selected_frames)}/{total_num} frames."
    )


def main(data_root):
    """
    Processes all scene folders within the data_root directory.

    Args:
        data_root (str): Path to the root data directory containing scene folders.
    """
    pattern = os.path.join(data_root, "*/*/nerfstudio/nerfstudio/transforms_undistorted.json")
    scene_json_paths = glob.glob(pattern)

    if not scene_json_paths:
        print(f"No transforms_undistorted.json files found in {data_root}.")
        return

    print(f"Found {len(scene_json_paths)} scene(s) to process.")

    for scene_json in scene_json_paths:
        process_scene_json(scene_json)


if __name__ == "__main__":
    data_root = "/home/runyi_yang/benchmark2025/dataset/ScanNetpp/data"
    main(data_root)