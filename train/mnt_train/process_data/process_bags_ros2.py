
import os
import argparse
import yaml
import tqdm
import pickle
import numpy as np

from process_data_utils_ros2 import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(args: argparse.Namespace):
    # Load config
    with open(os.path.join(SCRIPT_DIR, "process_bags_config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect ROS2 bag paths.
    # - ROS2 Humble default storage: sqlite3 -> `*.db3` inside a bag directory
    # - ROS2 Jazzy often uses MCAP storage: `*.mcap` inside a bag directory
    # Prefer passing the BAG DIRECTORY (the folder containing `metadata.yaml`) when available.
    bag_paths = []
    seen = set()
    for root, _, files in os.walk(args.input_dir):
        files_set = set(files)

        # If this looks like a rosbag2 directory, add the directory itself once.
        if "metadata.yaml" in files_set:
            if root not in seen:
                bag_paths.append(root)
                seen.add(root)
            continue

        # Fallback: if metadata.yaml is missing, add individual storage files.
        for file in files:
            if file.endswith((".db3", ".mcap", ".bag")):
                p = os.path.join(root, file)
                if p not in seen:
                    bag_paths.append(p)
                    seen.add(p)
    if args.num_trajs >= 0:
        bag_paths = bag_paths[:args.num_trajs]

    # Iterate over bags
    for bag_path in tqdm.tqdm(bag_paths, desc="Bags processed"):
        bp = os.path.normpath(bag_path)
        # If `bag_path` is a directory, use its directory name; otherwise derive from the file name.
        if os.path.isdir(bp):
            traj_name = os.path.basename(bp)
        else:
            traj_name = "_".join(bp.split(os.sep)[-2:])
            traj_name = traj_name.replace(".db3", "").replace(".mcap", "").replace(".bag", "")
        
        try:
            result = get_images_lidar_and_odom(
                bag_path=bag_path,
                imtopics=config[args.dataset_name]["imtopics"],
                lidartopics=config[args.dataset_name]["lidartopics"],
                odomtopics=config[args.dataset_name]["odomtopics"],
                img_process_func=eval(config[args.dataset_name]["img_process_func"]),
                lidar_process_func=eval(config[args.dataset_name]["lidar_process_func"]),
                odom_process_func=eval(config[args.dataset_name]["odom_process_func"]),
                rate=args.sample_rate,
                ang_offset=config[args.dataset_name]["ang_offset"]
            )
        except Exception as e:
            print(f"Error processing {bag_path}: {e}")
            continue

        if result is None:
            print(f"{bag_path} did not have the topics we were looking for. Skipping...")
            continue

        img_data, lidar_data, traj_data = result
        cut_trajs = filter_backwards_with_lidar(img_data, lidar_data, traj_data)

        for i, (img_i, lidar_i, traj_i) in enumerate(cut_trajs):
            traj_name_i = traj_name + f"_{i}"
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)
            os.makedirs(traj_folder_i, exist_ok=True)

            with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_i, f)

            for j, img in tqdm.tqdm(enumerate(img_i), desc="Save images"):
                img.save(os.path.join(traj_folder_i, f"{j}.jpg"))
            for j, lidar in tqdm.tqdm(enumerate(lidar_i), desc="Save lidar"):
                np.save(os.path.join(traj_folder_i, f"{j}.npy"), lidar)
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in process_config.yaml)",
        default="tartan_drive",
        required=True,
    )
    parser.add_argument("--input-dir", required=True,
                        help="Root folder of ROS2 bags")
    parser.add_argument("--output-dir", required=True,
                        help="Where to dump processed data")
    parser.add_argument("--num-trajs", type=int, default=-1,
                        help="Max number of bags to process")
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )
    args = parser.parse_args()
    main(args)
