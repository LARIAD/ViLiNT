import os
import sys
from pathlib import Path
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import yaml

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped

deployment_dir = Path(__file__).resolve().parent
project_root = deployment_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Other imports
from deployment_utils import msg_to_pil, to_numpy, transform_images, load_model, process_lidar, get_robot_config, get_goal_direction, select_mode
from mnt_train.training.train_utils import get_action
import argparse
import time
from publish_imgwaypoints import pub_waypoints


#from test_utils import ErrorMetricsCalculator, get_final_position

# UTILS
from topics_names import (ODOM_TOPIC,
                        IMAGE_TOPIC,
                        LIDAR_TOPIC,
                        WAYPOINT_TOPIC,
                        SAMPLED_ACTIONS_TOPIC,
                        IMG_WAYPOINTS_TOPIC,
                        COLLISION_STATUS_TOPIC, 
                        GOAL_TOPIC)

# CONSTANTS
ROBOT_CONFIG_PATH = project_root / "deployment" / "config" / "robot.yaml"
MODEL_CONFIG_PATH = project_root / "deployment" / "config" / "models.yaml"

class ExplorationNode(Node):
    def __init__(self):
        super().__init__('exploration_node')

        # load model parameters
        with open(MODEL_CONFIG_PATH, "r") as f:
            self.model_config = yaml.safe_load(f)

        model_config_path = self.model_config[args.model]["config_path"]
        with open(project_root / model_config_path, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_size = self.model_params["context_size"]
        if self.model_params["model_type"] == "vilint":
            self.context_size_lidar = self.model_params["context_size_li"]
            self.diffusion_dim = 3
            self.use_vilint = True
        else:
            self.diffusion_dim = 2
            self.use_vilint = False
            
        with open(ROBOT_CONFIG_PATH, "r") as f:
            self.robot_config = yaml.safe_load(f)
        self.MAX_V = self.robot_config["max_v"]
        self.MAX_W = self.robot_config["max_w"]
        self.RATE = self.robot_config["frame_rate"]

        # Initialize global variables
        self.context_queue = []
        self.context_queue_lidar = []
        self.position_tensor = torch.zeros(3)
        self.orientation_tensor = torch.zeros(4)

        # Create publishers and subscribers
        self.create_subscription(Image, IMAGE_TOPIC, self.callback_obs, 5)
        self.create_subscription(PointCloud2, LIDAR_TOPIC, self.callback_lidar, 1)
        self.create_subscription(Odometry, ODOM_TOPIC, self.odometry_callback, 1)
        self.create_subscription(PoseStamped, GOAL_TOPIC, self.goal_callback, 1)

        self.waypoint_pub = self.create_publisher(Float32MultiArray, WAYPOINT_TOPIC, 1)
        self.sampled_actions_pub = self.create_publisher(Float32MultiArray, SAMPLED_ACTIONS_TOPIC, 1)
        self.collision_status_pub = self.create_publisher(Float32MultiArray, COLLISION_STATUS_TOPIC, 5)
        self.img_waypoints_pub = self.create_publisher(Image, IMG_WAYPOINTS_TOPIC, 10)

        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # load model weights
        ckpth_path = project_root / self.model_config[args.model]["ckpt_path"]
        if os.path.exists(ckpth_path):
            self.get_logger().info(f"Loading model from {ckpth_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {ckpth_path}")
        self.model = load_model(
            ckpth_path,
            self.model_params,
            self.device,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.physics = get_robot_config(self.robot_config, self.device)

        self.num_diffusion_iters = self.model_params["num_diffusion_iters"]
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_params["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.goal_position = torch.tensor([20.0, 0]).to(self.device)  # scenario 3 : (5.8, 8.6)

        self.image_waypoint_spacing = 1
        self.image_count = 0
        self.lidar_waypoint_spacing = 1
        self.lidar_count = 0

        # Input masks
        self.goal_mask = torch.zeros(1).long().to(self.device)
        self.image_mask = torch.ones(1).long().to(self.device) if self.model_config[args.model].get("mask_image") else None
        self.lidar_mask = torch.ones(1).long().to(self.device) if self.model_config[args.model].get("mask_lidar") else None
        print("Goal mask: ", self.goal_mask, " Image mask: ", self.image_mask, " Lidar mask: ", self.lidar_mask)

        # Heuristics
        self.use_exploration = self.model_config[args.model].get("use_exploration", False)
        self.use_distance_to_goal_heuristic = self.model_config[args.model].get("use_distance_to_goal_heuristic", False)
        self.use_collision_heuristic = self.model_config[args.model].get("use_collision_heuristic", False)

        # Collision risk
        self.approaching_col = False
        self.exploration_count = 0

        # Run the model
        self.create_timer(0.01, self.timer_callback)

    def callback_obs(self, msg):
        obs_img = msg_to_pil(msg)
        if self.context_size is not None:
            self.image_count += 1
            if len(self.context_queue) < self.context_size + 1 and self.image_count % self.image_waypoint_spacing == 0:
                self.context_queue.append(obs_img)
            elif self.image_count % self.image_waypoint_spacing == 0:
                self.context_queue.pop(0)
                self.context_queue.append(obs_img)

    def callback_lidar(self, msg):
        if self.model_params["model_type"]=="vilint":
            pc_gen = pc2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            )
            x = pc_gen['x']
            y = pc_gen['y']
            z = pc_gen['z']
            point_cloud_array = np.stack([x, y, z], axis=-1).astype(np.float32)
            if self.context_size_lidar is not None:
                self.lidar_count += 1
                if len(self.context_queue_lidar) < self.context_size_lidar + 1 and self.lidar_count % self.lidar_waypoint_spacing == 0:
                    self.context_queue_lidar.append(point_cloud_array)
                elif self.lidar_count % self.lidar_waypoint_spacing == 0:
                    self.context_queue_lidar.pop(0)
                    self.context_queue_lidar.append(point_cloud_array)
        else:
            return

    def odometry_callback(self, msg):
        position = msg.pose.pose.position
        self.position_tensor[0] = position.x
        self.position_tensor[1] = position.y
        self.position_tensor[2] = position.z

        orientation = msg.pose.pose.orientation
        self.orientation_tensor[0] = orientation.x
        self.orientation_tensor[1] = orientation.y
        self.orientation_tensor[2] = orientation.z
        self.orientation_tensor[3] = orientation.w
    
    def goal_callback(self, msg):
        position = msg.pose.position
        self.goal_position[0] = position.x
        self.goal_position[1] = position.y
        print("Updated goal position: ", self.goal_position)
    
    def timer_callback(self):
        waypoint_msg = Float32MultiArray()
        if (len(self.context_queue) > self.model_params["context_size"] and ((len(self.context_queue_lidar) > self.model_params["context_size_li"]) if self.use_vilint else True)):

            obs_images = transform_images(self.context_queue, self.model_params["image_size"], center_crop=False)
            obs_images.to(self.device)
            if self.use_vilint and self.model_params["pc_encoder"] is not None:
                obs_lidar = process_lidar(self.context_queue_lidar, self.model.vision_encoder.pc_encoder_channels, self.device, self.model_params["pc_encoder"])
            else:
                obs_lidar = None
            obs_images = obs_images.to(self.device)
            
            if args.data:
                goal_coord = self.goal_position
                goal_coord = goal_coord.unsqueeze(0)
            elif self.position_tensor is not None:
                goal_coord = get_goal_direction(self.position_tensor, self.goal_position, self.orientation_tensor).to(self.device)
                goal_coord = goal_coord.unsqueeze(0)
                print("Goal direction vector: ", goal_coord)
            else: 
                goal_coord = torch.randn((1, 2)).to(self.device)                

            # infer action
            with torch.no_grad():
                # encoder vision features
                if self.use_vilint:
                    obs_feats = self.model('vision_encoder', obs_img=obs_images, obs_pc=obs_lidar, goal_coord=goal_coord,
                                            physics=self.physics.unsqueeze(0), input_goal_mask=self.goal_mask, input_image_mask=self.image_mask, input_lidar_mask=self.lidar_mask)
                    scenes = obs_feats["scene"]          # [1,D]
                    lidar_ctx = obs_feats["lidar_ctx"]   # [1,K,D]
                elif self.model_params["use_2d_goal"]:
                    obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=None, goal_coord=goal_coord, input_goal_mask=None)
                else:
                    fake_goal = torch.randn((1, 3, *self.model_params["image_size"])).to(self.device)
                    obs_cond = self.model('vision_encoder', obs_img=obs_images, goal_img=fake_goal, goal_coord=goal_coord, input_goal_mask=self.goal_mask)     

                if self.use_vilint:
                    obs_cond = scenes.repeat(args.num_samples, 1)
                else:
                    # (B, obs_horizon * obs_dim)
                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(args.num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                
                # initialize action from Gaussian noise
                if self.use_vilint:
                    noisy_action = torch.randn(
                        (args.num_samples, self.model_params["len_traj_pred"], 2+2*int(self.model_params ["learn_angle"])), device=self.device)
                else:   
                    noisy_action = torch.randn(
                        (args.num_samples, self.model_params["len_traj_pred"], 2), device=self.device)
                naction = noisy_action

                if self.use_vilint and self.model_params["use_imle"]:
                    start_time = time.time()
                    naction = self.model("policy_net", global_cond=obs_cond, sample=noisy_action)
                    print("time elapsed:", time.time() - start_time)
                else:
                    # init scheduler
                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    start_time = time.time()
                    for k in self.noise_scheduler.timesteps[:]:
                        # predict noise
                        noise_pred = self.model(
                            'noise_pred_net',
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond,
                        )

                        # inverse diffusion step (remove noise)
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                    print("time elapsed:", time.time() - start_time)
            naction = to_numpy(get_action(naction))
            if self.use_vilint:
                lidar_rep = lidar_ctx.repeat(args.num_samples, 1, 1)
                scene_rep = scenes.repeat(args.num_samples, 1)
                width_rep = self.physics[:2].unsqueeze(0).repeat(args.num_samples, 1)
                col_status_logits = self.model("collision_pred_net", lidar_ctx=lidar_rep, traj=torch.from_numpy(naction[...,:2]).to(self.device), width=width_rep, scene=scene_rep)
                col_dist = to_numpy(col_status_logits)
            raw_naction = naction
            
            sampled_actions_msg = Float32MultiArray()
            arr32 = np.concatenate([
                np.array([0], dtype=np.float32),      # prepend a zero of correct type
                naction.flatten().astype(np.float32)  # flatten & cast
            ])

            # 2) convert to Python list
            py_list:    list[float] = arr32.tolist()
            sampled_actions_msg.data = py_list
            self.sampled_actions_pub.publish(sampled_actions_msg)

            goal_coord = to_numpy(goal_coord)
            if self.use_vilint and self.use_collision_heuristic:
                collision_status_msg = Float32MultiArray()
                value: list[float] = col_dist[0].tolist()
                collision_status_msg.data = value
                self.collision_status_pub.publish(collision_status_msg)
                naction, col_dist, d_goal, clear = select_mode(naction, col_dist, mode="min", goal_coord=goal_coord)
                print(" Clear: ", clear, " Distance to goal: ", d_goal)          
            elif self.use_distance_to_goal_heuristic:
                dist_2goal = np.linalg.norm(to_numpy(self.goal_position) - naction[..., :2])
                idx = np.argmin(dist_2goal)
                naction = naction[idx]   
            else:
                naction = naction[0] # change this based on heuristic
            
            chosen_waypoint = naction[args.waypoint]
            print("Chosen waypoint: ", chosen_waypoint)
            
            if self.use_exploration:
                if clear < 1.0:
                    self.goal_mask = torch.ones(1).long().to(self.device)
                elif clear > 1.5:
                    self.goal_mask = torch.zeros(1).long().to(self.device)

            if self.model_params["normalize"]:
                chosen_waypoint[:2] *= (self.MAX_V / self.RATE)
            
            if args.imgwaypoints:
                imgwp = pub_waypoints(self.context_queue[-1], raw_naction, args.waypoint)
                br = CvBridge()
                self.img_waypoints_pub.publish(br.cv2_to_imgmsg(imgwp))

        
            waypt:  list[float] = chosen_waypoint.tolist()
            waypoint_msg.data = waypt
            self.waypoint_pub.publish(waypoint_msg)
            print("Published waypoint: ", chosen_waypoint)
            


def main(args: argparse.Namespace):
    global context_size
    global context_size_lidar

    rclpy.init(args=sys.argv)
    node = ExplorationNode()

    # traj_selector = ModeSelector()

    print("Registered with master node. Waiting for image observations...")
    # EXPLORATION MODE

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code to run GNM DIFFUSION EXPLORATION on the locobot")
    parser.add_argument(
        "--model",
        "-m",
        default="vilint",
        type=str,
        help="model name (hint: check ../config/models.yaml) (default: vilint)",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=4, # close waypoints exihibit straight line motion (the middle waypoint is a good default)
        type=int,
        help=f"""index of the waypoint used for navigation (between 0 and 4 or 
        how many waypoints your model predicts) (default: 2)""",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help=f"Number of actions sampled from the exploration model (default: 8)",
    )
    parser.add_argument(
        "--imgwaypoints",
        action='store_true',
        help=f"publish image topic to show predicted waypoints",
    )
    parser.add_argument(
        "--data",
        action="store_true",
        help="Change goal definition when using vilint to make data."
    )
    args = parser.parse_args()
    main(args)


