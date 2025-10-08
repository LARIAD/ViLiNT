# topic names for ROS communication

# image obs topics
FRONT_IMAGE_TOPIC = "/usb_cam_front/image_raw"
REVERSE_IMAGE_TOPIC = "/usb_cam_reverse/image_raw"
#IMAGE_TOPIC = "/usb_cam/image_raw"
#IMAGE_TOPIC = "/zed_node/left_raw/image_raw_color" #To use zed camera online
#IMAGE_TOPIC = "/zed2/zed_node/rgb/image_rect_color" #To use barrakuda rosbags topic
#IMAGE_TOPIC = "/nerian/left/image_raw" #To use zed camera online
IMAGE_TOPIC = "/rgb" #To use zed camera online

# lidar obs
LIDAR_TOPIC = "/point_cloud"

# goal topic
GOAL_TOPIC = "/goal"

# exploration topics
SUBGOALS_TOPIC = "/subgoals"
GRAPH_NAME_TOPIC = "/graph_name"
WAYPOINT_TOPIC = "/waypoint"
REVERSE_MODE_TOPIC = "/reverse_mode"
SAMPLED_OUTPUTS_TOPIC = "/sampled_outputs"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"
SAMPLED_WAYPOINTS_GRAPH_TOPIC = "/sampled_waypoints_graph"
BACKTRACKING_IMAGE_TOPIC = "/backtracking_image"
FRONTIER_IMAGE_TOPIC = "/frontier_image"
SUBGOALS_SHAPE_TOPIC = "/subgoal_shape"
SAMPLED_ACTIONS_TOPIC = "/sampled_actions"
ANNOTATED_IMAGE_TOPIC = "/annotated_image"
CURRENT_NODE_IMAGE_TOPIC = "/current_node_image"
FLIP_DIRECTION_TOPIC = "/flip_direction"
TURNING_TOPIC = "/turning"
SUBGOAL_GEN_RATE_TOPIC = "/subgoal_gen_rate"
MARKER_TOPIC = "/visualization_marker_array"
VIZ_NAV_IMAGE_TOPIC = "/nav_image"
IMG_WAYPOINTS_TOPIC = "/img_waypoints"

# visualization topics
CHOSEN_SUBGOAL_TOPIC = "/chosen_subgoal"

# recorded ont the robot
ODOM_TOPIC = "/odom"
MODEL_STATES_TOPIC ="/gazebo/model_states"
BUMPER_TOPIC = "/mobile_base/events/bumper"
JOY_BUMPER_TOPIC = "/joy_bumper"

# collision detection
COLLISION_STATUS_TOPIC = "/collision_status"

# move the robot