#!/usr/bin/env python3
import os
import yaml
import numpy as np
from typing import Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Bool

from deployment_utils import clip_angle
from topics_names import WAYPOINT_TOPIC, REACHED_GOAL_TOPIC, ODOM_TOPIC

# Load robot constants
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/robot.yaml')
with open(CONFIG_PATH, 'r') as f:
    robot_config = yaml.safe_load(f)

MAX_V         = robot_config["max_v"]
MAX_W         = robot_config["max_w"]
VEL_TOPIC     = robot_config["vel_navi_topic"]
DT            = 1.0 / robot_config["frame_rate"]
RATE          = 100              # Hz

# Fallback P-control gains & thresholds (used when no new waypoint arrives)
K_V     = robot_config.get("k_v", 0.5)     # linear gain [m/s per m]
K_W     = robot_config.get("k_w", 1.5)     # angular gain [rad/s per rad]
KD_V    = robot_config.get("kd_v", 0.1)   # linear D gain [m/s per m/s]
KD_W    = robot_config.get("kd_w", 0.2)   # angular D gain [rad/s per rad/s]
K_YAW   = robot_config.get("k_yaw", 0.5)   # yaw-only gain when at position
POS_EPS = robot_config.get("pos_eps", 0.05)  # position tolerance [m]
YAW_EPS = robot_config.get("yaw_eps", 0.087) # yaw tolerance [rad] (~5 deg)

EPS           = 1e-8

# --- Helpers for fallback P-control when no new waypoint is received ---
def _rot2d(theta: float, vec: np.ndarray) -> np.ndarray:
    """Rotate 2D vector by angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return R @ vec


class PDControllerNode(Node):
    def __init__(self):
        super().__init__('pd_controller')
        self.reached_goal  = False
        self.reverse_mode  = False

        self.current_pose = None   # (x, y, yaw) in odom
        self.current_twist = (0.0, 0.0)  # (v, w) from odom

        # Latched goal for fallback P-control
        self.target_error: np.ndarray | None = None  # stores last received waypoint (2 or 4 floats)
        self.have_goal: bool = False                 # whether we have an active goal to pursue
        self.last_cmd_vw: Tuple[float, float] = (0.0, 0.0)  # last (v,w) we published
        self.last_waypoint_raw: np.ndarray | None = None
        self._waiting_for_odom_logged = False

        # publishers & subscribers
        self.create_subscription(
            Float32MultiArray,
            WAYPOINT_TOPIC,
            self.callback_drive,
            1
        )
        self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.callback_odom,
            10
        )
        self.create_subscription(
            Bool,
            REACHED_GOAL_TOPIC,
            self.callback_reached_goal,
            1
        )
        self.vel_pub = self.create_publisher(Twist, VEL_TOPIC, 1)

        # timer at 100 Hz
        self.timer = self.create_timer(1.0 / RATE, self.on_timer)

        self.get_logger().info('PD Controller node started, waiting for waypoints…')

    def callback_drive(self, msg: Float32MultiArray):
        wp = np.asarray(msg.data, dtype=np.float32)

        # latch waypoint ALWAYS
        self.last_waypoint_raw = wp
        self.have_goal = True

        # if odom already available, compute odom-frame target now
        if self.current_pose is not None:
            self.target_error = self._wp_to_odom_target(wp, self.current_pose)
            self._waiting_for_odom_logged = False

    def callback_reached_goal(self, msg: Bool):
        self.reached_goal = msg.data
    
    def callback_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            float(clip_angle(yaw))
        )
        self.current_twist = (
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        )

        # NEW: if we already have a waypoint but couldn’t compute target yet, do it now
        if self.have_goal and self.target_error is None and self.last_waypoint_raw is not None:
            self.target_error = self._wp_to_odom_target(self.last_waypoint_raw, self.current_pose)
            self._waiting_for_odom_logged = False

    def _wp_to_odom_target(self, wp: np.ndarray, pose: tuple) -> np.ndarray:
        """Convert robot-frame waypoint (dx,dy[,hx,hy]) to odom-frame target pose [x,y,yaw]."""
        x, y, yaw = pose
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]], dtype=np.float32)
        dx, dy = wp[0], wp[1]
        gx, gy = np.array([x, y], dtype=np.float32) + R @ np.array([dx, dy], dtype=np.float32)
        if wp.size == 4:
            heading_yaw = np.arctan2(wp[3], wp[2])
            gyaw = clip_angle(yaw + heading_yaw)
        else:
            gyaw = yaw
        return np.array([gx, gy, gyaw], dtype=np.float32)

    def on_timer(self):
        vel = Twist()

        if self.reached_goal:
            self.last_cmd_vw = (0.0, 0.0)
            self.vel_pub.publish(vel)
            self.get_logger().info('Reached goal – stopping and shutting down.')
            rclpy.shutdown()
            return

        # Need odom first
        if self.current_pose is None:
            if not self._waiting_for_odom_logged:
                self.get_logger().info("Waiting for odometry...")
                self._waiting_for_odom_logged = True
            self.vel_pub.publish(vel)
            return

        # Need a latched waypoint; if we have one but target_error not computed yet, compute it
        if self.have_goal and self.target_error is None and self.last_waypoint_raw is not None:
            self.target_error = self._wp_to_odom_target(self.last_waypoint_raw, self.current_pose)

        # Still no goal => publish zero
        if (not self.have_goal) or (self.target_error is None):
            self.last_cmd_vw = (0.0, 0.0)
            self.vel_pub.publish(vel)
            return

        # --- rest of your control law unchanged ---
        x, y, yaw = self.current_pose
        gx, gy, gyaw = self.target_error
        dx = gx - x
        dy = gy - y
        ex, ey = _rot2d(-yaw, np.array([dx, dy], dtype=np.float32))
        yaw_err = clip_angle(gyaw - yaw)

        dist = float(np.hypot(ex, ey))
        if dist <= POS_EPS and abs(yaw_err) <= YAW_EPS:
            self.have_goal = False
            self.target_error = None
            self.last_cmd_vw = (0.0, 0.0)
            self.vel_pub.publish(vel)
            return

        cur_v, cur_w = self.current_twist
        angle_to_target = np.arctan2(ey, ex) if dist > EPS else 0.0
        v = K_V * dist + KD_V * (0.0 - cur_v)
        w = K_W * angle_to_target + K_YAW * yaw_err + KD_W * (0.0 - cur_w)
        v = float(np.clip(v, 0.0, MAX_V))
        w = float(np.clip(w, -MAX_W, MAX_W))
        self.last_cmd_vw = (v, w)

        if self.reverse_mode:
            v, w = self.last_cmd_vw
            self.last_cmd_vw = (-v, w)

        vel.linear.x = float(np.clip(self.last_cmd_vw[0], -MAX_V, MAX_V))
        vel.angular.z = float(np.clip(self.last_cmd_vw[1], -MAX_W, MAX_W))
        print("Published vel:", vel.linear.x, vel.angular.z)
        self.vel_pub.publish(vel)


def main(args=None):
    rclpy.init(args=args)
    node = PDControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
