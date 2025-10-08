import csv
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

ENV_CONFIGS = {
    "easy": "/workspace/simulation/assets/easy_offroad_v4_3cams_headless_newTF.usd",
    "difficult": "/workspace/simulation/assets/difficult_offroad_v8_3cams_headless_newTF.usd",
    "scenario2": "/workspace/barakuda_isaac_sim/assets/scenario2.usd",
}
ENV_SEQUENCE = ["easy", "difficult", "scenario2"]
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = REPO_ROOT / "deployment" / "config" / "models.yaml"
RESULTS_YAML_PATH = Path(__file__).resolve().parent / "evaluation_results.yaml"
RESULTS_CSV_PATH = Path(__file__).resolve().parent / "evaluation_results.csv"

ROBOT_PRIM_PATH = "/map/husky" 

# Goal sampling bounds (meters) in world frame
GOAL_THETA_BOUNDS = (-np.pi/2, np.pi/2)
GOAL_Z = 0.0

PHYSICS_DT = 1.0 / 600.0
RENDERING_DT = 1.0 / 60.0

MAX_EPISODE_SECONDS = 600.0
GOAL_TOL_M = 1.0

# Failure criteria
MAX_TILT_DEG = 60.0          # fail if roll/pitch exceed this
STUCK_WINDOW_SECONDS = 400.0   # fail if not making progress for this long
MIN_PROGRESS_M = 0.05        # minimal distance improvement over the stuck window

N_EPISODES = 100
RNG_SEED = 0

# ------------------------- ROS2 I/O -------------------------
GOAL_TOPIC = "/goal"
ODOM_TOPIC = "/odom"
GOAL_FRAME_ID = "map"  # change if your setup uses another fixed frame
CMD_VEL_TOPIC = "/cmd_vel"
CMD_VEL_WAIT_SECONDS = 10.0          # wall-clock timeout waiting for first /cmd_vel

GOAL_PUB_PERIOD_SECONDS = 1.0        # keep publishing goal periodically
POST_RESET_SETTLE_SECONDS = 1.0      # allow reset + ROS graph to settle before evaluating success
MIN_SUCCESS_CHECK_SECONDS = 0.5      # ignore immediate stale "success" detections right after reset

# ------------------------- Collision failure (PhysX contact reports) -------------------------
COLLISION_FAIL_ON_CONTACT = True
# Threshold on contact impulse for reporting (0.0 reports all contacts; increase to ignore tiny touches)
COLLISION_REPORT_THRESHOLD = 0.0
# Actors to ignore as "environment collisions" (adjust to your USD)
COLLISION_IGNORE_ACTOR_PREFIXES = (
    "/map/heightmap",
    "/map/GeneratedForest/Terrain_Parent",
)


def rot_to_roll_pitch_deg(rot3x3):
    """
    rot3x3 is row-major 3x3m or quaternion.
    Compute roll/pitch from rotation matrix (ZYX convention).
    """
    if rot3x3.shape == (4,):
        r = R.from_quat([rot3x3[0], rot3x3[1], rot3x3[2], rot3x3[3]])  # xyzw
        rot3x3 = r.as_matrix()
    r00, r01, r02 = rot3x3[0]
    r10, r11, r12 = rot3x3[1]
    r20, r21, r22 = rot3x3[2]

    # pitch = asin(-r20)
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    # roll = atan2(r21, r22)
    roll = math.atan2(r21, r22)
    return math.degrees(roll), math.degrees(pitch)


def find_robot_prim_path(stage):
    """
    Heuristic: pick first prim that looks like an articulation root or is named like a robot.
    If this fails, set ROBOT_PRIM_PATH explicitly.
    """
    from pxr import UsdPhysics

    candidates = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        path = prim.GetPath().pathString
        name = prim.GetName().lower()

        score = 0
        if "robot" in name or "husky" in name or "jackal" in name:
            score += 3
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            score += 5
        if prim.GetTypeName() in ("Xform", "Scope"):
            score += 0

        if score > 0:
            candidates.append((score, path))

    if not candidates:
        raise RuntimeError(
            "Could not auto-detect robot prim. Set ROBOT_PRIM_PATH explicitly."
        )

    candidates.sort(reverse=True)
    return candidates[0][1]


def set_goal_marker(stage, goal_xyz, prim_path="/World/Goal"):
    """
    Create or move a visible marker (a sphere) to the goal location.
    Safe in headless (useful if you later run non-headless).
    """
    from pxr import UsdGeom, Gf

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        sphere = UsdGeom.Sphere.Define(stage, prim_path)
        sphere.CreateRadiusAttr(0.2)
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*goal_xyz))
    else:
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        if not ops:
            xform.AddTranslateOp().Set(goal_xyz)
        else:
            # best-effort: set first translate op
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(goal_xyz)
                    break
            else:
                xform.AddTranslateOp().Set(goal_xyz)


# ------------------------- ROS2 helpers -------------------------

class RosInterface:
    """Small ROS2 wrapper: publishes goal and detects first /cmd_vel message."""

    def __init__(self, node, rclpy_mod, PoseStamped, Twist, Odometry, goal_pub):
        self.node = node
        self._rclpy = rclpy_mod
        self._PoseStamped = PoseStamped
        self._Twist = Twist
        self._Odometry = Odometry
        self.goal_pub = goal_pub

        self.got_cmd_vel = False
        self.last_cmd_vel_wall_time = None

        self._last_goal = None  # (goal_xyz, frame_id)
        self._goal_timer = None

        self.position_tensor = np.zeros(3)
        self.orientation_tensor = np.zeros(4)
        self.last_odom_wall_time = None

        # Subscribe to /cmd_vel to know when a controller is active
        self._cmd_vel_sub = self.node.create_subscription(
            self._Twist, CMD_VEL_TOPIC, self._on_cmd_vel, 10
        )
        self.node.create_subscription(self._Odometry, ODOM_TOPIC, self.odometry_callback, 1)

    def _on_cmd_vel(self, _msg):
        self.got_cmd_vel = True
        self.last_cmd_vel_wall_time = time.perf_counter()

    def spin_once(self, timeout_sec: float = 0.0):
        # Drives timers + subscriptions
        self._rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def set_goal(self, goal_xyz, frame_id: str = GOAL_FRAME_ID):
        self._last_goal = (goal_xyz, frame_id)
        if self._goal_timer is None:
            self._goal_timer = self.node.create_timer(
                GOAL_PUB_PERIOD_SECONDS, self._publish_last_goal
            )
        # publish immediately too
        self._publish_last_goal()

    def clear_goal(self):
        self._last_goal = None
        if self._goal_timer is not None:
            self._goal_timer.cancel()
            try:
                self._goal_timer.destroy()
            except Exception:
                pass
            self._goal_timer = None

    def _publish_last_goal(self):
        if self._last_goal is None:
            return
        goal_xyz, frame_id = self._last_goal

        msg = self._PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(goal_xyz[0])
        msg.pose.position.y = float(goal_xyz[1])
        msg.pose.position.z = float(goal_xyz[2])
        msg.pose.orientation.w = 1.0

        self.goal_pub.publish(msg)
    
    def odometry_callback(self, msg):
        self.last_odom_wall_time = time.perf_counter()
        position = msg.pose.pose.position
        self.position_tensor[0] = position.x
        self.position_tensor[1] = position.y
        self.position_tensor[2] = position.z

        orientation = msg.pose.pose.orientation
        self.orientation_tensor[0] = orientation.x
        self.orientation_tensor[1] = orientation.y
        self.orientation_tensor[2] = orientation.z
        self.orientation_tensor[3] = orientation.w


def init_ros():
    """Initialize ROS2 node + goal publisher. Raises RuntimeError if ROS2 Python libs are missing."""
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped, Twist
        from nav_msgs.msg import Odometry
    except Exception as e:
        raise RuntimeError(
            "ROS2 Python libraries not available in this environment (rclpy/geometry_msgs). "
            "Make sure your Isaac Sim process has ROS2 Python packages available."
        ) from e

    rclpy.init(args=None)
    node = rclpy.create_node("isaac_goal_evaluator")
    goal_pub = node.create_publisher(PoseStamped, GOAL_TOPIC, 10)

    ros_if = RosInterface(node=node, rclpy_mod=rclpy, PoseStamped=PoseStamped, Twist=Twist, Odometry=Odometry, goal_pub=goal_pub)
    return ros_if, rclpy



# ------------------------- Collision monitor (PhysX contact reports) -------------------------

class CollisionMonitor:
    """Detects contacts involving the robot using PhysX Contact Report API."""

    def __init__(self, stage, robot_root_path: str):
        self.stage = stage
        self.robot_root_path = robot_root_path

        self.collided = False
        self.last_other_actor = None
        self.last_other_collider = None
        self.last_event_type = None

        self._sub = None

        self._setup_contact_report()

    def reset(self):
        self.collided = False
        self.last_other_actor = None
        self.last_other_collider = None
        self.last_event_type = None

    def shutdown(self):
        if self._sub is not None:
            try:
                self._sub.unsubscribe()
            except Exception:
                pass
            self._sub = None

    def _setup_contact_report(self):
        from pxr import PhysxSchema, PhysicsSchemaTools
        from omni.physx import get_physx_simulation_interface

        robot_prim = self.stage.GetPrimAtPath(self.robot_root_path)
        if not robot_prim or not robot_prim.IsValid():
            raise RuntimeError(f"Robot prim path not valid for collision monitor: {self.robot_root_path}")

        # Enable contact reporting on the robot prim (body or articulation root)
        contact_api = PhysxSchema.PhysxContactReportAPI.Apply(robot_prim)
        # Report threshold is based on contact impulse
        try:
            contact_api.CreatePhysxContactReportThresholdAttr().Set(float(COLLISION_REPORT_THRESHOLD))
        except Exception:
            # Older schemas sometimes use a different attribute name
            try:
                contact_api.CreateThresholdAttr().Set(float(COLLISION_REPORT_THRESHOLD))
            except Exception:
                pass

        # Subscribe to global contact report events; we filter in the callback
        self._decode = PhysicsSchemaTools.intToSdfPath
        self._sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)

    def _is_ignored_actor(self, actor_path: str) -> bool:
        for pfx in COLLISION_IGNORE_ACTOR_PREFIXES:
            if actor_path.startswith(pfx):
                return True
        return False

    def _on_contact_report_event(self, contact_headers, contact_data):
        # Called after each simulation step.
        if self.collided:
            return

        for h in contact_headers:
            try:
                actor0 = str(self._decode(h.actor0))
                actor1 = str(self._decode(h.actor1))
                collider0 = str(self._decode(h.collider0))
                collider1 = str(self._decode(h.collider1))
            except Exception:
                continue

            # Only treat contacts involving the robot as collisions
            a0_is_robot = actor0.startswith(self.robot_root_path)
            a1_is_robot = actor1.startswith(self.robot_root_path)
            if not (a0_is_robot or a1_is_robot):
                continue

            other_actor = actor1 if a0_is_robot else actor0
            other_collider = collider1 if a0_is_robot else collider0

            # Filter ground/terrain etc.
            if self._is_ignored_actor(other_actor):
                continue

            # Ignore self-collisions
            if other_actor.startswith(self.robot_root_path):
                continue

            # Flag collision
            self.collided = True
            self.last_other_actor = other_actor
            self.last_other_collider = other_collider
            self.last_event_type = getattr(h, "type", None)
            return

# ------------------------- episode logic -------------------------

@dataclass
class EpisodeResult:
    success: bool
    steps: int
    sim_seconds: float
    reason: str
    spl_contribution: float

def load_results_yaml(path: Path):
    if not path.exists():
        return {"runs": []}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {"runs": []}
    if "runs" not in data or not isinstance(data["runs"], list):
        data["runs"] = []
    return data


def write_results_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_active_model_metadata(path: Path):
    if not path.exists():
        print(f"Warning: model config not found at {path}; result metadata will be partial.")
        return None, {}

    with path.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    if not isinstance(config_data, dict) or not config_data:
        print(f"Warning: no model entries found in {path}; result metadata will be partial.")
        return None, {}

    model_name, model_config = next(iter(config_data.items()))
    if not isinstance(model_config, dict):
        print(
            f"Warning: model entry '{model_name}' in {path} is not a mapping; "
            "result metadata will be partial."
        )
        return model_name, {}

    return model_name, model_config


def _format_metric(value):
    if value is None:
        return ""
    return f"{float(value):.3f}"


def build_results_csv_row(run_record):
    env_records = run_record.get("environments", [])
    env_names = []
    result_parts = []
    success_rates = []
    collision_rates = []
    spl_values = []

    for env in env_records:
        env_name = str(env.get("environment", ""))
        env_names.append(env_name)

        success_rate = env.get("success_rate")
        collision_rate = env.get("collision_rate")
        spl = env.get("spl")

        if success_rate is not None:
            success_rates.append(float(success_rate))
        if collision_rate is not None:
            collision_rates.append(float(collision_rate))
        if spl is not None:
            spl_values.append(float(spl))

        result_text = (
            f"{env_name}: SR={_format_metric(success_rate)}, "
            f"CR={_format_metric(collision_rate)}"
        )
        if spl is not None:
            result_text += f", SPL={_format_metric(spl)}"
        result_parts.append(result_text)

    avg_sr = f"{sum(success_rates) / len(success_rates):.3f}" if success_rates else ""
    avg_cr = f"{sum(collision_rates) / len(collision_rates):.3f}" if collision_rates else ""
    avg_spl = f"{sum(spl_values) / len(spl_values):.3f}" if spl_values else ""

    ckpt_path = run_record.get("ckpt_path")
    model_label = Path(ckpt_path).name if ckpt_path else str(run_record.get("model_name", ""))

    return {
        "model": model_label,
        "masks": (
            f"lidar={run_record.get('mask_lidar')}, "
            f"image={run_record.get('mask_image')}"
        ),
        "dt": (
            f"physics={run_record.get('physics_dt', '')}, "
            f"rendering={run_record.get('rendering_dt', '')}"
        ),
        "heuristics": (
            f"exploration={run_record.get('use_exploration')}, "
            f"distance_to_goal={run_record.get('use_distance_to_goal_heuristic')}, "
            f"collision={run_record.get('use_collision_heuristic')}"
        ),
        "envs": "; ".join(env_names),
        "results": " | ".join(result_parts),
        "avg_SR": avg_sr,
        "avg_CR": avg_cr,
        "avg_SPL": avg_spl,
        "notes": "",
    }


def append_results_csv(path: Path, run_record):
    fieldnames = [
        "model",
        "masks",
        "dt",
        "heuristics",
        "envs",
        "results",
        "avg_SR",
        "avg_CR",
        "avg_SPL",
        "notes",
    ]
    row = build_results_csv_row(run_record)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def sample_goal(rng: random.Random, env: str = "difficult"):
    if env == "difficult":
        goals_list = [[7.5, -7.75, GOAL_Z], [12.65, -16.71, GOAL_Z], [24.8, -4.5, GOAL_Z],
                    [33.8, -16.88, GOAL_Z], [46.94, -7.75, GOAL_Z], [47, 10.9, GOAL_Z],
                    [23.44, 14.01, GOAL_Z], [21.92, 4.16, GOAL_Z], [10.3, 14.66, GOAL_Z],
                    [15.62, 6.57, GOAL_Z]]
        return goals_list[rng.randint(0, len(goals_list)-1)]
    elif env == "easy":
        goals_list = [[15.7, -7.15, GOAL_Z], [25.9, -14.7, GOAL_Z], [22, -0.9, GOAL_Z], 
                      [33.6, 2.0, GOAL_Z], [43.3, -12.5, GOAL_Z], [44.7, 10.6, GOAL_Z], 
                      [33.8, 16.3, GOAL_Z], [18.7, 17.8, GOAL_Z], [11.9, 5.17, GOAL_Z]]
        return goals_list[rng.randint(0, len(goals_list)-1)]
    elif env == "scenario2":
        (theta_min, theta_max) = GOAL_THETA_BOUNDS
        theta = rng.uniform(theta_min, theta_max)
        r = rng.uniform(18.0, 20.0)
        return (r*np.cos(theta), r*np.sin(theta), GOAL_Z)


def run_episode(sim, stage, robot_path: str, goal_xyz, ros_if: RosInterface, col: CollisionMonitor) -> EpisodeResult:
    max_steps = int(MAX_EPISODE_SECONDS / PHYSICS_DT)

    # progress / stuck tracking
    window_steps = max(1, int(STUCK_WINDOW_SECONDS / PHYSICS_DT))
    dist_hist = []

    # Render periodically to drive OmniGraph nodes that tick on playback/render
    steps_per_render = max(1, int(round(RENDERING_DT / PHYSICS_DT)))

    # Start playing (physics + timeline)
    sim.play()
    try:
        import omni.timeline
        omni.timeline.get_timeline_interface().play()
    except Exception:
        pass

    # Reset collision flag for this episode
    if col is not None:
        col.reset()

    # Publish goal and wait for first /cmd_vel (controller becoming active)
    ros_if.got_cmd_vel = False
    ros_if.set_goal(goal_xyz, frame_id=GOAL_FRAME_ID)
    episode_start_wall_time = time.perf_counter()
    odom_timestamp_before_goal = ros_if.last_odom_wall_time

    start_position = ros_if.position_tensor.copy()
    previous_position = start_position.copy()
    path_length = 0.0

    def update_path_length():
        nonlocal previous_position, path_length
        current_position = ros_if.position_tensor.copy()
        path_length += float(np.linalg.norm(current_position - previous_position))
        previous_position = current_position

    def compute_spl_contribution(success: bool) -> float:
        if not success:
            return 0.0
        start_to_goal = float(np.linalg.norm(np.asarray(goal_xyz) - start_position))
        if start_to_goal <= 1e-8 or path_length <= 1e-8:
            return 0.0
        return start_to_goal / path_length

    # Let the reset settle and wait for at least one fresh odometry message after the new goal is published.
    settle_steps = int(POST_RESET_SETTLE_SECONDS / PHYSICS_DT)
    for _ in range(settle_steps):
        do_render = (_ % steps_per_render) == 0
        sim.step(render=do_render)
        ros_if.spin_once(timeout_sec=0.0)
        update_path_length()
        if (
            ros_if.last_odom_wall_time is not None
            and (
                odom_timestamp_before_goal is None
                or ros_if.last_odom_wall_time > odom_timestamp_before_goal
            )
        ):
            start_position = ros_if.position_tensor.copy()
            previous_position = start_position.copy()
            path_length = 0.0
            episode_start_wall_time = time.perf_counter()
            break

    wait_start = time.perf_counter()
    waited_steps = 0
    while (not ros_if.got_cmd_vel) and ((time.perf_counter() - wait_start) < CMD_VEL_WAIT_SECONDS):
        do_render = (waited_steps % steps_per_render) == 0
        sim.step(render=do_render)
        waited_steps += 1
        ros_if.spin_once(timeout_sec=0.0)
        update_path_length()
        if COLLISION_FAIL_ON_CONTACT and col is not None and col.collided:
            return EpisodeResult(
                False,
                waited_steps,
                float(sim.current_time),
                f"collision:{col.last_other_actor}",
                compute_spl_contribution(False),
            )

    if not ros_if.got_cmd_vel:
        return EpisodeResult(
            False,
            waited_steps,
            float(sim.current_time),
            "no_cmd_vel",
            compute_spl_contribution(False),
        )

    for k in range(max_steps):
        do_render = (k % steps_per_render) == 0
        sim.step(render=do_render)
        ros_if.spin_once(timeout_sec=0.0)
        update_path_length()
        if COLLISION_FAIL_ON_CONTACT and col is not None and col.collided:
            return EpisodeResult(
                False,
                k + 1,
                float(sim.current_time),
                f"collision:{col.last_other_actor}",
                compute_spl_contribution(False),
            )

        dx, dy, dz = (ros_if.position_tensor[0] - goal_xyz[0]), (ros_if.position_tensor[1] - goal_xyz[1]), (ros_if.position_tensor[2] - goal_xyz[2])
        dist = math.sqrt(dx * dx + dy * dy)
        rot = ros_if.orientation_tensor  # (x, y, z, w)

        # success
        if dist <= GOAL_TOL_M and (time.perf_counter() - episode_start_wall_time) >= MIN_SUCCESS_CHECK_SECONDS:
            return EpisodeResult(
                True,
                k + 1,
                float(sim.current_time),
                "goal_reached",
                compute_spl_contribution(True),
            )

        # tilt failure
        roll_deg, pitch_deg = rot_to_roll_pitch_deg(rot)
        if abs(roll_deg) > MAX_TILT_DEG or abs(pitch_deg) > MAX_TILT_DEG:
            return EpisodeResult(
                False,
                k + 1,
                float(sim.current_time),
                "tilt_limit",
                compute_spl_contribution(False),
            )

        # stuck failure (not enough progress over a sliding window)
        dist_hist.append(dist)
        if len(dist_hist) > window_steps:
            dist_hist.pop(0)
            if (dist_hist[0] - dist_hist[-1]) < MIN_PROGRESS_M:
                return EpisodeResult(
                    False,
                    k + 1,
                    float(sim.current_time),
                    "stuck_no_progress",
                    compute_spl_contribution(False),
                )

    return EpisodeResult(
        False,
        max_steps,
        float(sim.current_time),
        "timeout",
        compute_spl_contribution(False),
    )


def reset_to_clean_state(simulation_app, usd_path: str):
    """
    Most robust “clean reset”: stop, reopen USD, re-init SimulationContext.
    """
    import omni.usd
    from isaacsim.core.api import SimulationContext

    # Stop any running sim
    try:
        import omni.timeline
        omni.timeline.get_timeline_interface().stop()
    except Exception:
        pass

    omni.usd.get_context().open_stage(usd_path)

    # Wait until stage is ready
    while omni.usd.get_context().get_stage() is None and simulation_app.is_running():
        simulation_app.update()

    # Give Kit a few frames to initialize OmniGraph graphs, sensors, and extensions
    for _ in range(5):
        if not simulation_app.is_running():
            break
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    sim = SimulationContext(stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.set_simulation_dt(physics_dt=PHYSICS_DT, rendering_dt=RENDERING_DT)
    return sim, stage


# ------------------------- main -------------------------

def main():
    rng = random.Random(RNG_SEED)

    from isaacsim.simulation_app import SimulationApp
    simulation_app = SimulationApp({"headless": True, "enable_motion_bvh": True,})

    # Enable ROS2 bridge BEFORE opening the stage (if the USD contains ROS graphs)
    from isaacsim.core.utils import extensions
    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    # ROS2 init (publisher + /cmd_vel detection)
    ros_if, rclpy_mod = init_ros()

    yaml_data = load_results_yaml(RESULTS_YAML_PATH)
    model_name, model_metadata = load_active_model_metadata(MODEL_CONFIG_PATH)
    run_record = {
        "model_name": model_name,
        "ckpt_path": model_metadata.get("ckpt_path"),
        "use_exploration": model_metadata.get("use_exploration"),
        "use_distance_to_goal_heuristic": model_metadata.get("use_distance_to_goal_heuristic"),
        "use_collision_heuristic": model_metadata.get("use_collision_heuristic"),
        "mask_lidar": model_metadata.get("mask_lidar"),
        "mask_image": model_metadata.get("mask_image"),
        "rng_seed": RNG_SEED,
        "n_episodes": N_EPISODES,
        "physics_dt": PHYSICS_DT,
        "rendering_dt": RENDERING_DT,
        "max_episode_seconds": MAX_EPISODE_SECONDS,
        "goal_tolerance_m": GOAL_TOL_M,
        "collision_fail_on_contact": COLLISION_FAIL_ON_CONTACT,
        "environments": [],
    }

    try:
        for env_name in ENV_SEQUENCE:
            usd_path = ENV_CONFIGS[env_name]
            sim, stage = reset_to_clean_state(simulation_app, usd_path)

            # Robot prim path
            robot_path = ROBOT_PRIM_PATH.strip() or find_robot_prim_path(stage)
            print(f"Robot prim for {env_name}: {robot_path}")

            # Collision monitor (requires the robot prim to have PhysX contact reporting)
            col = CollisionMonitor(stage, robot_path)

            total = 0
            successes = 0
            collisions = 0
            spl_sum = 0.0

            try:
                for ep in range(N_EPISODES):
                    goal = sample_goal(rng, env_name)

                    try:
                        set_goal_marker(stage, goal, prim_path="/World/Goal")
                    except Exception:
                        pass  # marker is optional

                    res = run_episode(sim, stage, robot_path, goal, ros_if, col)
                    total += 1
                    successes += int(res.success)
                    collisions += int(res.reason.startswith("collision"))
                    spl_sum += res.spl_contribution

                    print(
                        f"[{env_name}][ep {ep:04d}] success={res.success} "
                        f"steps={res.steps} sim_t={res.sim_seconds:.3f}s "
                        f"reason={res.reason} success_rate={successes/total:.3f} "
                        f"collision_rate={collisions/total:.3f} spl={spl_sum/total:.3f}"
                    )

                    ros_if.clear_goal()

                    # Clean reset for next episode within the same environment
                    sim.stop()
                    try:
                        col.shutdown()
                    except Exception:
                        pass
                    sim, stage = reset_to_clean_state(simulation_app, usd_path)
                    robot_path = ROBOT_PRIM_PATH.strip() or find_robot_prim_path(stage)
                    col = CollisionMonitor(stage, robot_path)
            finally:
                ros_if.clear_goal()
                try:
                    sim.stop()
                except Exception:
                    pass
                try:
                    col.shutdown()
                except Exception:
                    pass

            env_record = {
                "environment": env_name,
                "usd_path": usd_path,
                "total": total,
                "successes": successes,
                "collisions": collisions,
                "success_rate": float(successes / total) if total > 0 else 0.0,
                "collision_rate": float(collisions / total) if total > 0 else 0.0,
                "spl": float(spl_sum / total) if total > 0 else 0.0,
            }
            run_record["environments"].append(env_record)

        yaml_data["runs"].append(run_record)
        write_results_yaml(RESULTS_YAML_PATH, yaml_data)
        append_results_csv(RESULTS_CSV_PATH, run_record)
        print(f"Saved evaluation results to {RESULTS_YAML_PATH}")
        print(f"Appended evaluation results to {RESULTS_CSV_PATH}")

    finally:
        ros_if.clear_goal()
        rclpy_mod.shutdown()
        simulation_app.close()

if __name__ == "__main__":
    main()