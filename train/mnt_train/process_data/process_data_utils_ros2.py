from cv_bridge import CvBridge
import cv2
from PIL import Image as PILImage
import numpy as np
from pathlib import Path
import yaml


# for PointCloud2 parsing
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import Odometry

# for odom→yaw conversion
# from tf_transformations import euler_from_quaternion

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from typing import List, Any, Tuple, Dict

import tqdm

def process_husky_img(msg) -> PILImage:
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgra8")
    bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(rgb_image)
    return pil_image

def process_rellis3D_lidar(msg):
    xyz = np.array([[0,0,0]])

    for x in msg:
        xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)

    return xyz

def nav_to_xy_yaw(odom_msg, ang_offset: float) -> Tuple[List[float], float]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position
    """

    position = odom_msg.pose.pose.position
    orientation = odom_msg.pose.pose.orientation
    yaw = (
        quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        + ang_offset
    )
    return [position.x, position.y], yaw

def process_images(im_list: List, img_process_func) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    images = []
    for img_msg in tqdm.tqdm(im_list, desc="process images"):
        img = img_process_func(img_msg)
        images.append(img)
    return images

def process_odom(
    odom_list: List,
    odom_process_func: Any,
    ang_offset: float = 0.0,
) -> Dict[np.ndarray, np.ndarray]:
    """
    Process odom data from a topic that publishes nav_msgs/Odometry into position and yaw
    """
    xys = []
    yaws = []
    for odom_msg in tqdm.tqdm(odom_list, desc="process odom"):
        xy, yaw = odom_process_func(odom_msg, ang_offset)
        xys.append(xy)
        yaws.append(yaw)
    return {"position": np.array(xys), "yaw": np.array(yaws)}

def process_lidar(lidar_list: List, lidar_process_func) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    lidar = []
    for lidar_msg in tqdm.tqdm(lidar_list, desc="process lidar"):
        pc = lidar_process_func(lidar_msg)
        lidar.append(pc)
    return lidar


def get_images_lidar_and_odom(
    bag_path: str,
    imtopics: List[str] or str,
    lidartopics: List[str] or str,
    odomtopics: List[str] or str,
    img_process_func: Any,
    lidar_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    # Prepare reader
    storage_id = infer_storage_id(bag_path)
    storage_options = StorageOptions(
        uri=str(Path(bag_path).parent if Path(bag_path).is_file() else Path(bag_path)),
        storage_id=storage_id,
    )
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_info = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topic_info}

    imtopic = imtopics if isinstance(imtopics, str) else next((t for t in imtopics if t in topic_types), None)
    odomtopic = odomtopics if isinstance(odomtopics, str) else next((t for t in odomtopics if t in topic_types), None)
    lidartopic = lidartopics if isinstance(lidartopics, str) else next((t for t in lidartopics if t in topic_types), None)

    if not imtopic or not odomtopic:
        return None

    synced_imdata, synced_lidardata, synced_odomdata = [], [], []
    currtime = 0.0
    curr_imdata, curr_odomdata, curr_lidardata = None, None, None
    while reader.has_next():
        topic, data, t = reader.read_next()
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_type)

        t_sec = t / 1e9
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
        elif topic == lidartopic:
            curr_lidardata = msg 
            
        if (t_sec - currtime) >= 1.0 / rate:
            if curr_imdata and curr_lidardata is not None and curr_odomdata:
                synced_imdata.append(curr_imdata)
                synced_lidardata.append(curr_lidardata)
                synced_odomdata.append(curr_odomdata)
                currtime = t_sec

    img_data = process_images(synced_imdata, img_process_func)
    lidar_data = process_lidar(synced_lidardata, lidar_process_func)
    traj_data = process_odom(synced_odomdata, odom_process_func, ang_offset=ang_offset)

    return img_data, lidar_data, traj_data


def quat_to_yaw(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Convert a batch quaternion into a yaw angle
    yaw is rotation around z in radians (counterclockwise)
    """
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return yaw


def pc2_to_numpy(cloud_msg: PointCloud2) -> np.ndarray:
    """
    Convert a sensor_msgs/PointCloud2 into an (N,3) NumPy array of XYZ points.
    """
    # read_points yields (x,y,z, …); we only take the first three fields
    points = pc2.read_points(cloud_msg,
                             field_names=("x","y","z"),
                             skip_nans=True)
    x = points['x']
    y = points['y']
    z = points['z']
    point_cloud_array = np.stack([x, y, z], axis=-1).astype(np.float32)
    return point_cloud_array  # shape (N,3)

def pc2_to_numpy_rot(cloud_msg: PointCloud2) -> np.ndarray:
    """
    Convert a sensor_msgs/PointCloud2 into an (N,3) NumPy array of XYZ points.
    """
    # read_points yields (x,y,z, …); we only take the first three fields
    points = pc2.read_points(
        cloud_msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    x = points["x"]
    y = points["y"]
    z = points["z"]
    point_cloud_array = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Rotate by quaternion (x=0, y=0, z=1, w=0) i.e. 180° about Z:
    # (x, y, z) -> (-x, -y, z)
    point_cloud_array[:, 0] *= -1.0
    point_cloud_array[:, 1] *= -1.0

    return point_cloud_array  # shape (N,3)

def is_backwards(
    pos1: np.ndarray, yaw1: float, pos2: np.ndarray, eps: float = 1e-5
) -> bool:
    """
    Check if the trajectory is going backwards given the position and yaw of two points
    Args:
        pos1: position of the first point

    """
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps

def filter_backwards_with_lidar(
    img_list: List[Image],
    lidar_list: List[np.ndarray],
    traj_data: Dict[str, np.ndarray],
    start_slack: int = 0,
    end_slack: int = 0,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Cut out non-positive velocity segments of the trajectory
    Args:
        traj_type: type of trajectory to cut
        img_list: list of images
        traj_data: dictionary of position and yaw data
        start_slack: number of points to ignore at the start of the trajectory
        end_slack: number of points to ignore at the end of the trajectory
    Returns:
        cut_trajs: list of cut trajectories
        start_times: list of start times of the cut trajectories
    """
    traj_pos = traj_data["position"]
    traj_yaws = traj_data["yaw"]
    cut_trajs = []
    start = True

    def process_pair(traj_pair: list) -> Tuple[List, Dict]:
        new_img_list, new_lidar_list, new_traj_data = zip(*traj_pair)
        new_traj_data = np.array(new_traj_data)
        new_traj_pos = new_traj_data[:, :2]
        new_traj_yaws = new_traj_data[:, 2]
        return (new_img_list, new_lidar_list, {"position": new_traj_pos, "yaw": new_traj_yaws})

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]
        yaw1 = traj_yaws[i - 1]
        pos2 = traj_pos[i]
        backwards = is_backwards(pos1, yaw1, pos2)
        backwards = False # consider that we are going forward - if True: it messes up the rellis3D datasets by creating folder with only 1 frame
        if not backwards:
            if start:
                new_traj_pairs = [
                    (img_list[i - 1], lidar_list[i -1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                ]
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))
            else:
                new_traj_pairs.append(
                    (img_list[i - 1], lidar_list[i -1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                )
        elif not start:
            cut_trajs.append(process_pair(new_traj_pairs))
            start = True
    return cut_trajs

def infer_storage_id(bag_uri: str) -> str:
    p = Path(bag_uri)

    # rosbag2_py expects the BAG DIRECTORY as uri in almost all cases
    if p.is_file():
        p = p.parent

    meta = p / "metadata.yaml"
    if meta.exists():
        md = yaml.safe_load(meta.read_text())
        sid = md.get("storage_identifier")
        if sid:
            return sid
        # some metadata.yaml nest info (be tolerant)
        info = md.get("rosbag2_bagfile_information", {}) or {}
        sid = info.get("storage_identifier")
        if sid:
            return sid

    # fallback: detect by files present
    if any(p.glob("*.mcap")):
        return "mcap"
    if any(p.glob("*.db3")):
        return "sqlite3"
    return "sqlite3"