import numpy as np
import io
import os
import rosbag
from PIL import Image
import cv2
from typing import Any, Tuple, List, Dict
import torchvision.transforms.functional as TF
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan

import tqdm
from cv_bridge import CvBridge
import traceback

from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped

IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = 4 / 3


def process_images(im_list: List, img_process_func) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    images = []
    for img_msg in tqdm.tqdm(im_list, desc="process images"):
        img = img_process_func(img_msg)
        images.append(img)
    return images


def process_tartan_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the tartan_drive dataset
    """
    img = ros_to_numpy(msg, output_resolution=IMAGE_SIZE) * 255
    img = img.astype(np.uint8)
    # reverse the axis order to get the image in the right orientation
    img = np.moveaxis(img, 0, -1)
    # convert rgb to bgr
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
    return img

def process_locobot_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/Image to a PIL image for the locobot dataset
    """
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, -1)
    pil_image = Image.fromarray(img)
    return pil_image


def process_scand_img(msg) -> Image:
    """
    Process image data from a topic that publishes sensor_msgs/CompressedImage to a PIL image for the scand dataset
    """
    # convert sensor_msgs/CompressedImage to PIL image
    img = Image.open(io.BytesIO(msg.data))
    # center crop image to 4:3 aspect ratio
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * IMAGE_ASPECT_RATIO))
    )  # crop to the right ratio
    # resize image to IMAGE_SIZE
    img = img.resize(IMAGE_SIZE)
    return img


############## Add custom image processing functions here #############

def process_sacson_img(msg) -> Image:
    np_arr = np.fromstring(msg.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_np)
    return pil_image

# def process_rellis3D_img(msg) -> Image: # kinda works - but sky is yellow
#     width = msg.width
#     height = msg.height
#     step = msg.step
#     bayer_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width)) # msg.encoding=bayer_rggb8
#     rgb_image = cv2.cvtColor(bayer_data, cv2.COLOR_BAYER_RG2RGB)
#     pil_image = Image.fromarray(rgb_image)
#     return pil_image

def process_rellis3D_img(msg) -> Image:
    width = msg.width
    height = msg.height
    bayer_data = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width)) # msg.encoding=bayer_rggb8
    bgr_image = cv2.cvtColor(bayer_data, cv2.COLOR_BayerRGGB2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return pil_image

def process_husky_img(msg) -> Image:
    try: 
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgra8")
        bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return pil_image
    
    except ImportError as e:
        print(f"ImportError: {e}")
        traceback.print_exc()
        return None

    except Exception as e:
        print(f"An error occured: {e}")
        traceback.print_exc()
        return None 


#######################################################################

def process_lidar(lidar_list: List, lidar_process_func) -> List:
    """
    Process image data from a topic that publishes ros images into a list of PIL images
    """
    lidar = []
    for lidar_msg in tqdm.tqdm(lidar_list, desc="process lidar"):
        pc = lidar_process_func(lidar_msg)
        lidar.append(pc)
    return lidar

def process_rellis3D_lidar(msg):
    xyz = np.array([[0,0,0]])

    for x in msg:
        xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)

    return xyz

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


############ Add custom odometry processing functions here ############


#######################################################################

def get_images_lidar_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    lidartopics: List[str] or str,
    odomtopics: List[str] or str,
    img_process_func: Any,
    lidar_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    """
    Get image and odom data from a bag file

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str] or str): topic name(s) for image data
        lidartopics (list[str] or str): topic name(s) for lidar data
        odomtopics (list[str] or str): topic name(s) for odom data
        img_process_func (Any): function to process image data
        lidar_process_func (Any): function to process lidar data
        odom_process_func (Any): function to process odom data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.
    Returns:
        img_data (list): list of PIL images
        lidar_data (list): list of ply point cloud
        traj_data (list): list of odom data
    """
    # check if bag has both topics
    odomtopic = None
    imtopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    if not (imtopic and odomtopic):
        # bag doesn't have both topics
        return None, None

    lidartopic = lidartopics

    synced_imdata = []
    synced_lidardata = []
    synced_odomdata = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()

    curr_imdata = None
    curr_lidardata = None
    curr_odomdata = None

    # print("==== BAG CONTENTS ====")
    # for topic, info in bag.get_type_and_topic_info().topics.items():
    #     print(f"  {topic:20s} → {info.msg_type}")
    # print("======================")

    # tf_dict = extract_all_transforms(bag)
    # tf_si_v  = tf_dict['vehicle']      # sensor_init → vehicle
    # tf_v_l   = tf_dict['velodyne_1']   # vehicle      → velodyne_1
    # if tf_si_v is None or tf_v_l is None:
    #     raise RuntimeError(f"Missing one of ['vehicle','velodyne_1'] in TFs: got {list(tf_dict)}")
    # t_si_v, R_si_v   = tfmsg_to_matrix(tf_si_v)
    # t_v_l,  R_v_l    = tfmsg_to_matrix(tf_v_l)

    # t_si_l, R_si_l   = compose_transform(t_si_v, R_si_v, t_v_l, R_v_l)
    # t_l_si, R_l_si   = invert_transform(t_si_l, R_si_l)


    for topic, msg, t in bag.read_messages(topics=[imtopic, lidartopic, odomtopic]):
        if topic == imtopic:
            curr_imdata = msg
            #print(type(curr_imdata))
        elif topic == odomtopic:
            curr_odomdata = msg
        elif topic == lidartopic:
            curr_lidardata = msg
            # curr_lidardata = (R_l_si @ curr_lidardata.T).T + t_l_si
            #print(type(curr_lidardata))
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_lidardata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_lidardata.append(curr_lidardata)
                synced_odomdata.append(curr_odomdata)
                currtime = t.to_sec()

    img_data = process_images(synced_imdata, img_process_func)
    lidar_data = process_lidar(synced_lidardata, lidar_process_func)
    traj_data = process_odom(
        synced_odomdata,
        odom_process_func,
        ang_offset=ang_offset,
    )

    return img_data, lidar_data, traj_data

def get_images_and_odom(
    bag: rosbag.Bag,
    imtopics: List[str] or str,
    odomtopics: List[str] or str,
    img_process_func: Any,
    odom_process_func: Any,
    rate: float = 4.0,
    ang_offset: float = 0.0,
):
    """
    Get image and odom data from a bag file

    Args:
        bag (rosbag.Bag): bag file
        imtopics (list[str] or str): topic name(s) for image data
        odomtopics (list[str] or str): topic name(s) for odom data
        img_process_func (Any): function to process image data
        odom_process_func (Any): function to process odom data
        rate (float, optional): rate to sample data. Defaults to 4.0.
        ang_offset (float, optional): angle offset to add to odom data. Defaults to 0.0.
    Returns:
        img_data (list): list of PIL images
        traj_data (list): list of odom data
    """
    # check if bag has both topics
    odomtopic = None
    imtopic = None
    if type(imtopics) == str:
        imtopic = imtopics
    else:
        for imt in imtopics:
            if bag.get_message_count(imt) > 0:
                imtopic = imt
                break
    if type(odomtopics) == str:
        odomtopic = odomtopics
    else:
        for ot in odomtopics:
            if bag.get_message_count(ot) > 0:
                odomtopic = ot
                break
    if not (imtopic and odomtopic):
        # bag doesn't have both topics
        return None, None

    synced_imdata = []
    synced_odomdata = []
    # get start time of bag in seconds
    currtime = bag.get_start_time()

    curr_imdata = None
    curr_odomdata = None

    for topic, msg, t in bag.read_messages(topics=[imtopic, odomtopic]):
        if topic == imtopic:
            curr_imdata = msg
        elif topic == odomtopic:
            curr_odomdata = msg
        if (t.to_sec() - currtime) >= 1.0 / rate:
            if curr_imdata is not None and curr_odomdata is not None:
                synced_imdata.append(curr_imdata)
                synced_odomdata.append(curr_odomdata)
                currtime = t.to_sec()

    img_data = process_images(synced_imdata, img_process_func)
    traj_data = process_odom(
        synced_odomdata,
        odom_process_func,
        ang_offset=ang_offset,
    )

    return img_data, traj_data


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


# cut out non-positive velocity segments of the trajectory
def filter_backwards_with_lidar(
    img_list: List[Image.Image],
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

# cut out non-positive velocity segments of the trajectory
def filter_backwards(
    img_list: List[Image.Image],
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
        new_img_list, new_traj_data = zip(*traj_pair)
        new_traj_data = np.array(new_traj_data)
        new_traj_pos = new_traj_data[:, :2]
        new_traj_yaws = new_traj_data[:, 2]
        return (new_img_list, {"position": new_traj_pos, "yaw": new_traj_yaws})

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]
        yaw1 = traj_yaws[i - 1]
        pos2 = traj_pos[i]
        backwards = is_backwards(pos1, yaw1, pos2)
        backwards = False # consider that we are going forward - if True: it messes up the rellis3D datasets by creating folder with only 1 frame
        if not backwards:
            if start:
                new_traj_pairs = [
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                ]
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))
            else:
                new_traj_pairs.append(
                    (img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])
                )
        elif not start:
            cut_trajs.append(process_pair(new_traj_pairs))
            start = True
    return cut_trajs


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


def ros_to_numpy(
    msg, nchannels=3, empty_value=None, output_resolution=None, aggregate="none"
):
    """
    Convert a ROS image message to a numpy array
    """
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    is_rgb = "8" in msg.encoding
    if is_rgb:
        data = np.frombuffer(msg.data, dtype=np.uint8).copy()
    else:
        data = np.frombuffer(msg.data, dtype=np.float32).copy()

    #print(data)

    data = data.reshape(msg.height, msg.width, nchannels)

    if empty_value:
        mask = np.isclose(abs(data), empty_value)
        fill_value = np.percentile(data[~mask], 99)
        data[mask] = fill_value

    data = cv2.resize(
        data,
        dsize=(output_resolution[0], output_resolution[1]),
        interpolation=cv2.INTER_AREA,
    )

    if aggregate == "littleendian":
        data = sum([data[:, :, i] * (256**i) for i in range(nchannels)])
    elif aggregate == "bigendian":
        data = sum([data[:, :, -(i + 1)] * (256**i) for i in range(nchannels)])

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=0)
    else:
        data = np.moveaxis(data, 2, 0)  # Switch to channels-first

    if is_rgb:
        data = data.astype(np.float32) / (
            255.0 if aggregate == "none" else 255.0**nchannels
        )

    return data

def tfmsg_to_matrix(tf_st: TransformStamped):
    """
    Turn a TransformStamped (parent → child) into (t, R) where
      x_child = R @ x_parent + t.
    """
    q = tf_st.transform.rotation
    v = np.array([q.x, q.y, q.z, q.w])
    v /= np.linalg.norm(v)
    qx, qy, qz, qw = v

    # quaternion → rotation matrix
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz

    R = np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)]
    ])

    tr = tf_st.transform.translation
    t = np.array([tr.x, tr.y, tr.z])
    return t, R


def extract_static_lidar_tf(bag: rosbag.Bag,
                            lidar_frame: str = "velodyne_1") -> TransformStamped:
    """
    Scans /tf_static and /tf in the bag for the first
    TransformStamped whose child_frame_id matches `lidar_frame`.
    Returns that TransformStamped (translation+rotation).
    Raises RuntimeError if none found.
    """
    for topic, msg, t in bag.read_messages(topics=['/tf_static', '/tf']):
        if not isinstance(msg, TFMessage):
            continue
        for tf_st in msg.transforms:
            # child_frame_id is often e.g. "velodyne_1"
            if tf_st.child_frame_id == lidar_frame:
                return tf_st

    raise RuntimeError(f"No static transform for frame '{lidar_frame}' found in bag")

def compose_transform(t_ab, R_ab, t_bc, R_bc):
    """
    Given A→B (t_ab,R_ab) and B→C (t_bc,R_bc),
    return A→C:
      R_ac = R_ab @ R_bc
      t_ac = R_ab @ t_bc + t_ab
    """
    R_ac = R_ab @ R_bc
    t_ac = R_ab @ t_bc + t_ab
    return t_ac, R_ac

def invert_transform(t, R):
    """
    Invert A→B (t,R) to get B→A:
      R_inv = R^T
      t_inv = -R^T @ t
    """
    R_inv = R.T
    t_inv = -R_inv @ t
    return t_inv, R_inv

def extract_all_transforms(bag: rosbag.Bag,
                           search_topics=('/tf_static','/tf')) -> dict:
    """
    Returns a dict mapping each child_frame_id → the last
    TransformStamped seen for that frame.
    """
    # ensure leading slash on topic names
    topics = [t if t.startswith('/') else '/' + t for t in search_topics]
    tf_dict = {}
    for topic, msg, _ in bag.read_messages(topics=topics):
        transforms = getattr(msg, 'transforms', None)
        if transforms is None:
            continue
        for tf_st in transforms:
            tf_dict[tf_st.child_frame_id] = tf_st
    return tf_dict

def pc2_to_numpy(cloud_msg: PointCloud2) -> np.ndarray:
    """
    Convert a sensor_msgs/PointCloud2 into an (N,3) NumPy array of XYZ points.
    """
    # read_points yields (x,y,z, …); we only take the first three fields
    points = pc2.read_points(cloud_msg,
                             field_names=("x","y","z"),
                             skip_nans=True)
    return np.array(list(points))  # shape (N,3)

def laserscan_to_numpy(scan_msg: LaserScan) -> np.ndarray:
    """
    Convert a sensor_msgs/LaserScan into an (N,3) NumPy array of XYZ points
    in the laser frame (z = 0).

    Points with invalid or out-of-range distances are skipped.
    """
    angles = scan_msg.angle_min + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment
    ranges = np.array(scan_msg.ranges, dtype=np.float32)

    # Mask invalid / out-of-range values
    valid = np.isfinite(ranges)
    valid &= ranges >= scan_msg.range_min
    valid &= ranges <= scan_msg.range_max

    ranges = ranges[valid]
    angles = angles[valid]

    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)
    zs = np.zeros_like(xs)

    return np.stack((xs, ys, zs), axis=-1)  # shape (N,3)