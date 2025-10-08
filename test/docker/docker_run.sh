docker run --name mnt-isaac-12.8 --rm --privileged -it --gpus all -e "ACCEPT_EULA=Y" --network=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e "PRIVACY_CONSENT=Y" \
    -e ROS_DOMAIN_ID=0 \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v ~/multimodal-navigation-transformer:/workspace/multimodal-navigation-transformer \
    -v ~/MNT-simulation:/workspace/MNT-simulation \
    -u root \
    mnt-isaac:cuda12.8