# `simple_depth_registration`

*faster ROS depth image registration*

This is a slim ROS library written in Python computing how the depth values of an RGB-D image would look like from the perspective of the RGB camera instead of the IR camera.

An existing solution (`depth_image_proc`) exists within the [`image_pipeline`](https://github.com/ros-perception/image_pipeline) package, but suffers from major performance loss when dealing with rather dense depth images.

## Running this node

The default input topics are `/camera/rgb/image_raw` and `/camera/depth/image_raw` respectively, but can be changed via ros params (see example launch file).

The registered depth image is published via the topic `/simple_depth_registration/depth_registered`, but can be customized, too.

The extrinsics (offset from RGB camera to IR camera) are set via `x_offset`, `y_offset` and `z_offset`. The intrinsics are fetched automatically via `camera_info` topics.

For testing purposes there are two additional image topics to watch: `/simple_depth_registration/info_image_unregistered` and `/simple_depth_registration/info_image_registered`, which are rgb and (un)registered depth image blended together.

## Dependencies

Python 2, NumPy, basic ROS packages.