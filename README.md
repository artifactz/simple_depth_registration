# `simple_depth_registration`

*faster ROS depth image registration*

This is a slim ROS library written in Python computing how the depth values of an RGB-D image would look like from the perspective of the RGB camera instead of the IR camera.

An existing solution (`depth_image_proc`) exists within the [`image_pipeline`](https://github.com/ros-perception/image_pipeline) package, but suffers from major performance loss when dealing with rather dense depth images.

## Running this node

See the [example launch file](launch/realsense_r200.launch).

### Subscribed topics

`/camera/rgb/image_raw` (`sensor_msgs/Image`)  
RGB camera images

`/camera/depth/image_raw` (`sensor_msgs/Image`)  
depth images

### Published topics

`/simple_depth_registration/depth_registered` (`sensor_msgs/Image`)  
registered depth image

`/simple_depth_registration/info_image_unregistered` (`sensor_msgs/Image`)  
the RGB image and the unregistered depth image blended together (for testing purposes)

`/simple_depth_registration/info_image_registered` (`sensor_msgs/Image`)  
the RGB image and the registered depth image blended together

### Parameters

`x_offset`, `y_offset` and `z_offset` (`float`, default: `0.0`)  
extrinsics (offset from RGB camera to IR camera)

`depth_scale` (`float`, default: `1000.0`)  
Depth values are treated as scaled by this value, which means all depth values are divided by it and then expected to be metric.

`rgb_topic` (`string`, default: `/camera/rgb/image_raw`)  
overrides RGB input topic name

`depth_topic` (`string`, default: `/camera/depth/image_raw`)  
overrides depth input topic name

`registered_topic` (`string`, default: `~depth_registered`)  
overrides registered depth output topic name

## Dependencies

Python 2, NumPy, basic ROS packages.