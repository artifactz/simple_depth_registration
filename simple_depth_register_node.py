import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from collections import namedtuple

import time

class DepthRegisterer(object):
    Intrinsics = namedtuple('Intrinsics', ['fx', 'fy', 'cx', 'cy'])

    def __init__(self, x_offset, y_offset, z_offset, depth_scale=1.0):
        self.extrinsics = x_offset, y_offset, z_offset
        self.intrinsics = {}
        self.depth_scale = depth_scale
        self.pixel_grid = None

    def set_intrinsics(self, cam_id, fx, fy, cx, cy):
        self.intrinsics[cam_id] = DepthRegisterer.Intrinsics(fx, fy, cx, cy)

    def has_intrinsics(self, cam_id):
        return cam_id in self.intrinsics

    def register(self, rgb_image, depth_image):
        if self.pixel_grid is None:
            t0 = time.time()
            self.pixel_grid = np.stack((
                np.array([np.arange(depth_image.shape[0]) for _ in xrange(depth_image.shape[1])]).T,
                np.array([np.arange(depth_image.shape[1]) for _ in xrange(depth_image.shape[0])])
                ), axis=2)

        registered_depth_image = np.zeros(rgb_image.shape[:2], dtype='float64')

        fx_rgb, fy_rgb, cx_rgb, cy_rgb = self.intrinsics['rgb']
        fx_d, fy_d, cx_d, cy_d = self.intrinsics['depth']
        x_offset, y_offset, z_offset = self.extrinsics

        valid_depths = depth_image > 0
        valid_pixels = self.pixel_grid[valid_depths]
        zs = depth_image[valid_depths] / self.depth_scale + z_offset
        ys = (((((valid_pixels[:, 0] - cy_d) * zs) / fy_d + y_offset) * fy_rgb / zs + cy_rgb)).astype('int')
        xs = (((((valid_pixels[:, 1] - cx_d) * zs) / fx_d + x_offset) * fx_rgb / zs + cx_rgb)).astype('int')
        valid_positions = np.logical_and(np.logical_and(np.logical_and(ys >= 0, ys < registered_depth_image.shape[0]), xs >= 0), xs < registered_depth_image.shape[1])

        registered_depth_image[ys[valid_positions], xs[valid_positions]] = zs[valid_positions]
        return registered_depth_image

    def process_images(self, **images):
        assert(len(images) == 2 and 'rgb' in images and 'depth' in images)
        assert(self.has_intrinsics('rgb') and self.has_intrinsics('depth'))
        return self.register(images['rgb'], images['depth'])


class DepthRegisterNode(object):
    def __init__(self):
        rospy.init_node('simple_depth_register_node')
        self.cv_bridge = CvBridge()
        self.dr = DepthRegisterer(-0.0589333333, 0, 0, depth_scale=1000.) # -0.0589333333, 0, 0 are the extrinsics (offset rgb cam -> depth cam)

        rgb_topic = rospy.get_param('~rgb_topic', '/camera/rgb/image_raw')
        depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_raw')

        rgb_info_topic, depth_info_topic = map(lambda t: t[:t.rfind('/')] + '/camera_info', (rgb_topic, depth_topic))
        self.camera_info_callback(rospy.wait_for_message(rgb_info_topic, CameraInfo), 'rgb')
        self.camera_info_callback(rospy.wait_for_message(depth_info_topic, CameraInfo), 'depth')
        rospy.loginfo('camera calibration data OK')

        rgb_sub = message_filters.Subscriber(rgb_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.sub = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 3, 0.1)
        self.sub.registerCallback(self.image_pair_callback)
        rospy.loginfo('synchronized subscriber OK')

        self.pub = rospy.Publisher('/simple_depth_register_node/depth_registered', Image, queue_size=5)
        self.pub_info_unregistered = rospy.Publisher('/simple_depth_register_node/info_image_unregistered', Image, queue_size=5)
        self.pub_info_registered = rospy.Publisher('/simple_depth_register_node/info_image_registered', Image, queue_size=5)

        self.rgb_image = None
        self.depth_image = None
        self.registered_depth_image = None

    def camera_info_callback(self, msg_camera_info, cam_id):
        fx, _, cx, _, fy, cy, _, _, _ = msg_camera_info.K
        self.dr.set_intrinsics(cam_id, fx, fy, cx, cy)

    def image_pair_callback(self, msg_rgb_image, msg_depth_image):
        try:
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(msg_rgb_image)
        except CvBridgeError as e:
            rospy.logwarn('error converting rgb image: %s' % e)
            return
        try:
            self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg_depth_image)
        except CvBridgeError as e:
            rospy.logwarn('error converting depth image: %s' % e)
            return

        self.registered_depth_image = self.dr.process_images(rgb=self.rgb_image, depth=self.depth_image)
        msg = self.cv_bridge.cv2_to_imgmsg(self.registered_depth_image)
        msg.header.stamp = msg_depth_image.header.stamp
        self.pub.publish(msg)

        img_unregistered_img, img_registered_img = self.get_info_images()
        self.pub_info_unregistered.publish(self.cv_bridge.cv2_to_imgmsg(img_unregistered_img))
        self.pub_info_registered.publish(self.cv_bridge.cv2_to_imgmsg(img_registered_img))

    def get_info_images(self):
        # clip to 5000 millimeters
        depth_image = self.depth_image / 5000. * 255.
        depth_image[depth_image > 255] = 255
        depth_image = cv2.cvtColor(cv2.resize(depth_image.astype('uint8'), (self.rgb_image.shape[1], self.rgb_image.shape[0]), cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

        # clip to 5 meters
        registered_depth_image = self.registered_depth_image.astype(float) * 255. / 5.
        registered_depth_image[registered_depth_image > 255] = 255
        registered_depth_image = cv2.cvtColor(registered_depth_image.astype('uint8'), cv2.COLOR_GRAY2BGR)

        return cv2.addWeighted(self.rgb_image, 0.5, depth_image, 0.5, 0), cv2.addWeighted(self.rgb_image, 0.5, registered_depth_image, 0.5, 0)

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    node = DepthRegisterNode()
    node.spin()
