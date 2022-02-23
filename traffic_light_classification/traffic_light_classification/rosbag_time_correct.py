import rclpy
from rclpy.node import Node
import sys

from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image
import message_filters

import numpy as np
import cv2

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.image_sub =  self.create_subscription(
            Image,
            '/image_raw/camera0_sec/uncompressed',self.image_callback,10)
        self.image_pub = self.create_publisher(Image, '/corrected_image', 10)

    def image_callback(self, image_data):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        image_data.header = h
        self.image_pub.publish(image_data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
