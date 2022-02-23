import rclpy
from rclpy.node import Node
import sys

from std_msgs.msg import String
from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import Image
import message_filters
from vision_msgs.msg import Classification2D
from vision_msgs.msg import ObjectHypothesis
import numpy as np
import cv2

from traffic_light_classification.modelClass import TrafficLightClassifier

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.model = TrafficLightClassifier("cuda")
        bbox_sub = message_filters.Subscriber(
            self,BoundingBoxes,
            '/darknet_ros/bounding_boxes')
        image_sub = message_filters.Subscriber(
            self,Image,
            '/corrected_image')
        synchronizer = message_filters.ApproximateTimeSynchronizer(
            [image_sub, bbox_sub],
            10,1
        )
        synchronizer.registerCallback(self.callback)
        self.classfication_publisher = self.create_publisher(Classification2D, '/classfication_result', 10)

    def callback(self, image_data,bbox_data):
        img = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        bboxes = bbox_data.bounding_boxes
        for bbox in bboxes:
            if("traffic light"==bbox.class_id):
                x1,y1,x2,y2 = bbox.xmin,bbox.ymin,bbox.xmax,bbox.ymax
                cropped_img = img[y1:y2,x1:x2,:].copy()
                
                cls_name,score = self.model.prediction(cropped_img)
                pub_msg =  Classification2D()
                obj_msg = ObjectHypothesis()
                obj_msg.id = cls_name
                obj_msg.score = score
                h = Header()
                h.stamp = self.get_clock().now().to_msg()
                pub_msg.header = h
                pub_msg.results = [obj_msg]
                pub_msg.source_img = image_data
                self.classfication_publisher.publish(pub_msg)

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
    
