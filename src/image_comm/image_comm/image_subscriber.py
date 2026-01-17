#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_sub_node')
        self.subscription = self.create_subscription(
            Image,
            'camera_topic',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.get_logger().info("Image Subscriber Node Started!")

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("Camera Feed", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
