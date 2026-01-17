#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_pub_node')
        self.publisher_ = self.create_publisher(Image, 'camera_topic', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.image = cv2.imread("/home/breeze/Desktop/test.jpg")
        self.bridge = CvBridge()
        self.get_logger().info("Image Publisher Node Started!")

    def timer_callback(self):
    ros_image = self.bridge.cv2_to_imgmsg(self.image, encoding="bgr8")
    self.publisher_.publish(ros_image)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
