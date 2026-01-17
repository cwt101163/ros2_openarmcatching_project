import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # 只保留Image导入
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from cv_bridge import CvBridge
import json
from tf2_ros import TransformListener, Buffer
import sys


# ========== 物体检测核心节点（无Open3D，无内存问题） ==========
class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")
        self.bridge = CvBridge()
        # TF2坐标转换（保留原功能）
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 订阅器：直接订阅相机原始消息
        self.rgb_sub = self.create_subscription(Image, "/camera/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, "/camera/depth/image_raw", self.depth_callback, 10)

        # 发布器：仅保留基坐标系物体坐标
        self.object_pub = self.create_publisher(PointStamped, "/object/base/position", 10)

        # 缓存RGB和深度图
        self.rgb_img = None
        self.depth_img = None
        # 相机内参（适配320x240分辨率）
        self.camera_matrix = np.array([[200, 0, 160],  # fx=200, cx=160
                                       [0, 200, 120],  # fy=200, cy=120
                                       [0, 0, 1]], dtype=np.float32)

        # 加载手眼标定结果
        self.calib_R, self.calib_t = self.load_calibration_result()
        self.get_logger().info("✅ 物体检测节点已启动（无Open3D版本）")

        # 初始化显示窗口
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 640, 480)

    def load_calibration_result(self):
        calib_path = "/home/breeze/ros2_ws/src/openarm_moveit_config/config/handeye_data/calibration_result.json"
        try:
            with open(calib_path, "r") as f:
                data = json.load(f)
            R = np.array(data["rotation"], dtype=np.float32)
            t = np.array(data["translation"], dtype=np.float32).reshape(3, 1)
            self.get_logger().info("✅ 手眼标定结果加载成功")
            return R, t
        except Exception as e:
            self.get_logger().warn(f"⚠️ 标定文件加载失败，用默认值：{str(e)}")
            return np.eye(3, dtype=np.float32), np.array([[0.3], [0.0], [0.8]], dtype=np.float32)

    def rgb_callback(self, msg):
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"❌ RGB转换失败：{str(e)}")

    def depth_callback(self, msg):
        if self.rgb_img is None:
            return

        try:
            # 1. 转换深度图（16UC1→米）
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            depth_m = self.depth_img.astype(np.float32) / 1000.0
            depth_m[depth_m == 0] = -1

            # 2. 颜色识别
            hsv = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([5, 30, 30])
            upper_yellow = np.array([45, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

            # 3. 找黄色区域质心
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self.get_logger().warn("⚠️ 未检测到黄色物体")
                cv2.imshow("Object Detection", self.rgb_img)
                cv2.waitKey(1)
                return

            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 4. 可视化
            vis_img = self.rgb_img.copy()
            cv2.drawContours(vis_img, [max_contour], -1, (0, 255, 0), 2)
            cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.imshow("Object Detection", vis_img)
            cv2.waitKey(1)

            # 5. 计算3D坐标（相机坐标系）
            if depth_m[cy, cx] == -1:
                self.get_logger().warn("⚠️ 质心处深度无效")
                return
            u, v = cx, cy
            z = depth_m[cy, cx]
            x = (u - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
            y = (v - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
            cam_point = np.array([[x], [y], [z]], dtype=np.float32)

            # 6. 坐标转换（相机→基坐标系）
            base_point = self.calib_R @ cam_point + self.calib_t

            # 7. 发布结果（显式转为float）
            object_msg = PointStamped()
            object_msg.header.frame_id = "base_link"
            object_msg.header.stamp = self.get_clock().now().to_msg()
            object_msg.point.x = float(base_point[0, 0])
            object_msg.point.y = float(base_point[1, 0])
            object_msg.point.z = float(base_point[2, 0])
            self.object_pub.publish(object_msg)

            # 打印日志
            self.get_logger().info(
                f"📷 相机坐标：X={x:.3f}, Y={y:.3f}, Z={z:.3f} | 🤖 基坐标：X={base_point[0, 0]:.3f}, Y={base_point[1, 0]:.3f}, Z={base_point[2, 0]:.3f}")

        except Exception as e:
            self.get_logger().error(f"❌ 检测失败：{str(e)}，行号：{sys.exc_info()[2].tb_lineno}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 节点终止")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()