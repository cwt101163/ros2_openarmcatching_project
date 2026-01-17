import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from cv_bridge import CvBridge
import json
from tf2_ros import TransformListener, Buffer
import sys
from sensor_msgs_py import point_cloud2  # 引入官方点云工具，避免手动构造错误


# ========== 修复版：用官方工具生成点云（完全符合ROS2规范） ==========
def cv2_to_ros_pointcloud(points, frame_id, width, height):
    """
    将OpenCV生成的3D点（Nx3）转为ROS PointCloud2消息
    修复点：使用官方工具构造，避免手动字节流导致的类型错误
    """
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = rclpy.clock.Clock().now().to_msg()

    # 保留图像分辨率（有序点云）
    msg.height = height
    msg.width = width
    msg.is_dense = False  # 允许无效点

    # 定义点云字段（严格符合ROS2 FLOAT32规范）
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
    ]
    
    # 过滤无效点（z<=0的点直接剔除，避免nan导致的类型问题）
    valid_mask = points[:, 2] > 0
    valid_points = points[valid_mask]
    
    # 使用ROS2官方工具构造点云（核心修复：避免手动处理字节流）
    msg = point_cloud2.create_cloud(msg.header, fields, valid_points.tolist())
    
    # 恢复有序点云的尺寸信息
    msg.height = height
    msg.width = width
    msg.is_dense = False
    return msg


# ========== 物体检测核心节点（修复版） ==========
class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")
        self.bridge = CvBridge()
        # TF2坐标转换（保留原功能）
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 订阅器：直接订阅相机原始消息
        self.rgb_sub = self.create_subscription(
            Image, "/camera/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback, 10)

        # 发布器：基坐标系物体坐标+调试用点云
        self.object_pub = self.create_publisher(
            PointStamped, "/object/base/position", 10)
        self.debug_pc_pub = self.create_publisher(
            PointCloud2, "/debug/pointcloud", 10)

        # 缓存RGB和深度图
        self.rgb_img = None
        self.depth_img = None
        # 相机内参（适配320x240分辨率）
        self.camera_matrix = np.array([[200, 0, 160],  # fx=200, cx=160
                                       [0, 200, 120],  # fy=200, cy=120
                                       [0, 0, 1]], dtype=np.float32)

        # 加载手眼标定结果
        self.calib_R, self.calib_t = self.load_calibration_result()
        self.get_logger().info("✅ 物体检测节点已启动（修复版）")

        # 初始化显示窗口
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 640, 480)

    def load_calibration_result(self):
        """保留原手眼标定加载逻辑，增加异常详细信息"""
        calib_path = "/home/breeze/ros2_ws/src/openarm_moveit_config/config/handeye_data/calibration_result.json"
        try:
            with open(calib_path, "r") as f:
                data = json.load(f)
            R = np.array(data["rotation"], dtype=np.float32)
            t = np.array(data["translation"], dtype=np.float32).reshape(3, 1)
            self.get_logger().info(f"✅ 手眼标定结果加载成功：R={R.shape}, t={t.shape}")
            return R, t
        except FileNotFoundError:
            self.get_logger().error(f"❌ 标定文件不存在：{calib_path}")
            return np.eye(3, dtype=np.float32), np.array([[0.3], [0.0], [0.8]], dtype=np.float32)
        except KeyError as e:
            self.get_logger().error(f"❌ 标定文件格式错误，缺少字段：{e}")
            return np.eye(3, dtype=np.float32), np.array([[0.3], [0.0], [0.8]], dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"⚠️ 标定文件加载失败，用默认值：{str(e)}")
            return np.eye(3, dtype=np.float32), np.array([[0.3], [0.0], [0.8]], dtype=np.float32)

    def rgb_callback(self, msg):
        """RGB图像回调：增加异常详细信息"""
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # 验证图像尺寸
            if self.rgb_img.shape[:2] != (240, 320):
                self.get_logger().warn(f"⚠️ RGB图像尺寸异常：{self.rgb_img.shape}，期望(240,320)")
        except Exception as e:
            self.get_logger().error(f"❌ RGB转换失败：{str(e)}，行号：{sys.exc_info()[2].tb_lineno}")

    def depth_callback(self, msg):
        """深度图回调+核心检测逻辑（修复版）"""
        if self.rgb_img is None:
            self.get_logger().warn("⚠️ RGB图像未加载，跳过深度处理")
            return

        try:
            # 1. 转换深度图（增加数据验证）
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            depth_m = self.depth_img.astype(np.float32) / 1000.0  # 毫米→米
            
            # 验证深度图尺寸和数据范围
            if depth_m.shape[:2] != (240, 320):
                self.get_logger().warn(f"⚠️ 深度图尺寸异常：{depth_m.shape}，期望(240,320)")
                return
            if np.all(depth_m == 0):
                self.get_logger().warn("⚠️ 深度图全为0，无有效深度数据")
                return

            # 2. 颜色识别（增加掩码优化）
            hsv = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([10, 50, 50])
            upper_yellow = np.array([40, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 形态学操作去除噪声
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 3. 找黄色区域质心（增加有效性检查）
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                self.get_logger().warn("⚠️ 未检测到黄色物体")
                cv2.imshow("Object Detection", self.rgb_img)
                cv2.waitKey(1)
                return

            # 取最大轮廓（避免小噪声）
            max_contour = max(contours, key=cv2.contourArea)
            # 过滤过小的轮廓（面积小于50像素视为噪声）
            if cv2.contourArea(max_contour) < 50:
                self.get_logger().warn("⚠️ 检测到的黄色区域过小，视为噪声")
                cv2.imshow("Object Detection", self.rgb_img)
                cv2.waitKey(1)
                return
            
            M = cv2.moments(max_contour)
            # 防止除零错误
            if M["m00"] == 0:
                self.get_logger().warn("⚠️ 轮廓矩计算失败（m00=0）")
                return
            
            cx = int(M["m10"] / M["m00"])  # 质心x坐标
            cy = int(M["m01"] / M["m00"])  # 质心y坐标

            # 4. 可视化结果
            vis_img = self.rgb_img.copy()
            cv2.drawContours(vis_img, [max_contour], -1, (0, 255, 0), 2)
            cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.imshow("Object Detection", vis_img)
            cv2.waitKey(1)

            # 5. 用OpenCV计算3D坐标（相机坐标系）
            # 边界检查：确保质心在图像范围内
            if cy < 0 or cy >= depth_m.shape[0] or cx < 0 or cx >= depth_m.shape[1]:
                self.get_logger().warn(f"⚠️ 质心坐标超出图像范围：({cx}, {cy})")
                return
            
            z = depth_m[cy, cx]
            if z <= 0:
                self.get_logger().warn(f"⚠️ 质心处深度无效：z={z}")
                return
            
            # 像素坐标→相机坐标
            x = (cx - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
            y = (cy - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
            cam_point = np.array([[x], [y], [z]], dtype=np.float32)

            # 6. 坐标转换（相机→基坐标系）
            base_point = self.calib_R @ cam_point + self.calib_t

            # 7. 发布结果
            object_msg = PointStamped()
            object_msg.header.frame_id = "base_link"
            object_msg.header.stamp = self.get_clock().now().to_msg()
            object_msg.point.x = float(base_point[0, 0])  # 显式转为float，避免类型问题
            object_msg.point.y = float(base_point[1, 0])
            object_msg.point.z = float(base_point[2, 0])
            self.object_pub.publish(object_msg)

            # 8. 发布调试点云（修复版）
            h, w = depth_m.shape
            u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
            x_grid = (u_grid - self.camera_matrix[0, 2]) * depth_m / self.camera_matrix[0, 0]
            y_grid = (v_grid - self.camera_matrix[1, 2]) * depth_m / self.camera_matrix[1, 1]
            z_grid = depth_m
            
            # 转为Nx3点云并过滤无效点
            points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T
            ros_pc = cv2_to_ros_pointcloud(points, "camera_link", w, h)
            self.debug_pc_pub.publish(ros_pc)

            # 打印日志
            self.get_logger().info(
                f"📷 相机坐标：X={x:.3f}, Y={y:.3f}, Z={z:.3f} | 🤖 基坐标：X={base_point[0,0]:.3f}, Y={base_point[1,0]:.3f}, Z={base_point[2,0]:.3f}")

        except Exception as e:
            self.get_logger().error(f"❌ 检测失败：{str(e)}，行号：{sys.exc_info()[2].tb_lineno}")
            # 打印完整异常堆栈，方便调试
            import traceback
            self.get_logger().error(f"❌ 异常详情：{traceback.format_exc()}")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 节点终止")
    except Exception as e:
        node.get_logger().error(f"❌ 节点运行异常：{str(e)}")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
