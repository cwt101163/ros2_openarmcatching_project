#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
import numpy as np
import cv2
from cv_bridge import CvBridge
import open3d as o3d
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class DataPreprocessingNode(Node):
    def __init__(self):
        super().__init__("data_preprocessing_node")
        self.bridge = CvBridge()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_depth = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            qos
        )
        self.sub_rgb = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.rgb_callback,
            qos
        )
        self.pub_pointcloud = self.create_publisher(PointCloud2, "/processed_pointcloud", 10)
        self.rgb_img = None
        self.get_logger().info("数据预处理节点已启动 ✅")

    def rgb_callback(self, msg):
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.rgb_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB)
            self.rgb_img = np.ascontiguousarray(self.rgb_img)
            self.get_logger().info(f"成功接收RGB图像！分辨率：{self.rgb_img.shape[1]}x{self.rgb_img.shape[0]}")  # 打印分辨率

            # ========== 关键修改：生成与RGB图像同分辨率的测试点云 ==========
            # img_height, img_width = self.rgb_img.shape[:2]
            # 生成和图像像素数量一致的点云（示例：按像素坐标生成3D点，z固定为0.5m）
            # 创建网格坐标（x: 图像列，y: 图像行，z: 固定深度）
            # x = np.tile(np.arange(img_width), img_height).reshape(-1, 1) / 1000.0  # 列坐标转米
        # y = np.repeat(np.arange(img_height), img_width).reshape(-1, 1) / 1000.0  # 行坐标转米
        #  z = np.ones((img_height * img_width, 1)) * 0.5  # 固定深度0.5米
        #  points = np.hstack([x, y, z])  # 组合成Nx3的点云数组

        # 创建Open3D点云
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)

        # 转换为ROS点云消息（保留图像分辨率信息）
        # ros_pcd = self.open3d_to_ros_pointcloud(pcd, "camera_link", img_width, img_height)
        # 发布
        #  self.pub_pointcloud.publish(ros_pcd)
        # self.get_logger().info(f"测试点云已发布！点数量：{len(points)}")
        except Exception as e:
            self.get_logger().error(f"RGB图像转换失败：{str(e)}")

    def open3d_to_ros_pointcloud(self, pcd: o3d.geometry.PointCloud, frame_id: str, width=None,
                                 height=None) -> PointCloud2:
        points = np.asarray(pcd.points)
        points = points[np.isfinite(points).all(axis=1)]

        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()

        # ========== 关键修改：保留图像分辨率（有序点云） ==========
        if width and height and width * height == len(points):
            msg.height = height  # 图像高度
            msg.width = width  # 图像宽度
        else:
            msg.height = 1  # 无序点云
            msg.width = len(points)

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 12  # x(4)+y(4)+z(4)
        msg.row_step = msg.point_step * msg.width
        msg.data = points.astype(np.float32).tobytes()
        msg.is_dense = True  # 无无效点
        return msg

    def depth_callback(self, msg):
        if self.rgb_img is None:
            self.get_logger().warn("RGB图像未就绪")
            return
        try:
            # 1. 转换深度图（关键修改：显式指定16UC1，增加类型检查）
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            # 检查数据类型是否正确
            if depth_img.dtype != np.uint16:
                self.get_logger().error(f"深度图类型错误！预期uint16，实际{depth_img.dtype}")
                return
            # 转换为米（16UC1是毫米单位）
            depth_img = depth_img.astype(np.float32) / 1000.0
            self.get_logger().info(f"深度图尺寸：{depth_img.shape}，数据范围：{np.min(depth_img)}~{np.max(depth_img)}米")

            # 2. 转换RGB图（假设self.rgb_img是OpenCV格式）
            rgb_img = cv2.cvtColor(self.rgb_img, cv2.COLOR_BGR2RGB)  # 转为RGB格式（Open3D要求）

            # 3. 创建RGBD图像（Open3D格式）
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_img),
                o3d.geometry.Image(depth_img),
                depth_scale=1.0,  # 深度图单位是米，无需缩放
                depth_trunc=2.0,  # 截断2米外的点
                convert_rgb_to_intensity=False
            )

            # 4. 创建相机内参（需匹配实际相机参数，这里用640x480默认值）
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            )

            # 5. 生成原始点云（不做后处理）
            pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, intrinsic
            )
            self.get_logger().info(f"原始点云数量：{len(np.asarray(pointcloud.points))}")

            # 6. 临时注释后处理（先发布原始点云）
            # pointcloud = pointcloud.voxel_down_sample(voxel_size=0.005)
            # cl, ind = pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            # pointcloud = pointcloud.select_by_index(ind)

            # ========== 新增：点云补全（保证点数与图像分辨率一致） ==========
            img_height, img_width = depth_img.shape
            expected_points = img_height * img_width
            current_points = np.asarray(pointcloud.points)
            if len(current_points) < expected_points:
                # 补充无效点（z=-1表示无效）
                pad_points = np.ones((expected_points - len(current_points), 3)) * -1
                all_points = np.vstack([current_points, pad_points])
                all_points = np.ascontiguousarray(all_points)  # 新增这行
                pointcloud.points = o3d.utility.Vector3dVector(all_points)

            # 7. 发布点云（保留分辨率信息）
            if len(np.asarray(pointcloud.points)) > 0:  # 确保点云非空
                ros_pc2 = self.open3d_to_ros_pointcloud(
                    pointcloud,
                    "camera_link",
                    depth_img.shape[1],  # 深度图宽度
                    depth_img.shape[0]  # 深度图高度
                )
                self.pub_pointcloud.publish(ros_pc2)
                self.get_logger().info(f"发布原始点云：{len(np.asarray(pointcloud.points))} 个点")
            else:
                self.get_logger().warn("点云为空，未发布")

        except Exception as e:
            self.get_logger().error(f"处理失败：{str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = DataPreprocessingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("节点被用户终止 ❌")
    finally:
        cv2.destroyAllWindows()  # 确保关闭所有OpenCV窗口
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
