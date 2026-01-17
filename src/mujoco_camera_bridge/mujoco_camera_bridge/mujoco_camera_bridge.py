import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import mujoco
import numpy as np
from builtin_interfaces.msg import Time


class MujocoCameraBridge(Node):
    def __init__(self):
        super().__init__('mujoco_camera_bridge')

        # Publishers
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.bridge = CvBridge()

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(
            '/home/breeze/ros2_ws/src/openarm_moveit_config/config/openarm_bimanual.xml')
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, width=320, height=240)

        # 标记当前模式（仿真/实物）
        self.mode = "simulation"

        # Get joint names
        self.joint_names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and 'free' not in name:
                self.joint_names.append(name)
        self.get_logger().info(f"Found joints: {self.joint_names}")

        # 预生成深度图掩码（模拟物体深度）
        self.depth_mask = self.generate_depth_mask()

        # Timer
        self.timer = self.create_timer(1.0 / 30.0, self.publish_data)

    def generate_depth_mask(self):
        """生成模拟物体的深度掩码"""
        mask = np.zeros((240, 320), dtype=np.bool_)
        y, x = np.ogrid[:240, :320]
        # 模拟物体区域（椭圆）
        mask[(x - 160) ** 2 / 100 + (y - 120) ** 2 / 100 <= 1] = True
        return mask

    def process_depth(self):
        """构造深度图：区分仿真/实物"""
        # 背景深度1米，物体深度0.5米
        depth = np.ones((240, 320), dtype=np.float32)
        depth[self.depth_mask] = 0.5

        if self.mode == "real":
            # 实物模式添加噪声
            depth += np.random.normal(0, 0.01, depth.shape)
            from scipy.ndimage import median_filter
            depth = median_filter(depth, size=3)

        # 转为毫米格式
        return (depth * 1000).astype(np.uint16)

    def publish_data(self):
        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # === 发布RGB图（保留原MuJoCo渲染） ===
        self.renderer.update_scene(self.data, camera='d435_front')
        rgb = self.renderer.render()
        img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera_link"
        self.image_pub.publish(img_msg)

        # === 发布深度图（构造，不依赖MuJoCo渲染） ===
        depth_processed = self.process_depth()
        depth_msg = self.bridge.cv2_to_imgmsg(depth_processed, encoding="16UC1")
        depth_msg.header.stamp = img_msg.header.stamp
        depth_msg.header.frame_id = "camera_link"
        self.depth_pub.publish(depth_msg)

        # === 发布关节状态 ===
        joint_state = JointState()
        now = self.get_clock().now().to_msg()
        joint_state.header.stamp = now
        joint_state.name = self.joint_names
        qpos = []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            addr = self.model.jnt_qposadr[jid]
            qpos.append(self.data.qpos[addr])
        joint_state.position = qpos
        self.joint_state_pub.publish(joint_state)

        self.get_logger().info(f'[{self.mode}] Published data', throttle_duration_sec=1)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoCameraBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
