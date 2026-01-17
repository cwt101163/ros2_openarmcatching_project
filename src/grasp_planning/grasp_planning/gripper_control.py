import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState  # 夹爪状态监听

class GripperControlNode(Node):
    def __init__(self):
        super().__init__("gripper_control_node")
        # 订阅抓取触发信号（5.5.1发布的）
        self.grasp_sub = self.create_subscription(
            PointStamped, "/grasp/trigger", self.grasp_callback, 10
        )
        # 订阅夹爪状态（用于验证控制是否生效）
        self.gripper_state_sub = self.create_subscription(
            JointTrajectoryControllerState, "/right_arm/gripper_controller/state", self.state_callback, 10
        )
        
        # 发布夹爪控制指令（话题名必须与机器人控制器配置一致）
        self.gripper_pub = self.create_publisher(
            JointTrajectory, "/right_arm/gripper_controller/joint_trajectory", 10
        )
        
        self.current_gripper_pos = 1.0  # 初始状态：夹爪打开（1.0）
        self.get_logger().info("夹爪控制节点已启动 ✅")

    def state_callback(self, msg):
        """监听夹爪当前位置"""
        if len(msg.actual.positions) > 0:
            self.current_gripper_pos = msg.actual.positions[0]
            self.get_logger().debug(f"当前夹爪位置：{self.current_gripper_pos:.3f}")  # 调试日志

    def grasp_callback(self, msg):
        """抓取触发回调：根据信号闭合/打开夹爪"""
        if msg.point.x == 1.0:
            # 检查当前夹爪状态，避免重复控制
            if abs(self.current_gripper_pos - 0.0) < 0.01:
                self.get_logger().info("夹爪已闭合，无需重复操作")
                return
            self.close_gripper()
        elif msg.point.x == 0.0:
            if abs(self.current_gripper_pos - 1.0) < 0.01:
                self.get_logger().info("夹爪已打开，无需重复操作")
                return
            self.open_gripper()
        else:
            self.get_logger().warn(f"无效的抓取信号：{msg.point.x}，仅支持1（抓取）/0（释放）")

    def close_gripper(self):
        """控制夹爪闭合"""
        # 创建关节轨迹消息
        traj = JointTrajectory()
        # 修正：关节名匹配joint_limits.yaml
        traj.joint_names = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]# 夹爪关节名（必须与URDF一致）
        traj.header.frame_id = "base_link"
        traj.header.stamp = self.get_clock().now().to_msg()
        
        # 轨迹点配置：2秒内闭合到0.0（完全闭合）
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.0]  # 闭合位置（0.0=完全闭合，需根据机器人调整）# 两个夹爪关节都闭合
        point.velocities = [0.5, 0.5]  # 闭合速度（0.5 rad/s，低速更稳定）# 两个夹爪关节都闭合
        point.time_from_start.sec = 2  # 完成时间：2秒
        point.time_from_start.nanosec = 0
        
        traj.points.append(point)
        # 发布控制指令
        self.gripper_pub.publish(traj)
        self.get_logger().info("🔒 发布夹爪闭合指令，2秒后完成")

    def open_gripper(self):
        """控制夹爪打开"""
        traj = JointTrajectory()
        traj.joint_names = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]
        traj.header.frame_id = "base_link"
        traj.header.stamp = self.get_clock().now().to_msg()
        
        # 轨迹点配置：2秒内打开到1.0（完全打开）
        point = JointTrajectoryPoint()
        point.positions = [1.0, 1.0]  # 打开位置 # 两个夹爪关节都打开
        point.velocities = [0.5, 0.5]
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0
        
        traj.points.append(point)
        self.gripper_pub.publish(traj)
        self.get_logger().info("🔓 发布夹爪打开指令，2秒后完成")

def main(args=None):
    rclpy.init(args=args)
    node = GripperControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
