import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class OpenArmController(Node):
    def __init__(self):
        super().__init__('openarm_controller')
        # 初始化关节轨迹动作客户端
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'  # 控制器动作名称
        )
        # 等待控制器上线
        while not self.trajectory_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('关节轨迹控制器未上线，等待中...')
        
        # OpenArm关节名称（根据实际模型调整，此处为示例）
        self.joint_names = [
            "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
            "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
            "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
            "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6"
        ]
        self.get_logger().info('OpenArm控制器初始化完成')

    def move_joints(self, target_positions, duration_sec=3.0):
        """
        控制机械臂关节到达目标位置
        :param target_positions: 关节目标位置列表（与joint_names顺序一致）
        :param duration_sec: 运动持续时间（秒）
        """
        if len(target_positions) != len(self.joint_names):
            self.get_logger().error(f"目标位置数量与关节数量不匹配（需{len(self.joint_names)}个，实际{len(target_positions)}个）")
            return

        # 创建轨迹目标消息
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        # 设置轨迹点
        point = JointTrajectoryPoint()
        point.positions = target_positions
        point.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        goal_msg.trajectory.points = [point]

        # 发送目标并等待结果
        self.get_logger().info(f"发送关节目标位置：{target_positions}，持续时间：{duration_sec}秒")
        future = self.trajectory_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('目标被控制器拒绝')
            return
        self.get_logger().info('目标被控制器接受，等待执行完成...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"运动执行完成，结果：{result}")