import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
# 1. 删除错误的 MoveGroupAction 导入，替换为 ROS 2 MoveIt! 正确接口
from moveit_msgs.msg import RobotState
from moveit_msgs.action import MoveGroup  # ROS 2 中 MoveGroup 是 Action 类型，不是 msg
from trajectory_msgs.msg import JointTrajectoryPoint
import action_msgs.msg
from rclpy.action import ActionClient
from rclpy.duration import Duration  # ROS2时间工具

class BimanualPlanningNode(Node):
    def __init__(self):
        super().__init__("bimanual_planning_node")
        # 订阅物体在基坐标系下的坐标（5.4.2发布的）
        self.object_sub = self.create_subscription(
            PointStamped, "/object/base/position", self.object_callback, 10
        )
        
        # 初始化MoveIt! Action客户端（对应双臂的MoveGroup）
        # 注意：group_name必须与OpenArm MoveIt!配置一致（left_arm/right_arm）
        self.left_arm_client = ActionClient(self, MoveGroup, "/left_arm/move_group")
        self.right_arm_client = ActionClient(self, MoveGroup, "/right_arm/move_group")
        
        # 发布抓取触发信号（供夹爪控制节点使用）
        self.grasp_pub = self.create_publisher(PointStamped, "/grasp/trigger", 10)
        
        self.object_position = None
        self.get_logger().info("双臂运动规划节点已启动 ✅")

    def object_callback(self, msg):
        """收到物体坐标后，触发运动规划"""
        self.object_position = msg
        self.get_logger().info(f"收到物体坐标：X={msg.point.x:.3f}, Y={msg.point.y:.3f}, Z={msg.point.z:.3f}")
        # 等待MoveIt! Action服务启动（避免规划失败）
        if not self.left_arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("左臂MoveGroup服务未启动")
            return
        if not self.right_arm_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("右臂MoveGroup服务未启动")
            return
        # 开始规划双臂轨迹
        self.plan_bimanual_grasp()

    def create_move_group_goal(self, group_name, target_pose):
        """创建MoveGroup目标（通用函数，适配左右臂）"""
        goal = MoveGroup.Goal()
        
        # 1. 基础配置
        goal.request.group_name = group_name  # 规划组名称
        goal.request.allowed_planning_time = 5.0  # 规划超时时间（5秒）
        goal.request.planner_id = "RRTConnectkConfigDefault"  # 规划算法（MoveIt!默认）
        
        # 2. 速度/加速度限制（新手建议低速，避免仿真卡顿）
        goal.request.max_velocity_scaling_factor = 0.1  # 最大速度缩放（1.0=满速）
        goal.request.max_acceleration_scaling_factor = 0.1  # 最大加速度缩放
        
        # 3. 目标位姿配置
        goal.request.target_pose.header.frame_id = "base_link"  # 基坐标系
        goal.request.target_pose.header.stamp = self.get_clock().now().to_msg()
        # 目标位置：物体坐标+偏移（避免碰撞）
        goal.request.target_pose.pose.position.x = target_pose.x
        goal.request.target_pose.pose.position.y = target_pose.y
        goal.request.target_pose.pose.position.z = target_pose.z + 0.1  # 高于物体10cm（安全高度）
        # 目标姿态：竖直向下（四元数，w=1表示无旋转，可根据机器人调整）
        goal.request.target_pose.pose.orientation.x = 0.0
        goal.request.target_pose.pose.orientation.y = 1.0
        goal.request.target_pose.pose.orientation.z = 0.0
        goal.request.target_pose.pose.orientation.w = 0.0
        
        # 4. 规划结果要求
        goal.request.plan_only = False  # 规划并执行
        return goal

    def plan_bimanual_grasp(self):
        """规划双臂协作抓取轨迹"""
        if self.object_position is None:
            self.get_logger().warn("物体坐标为空，跳过规划")
            return
        
        # ========== 步骤1：左臂规划（辅助固定） ==========
        left_target = self.object_position.point
        left_target.y += 0.15  # 物体左侧15cm（避免与右臂碰撞）
        left_goal = self.create_move_group_goal("left_arm", left_target)
        # 发送左臂规划请求（异步，不阻塞）
        left_future = self.left_arm_client.send_goal_async(left_goal)
        left_future.add_done_callback(self.left_arm_callback)
        
        # ========== 步骤2：右臂规划（抓取准备） ==========
        right_target = self.object_position.point
        right_target.y -= 0.15  # 物体右侧15cm
        right_goal = self.create_move_group_goal("right_arm", right_target)
        # 发送右臂规划请求
        right_future = self.right_arm_client.send_goal_async(right_goal)
        right_future.add_done_callback(self.right_arm_callback)

    def left_arm_callback(self, future):
        """左臂规划结果回调"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("左臂规划请求被拒绝")
                return
            # 获取规划结果
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.left_arm_result_callback)
        except Exception as e:
            self.get_logger().error(f"左臂规划回调失败：{str(e)}")

    def left_arm_result_callback(self, future):
        """左臂运动结果回调"""
        result = future.result().result
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info("✅ 左臂规划成功，已移动到辅助位置")
             # 新增：打印轨迹点数量，确认轨迹生成
            self.get_logger().info(f"左臂轨迹点数量：{len(result.planned_trajectory.joint_trajectory.points)}")
        else:
            self.get_logger().error(f"❌ 左臂规划失败，错误码：{result.error_code.val}")
            # 错误码说明：1=成功，-1=规划失败，-2=无效目标，-3=超时

    def right_arm_callback(self, future):
        """右臂规划结果回调"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("右臂规划请求被拒绝")
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self.right_arm_result_callback)
        except Exception as e:
            self.get_logger().error(f"右臂规划回调失败：{str(e)}")

    def right_arm_result_callback(self, future):
        """右臂运动结果回调：成功后触发抓取"""
        result = future.result().result
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info("✅ 右臂规划成功，已移动到抓取准备位置")
            self.get_logger().info(f"右臂轨迹点数量：{len(result.planned_trajectory.joint_trajectory.points)}")
            # 触发夹爪闭合（发布抓取信号）
            self.trigger_grasp()
        else:
            self.get_logger().error(f"❌ 右臂规划失败，错误码：{result.error_code.val}")

    def trigger_grasp(self):
        """发布抓取触发信号"""
        grasp_trigger = PointStamped()
        grasp_trigger.header.frame_id = "base_link"
        grasp_trigger.point.x = 1.0  # 约定：x=1→抓取，x=0→释放
        self.grasp_pub.publish(grasp_trigger)
        self.get_logger().info("📢 已发布抓取触发信号，夹爪即将闭合")

def main(args=None):
    rclpy.init(args=args)
    node = BimanualPlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
