import rclpy
from rclpy.node import Node
from openarm_control.arm_controller import OpenArmController


class OpenArmNode(Node):
    def __init__(self):
        super().__init__('openarm_node')
        self.controller = OpenArmController()  # 实例化控制器
        
        # 示例：启动后执行一个抬起双臂的动作
        self.get_logger().info("开始执行抬起双臂动作...")
        # 目标位置：左臂和右臂关节抬起（根据实际关节范围调整角度值）
        target_positions = [
            0.0, 1.5, -0.8, 0.0, 0.0, 0.0,  # 左臂关节1-6
            0.0, 1.5, 0.8, 0.0, 0.0, 0.0    # 右臂关节1-6（注意对称性）
        ]
        self.controller.move_joints(target_positions, duration_sec=2.0)  # 2秒内完成


def main(args=None):
    rclpy.init(args=args)
    node = OpenArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('用户中断，退出程序')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()