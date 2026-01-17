#这个脚本是你的MuJoCo 仿真启动入口，它的功能是：
#加载机器人模型文件（URDF/XML）
#初始化 MuJoCo 的仿真环境
#启动可视化界面，让你能在仿真中操作机械臂、相机等模型
#它是你后续实验（眼手标定、视觉检测、抓取规划）的基础仿真工具，所以必须让它正确加载包含相机、桌子的完整模型。


import mujoco
import mujoco.viewer

##一编
## 原来的代码（加载简化模型）
##model = mujoco.MjModel.from_xml_path('openarm_bimanual.xml')
##修改为加载你的完整URDF（包含相机和桌子）
#model = mujoco.MjModel.from_xml_path('openarm_bimanual_cam.urdf')

#二编,让脚本加载编译好的文件，就不用反复修改和编译
model = mujoco.MjModel.from_xml_path('/home/breeze/ros2_ws/install/openarm_moveit_config/share/openarm_moveit_config/config/openarm_bimanual_cam.urdf')


data = mujoco.MjData(model)

# 启动完整交互式 Viewer
mujoco.viewer.launch(model, data)
