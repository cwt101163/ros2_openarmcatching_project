from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_openarm_moveit_config = FindPackageShare('openarm_moveit_config')
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')

    # 获取 .xacro 文件路径
    xacro_file = PathJoinSubstitution([
        pkg_openarm_moveit_config,
        'config',
        'openarm.urdf.xacro'
    ])

    # 使用 xacro 命令解析为 URDF 字符串
    robot_description = Command(['xacro ', xacro_file])

    gui_arg = DeclareLaunchArgument(
        name='gui',
        default_value='true',
        description='Set to "false" to run headless.'
    )

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_gazebo_ros, 'launch', 'gazebo.launch.py'])
        ]),
        launch_arguments={'gui': LaunchConfiguration('gui')}.items(),
    )

    # spawn_entity 使用 -topic 方式
    # 或直接传入 robot_description 内容
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'openarm',
            '-topic', '/robot_description',  # 从 topic 读取
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='both',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_description}  # 传入解析后的 URDF
        ]
    )

    return LaunchDescription([
        gui_arg,
        gazebo_launch,
        robot_state_publisher,  
        spawn_entity,
    ])