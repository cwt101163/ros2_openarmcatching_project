from setuptools import find_packages, setup

package_name = 'openarm_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ldp',
    maintainer_email='ldp@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'openarm_control_node = openarm_control.openarm_node:main',
            # 'send_joint_goal = openarm_control.send_joint_goal:main',    # 添加新节点
            'send_left_arm_goal = openarm_control.send_joint_goal:main',  # 新增左臂控制入口
        ],
    },
)
