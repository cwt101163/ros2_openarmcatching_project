from setuptools import find_packages, setup

package_name = 'grasp_planning'

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
    maintainer='breeze',
    maintainer_email='breeze@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'bimanual_planning = grasp_planning.bimanual_planning:main',
            'gripper_control = grasp_planning.gripper_control:main',
        ],
    },
)
