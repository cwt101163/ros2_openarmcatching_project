import os
from glob import glob
from setuptools import setup

package_name = 'image_comm'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='breeze',
    maintainer_email='breeze@todo.todo',
    description='ROS2 Image Communication',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_pub = image_comm.image_publisher:main',
            'image_sub = image_comm.image_subscriber:main',
        ],
    },
    package_data={
        package_name: ['msg/*.msg'],
    },
)
