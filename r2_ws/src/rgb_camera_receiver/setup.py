from glob import glob
from setuptools import setup

package_name = 'rgb_camera_receiver'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robocon',
    maintainer_email='robocon@example.com',
    description='Protocol-independent five-color LED strip detector for R2.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'rgb_camera_receiver = rgb_camera_receiver.node:main',
            'evaluate_led_dataset = rgb_camera_receiver.evaluate:main',
            'calibrate_led_colors = rgb_camera_receiver.calibrate:main',
        ],
    },
)
