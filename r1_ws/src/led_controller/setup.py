from setuptools import setup

package_name = 'led_controller_nodes'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/led_controller']),
        ('share/led_controller', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gsp',
    maintainer_email='user@example.com',
    description='ROS 2 action server node for WS2811 LEDs through FT232H.',
    license='MIT',
)
