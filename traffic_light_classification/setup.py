from setuptools import setup
import os

package_name = 'traffic_light_classification'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eren',
    maintainer_email='eren.durgunlu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'classificator = traffic_light_classification.traffic_light_classification:main',
            'time_correct = traffic_light_classification.rosbag_time_correct:main'
        ],
    },
)
