from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rebar_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # INSTALL LAUNCH FILES
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotics',
    maintainer_email='user@todo.todo',
    description='Rebar tying logic using moveit_py',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Connects "ros2 run rebar_control rebar_mover" to your python file
            'rebar_mover = rebar_control.rebar_mover_py:main',
        ],
    },
)