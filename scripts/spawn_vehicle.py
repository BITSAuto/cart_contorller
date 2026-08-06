#!/usr/bin/env python3
"""
CARLA Heading Spawner
=====================
Spawns a golf-cart vehicle with RGB camera, LiDAR, and an IMU co-located with
the LiDAR (simulating a combined LiDAR+IMU unit like Ouster/Livox).

Based on carla_game_spawn.py.  The IMU publishes to:
    /carla/hero/imu    (sensor_msgs/Imu)

The heading driver (carla_heading.py) reads this topic to display the
car's real heading and compare it with the estimated road direction.

Usage:
    python carla_heading_spawn.py --ros2
    python carla_heading_spawn.py --ros2 --config my_config.json
"""

from __future__ import print_function

import sys
import os
import argparse
import json
import time
import math
from datetime import datetime
from pathlib import Path
import numpy as np

import carla

try:
    import rclpy
    from rclpy.node import Node as ROS2Node
    from carla_msgs.msg import CarlaEgoVehicleControl
    from std_msgs.msg import Float32
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


# =============================================================================
# Game Control Bridge  — persistent wheel angle, no snap-back
# =============================================================================

if ROS2_AVAILABLE:
    class GameControlBridge(ROS2Node):
        """
        Subscribes to /carla/<role>/vehicle_control_cmd and applies control
        to the CARLA vehicle, enforcing golf cart constraints.

        Maintains persistent wheel angle so CARLA doesn't snap wheels back to
        center when control isn't applied every tick.
        """

        CRUISE_SPEED_KMH = 10.0   # Fixed target speed

        def __init__(self, vehicle, role_name='hero', constraints=None):
            super().__init__(f'{role_name}_heading_control_bridge')
            self.vehicle = vehicle
            self.constraints = constraints or {}
            self._latest = None

            # Persistent wheel state
            self._current_steer = 0.0   # CARLA steer value [-1, 1]

            # Constraint values
            self.max_steer_deg = float(
                self.constraints.get('max_steer_angle_deg', 20.0))
            self.max_speed_kmh = float(
                self.constraints.get('max_speed_kmh', 15.0))

            # CARLA's max physical steer is ~70 degrees
            self.steer_limit = min(self.max_steer_deg / 70.0, 1.0)

            topic = f'/carla/{role_name}/vehicle_control_cmd'
            self.sub = self.create_subscription(
                CarlaEgoVehicleControl, topic, self._on_control, 10)
            self.get_logger().info(f'Subscribed to {topic}')

            # Publish the current wheel angle for the driver to read
            self.steer_pub = self.create_publisher(
                Float32, f'/carla/{role_name}/current_steer_angle', 10)

            # Publish speed (m/s) so the driver node can read it
            self.speed_pub = self.create_publisher(
                Float32, f'/carla/{role_name}/speedometer', 10)

        def _on_control(self, msg):
            """Read only steering from the driver."""
            self._latest = msg

        def apply_control(self):
            """
            Use throttle from driver if set, else enforce 10 km/h. Steering from driver.
            """
            ctrl = carla.VehicleControl()

            # 1. Steering
            if self._latest is not None:
                new_steer = float(
                    np.clip(self._latest.steer, -self.steer_limit, self.steer_limit))
                self._current_steer = new_steer
            
            ctrl.steer = self._current_steer

            # 2. Speed Control (Target 10 km/h)
            vel = self.vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            if self._latest is not None and self._latest.throttle > 0.0:
                # Use throttle from driver
                ctrl.throttle = self._latest.throttle
                ctrl.brake = 0.0
            else:
                # Enforce 10 km/h
                speed_error = self.CRUISE_SPEED_KMH - speed_kmh

                if speed_error > 0:
                    # Need to speed up
                    ctrl.throttle = float(min(speed_error * 0.4, 1.0))
                    ctrl.brake    = 0.0
                else:
                    # Need to slow down
                    ctrl.throttle = 0.0
                    ctrl.brake    = float(min(abs(speed_error) * 0.2, 1.0))

            # Absolute speed clamp
            if speed_kmh > self.max_speed_kmh:
                ctrl.throttle = 0.0
                ctrl.brake    = 1.0

            # CRITICAL: Force automatic transmission and release handbrake 
            # otherwise the CARLA simulator will hold the car rigidly in place
            ctrl.gear = 1
            ctrl.manual_gear_shift = False
            ctrl.hand_brake = False
            ctrl.reverse = False

            if self._latest is not None:
                ctrl.hand_brake = self._latest.hand_brake
                ctrl.reverse    = self._latest.reverse
                
            self.vehicle.apply_control(ctrl)

            # 3. Diagnostics publishing
            angle_deg = self._current_steer * 70.0
            steer_msg = Float32(data=float(angle_deg))
            self.steer_pub.publish(steer_msg)

            speed_msg = Float32(data=float(speed_kmh / 3.6)) # Publish as m/s
            self.speed_pub.publish(speed_msg)


# =============================================================================
# Config & Spawning
# =============================================================================

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    if 'type' not in config:
        print('[ERROR] Config must have a "type" field.')
        sys.exit(1)
    return config


def spawn_actors(world, config, ros2=False):
    """Spawn vehicle + sensors.  Returns (vehicle, sensor_actors, role_name)."""
    bp_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    if not spawn_points:
        print('[ERROR] No spawn points on this map.')
        sys.exit(1)

    vehicle_type = config.get('type', 'vehicle.*')
    role_name = config.get('id', 'hero')

    bp = bp_library.filter(vehicle_type)[0]
    bp.set_attribute('role_name', role_name)
    bp.set_attribute('ros_name', role_name)

    color = config.get('color')
    if color and bp.has_attribute('color'):
        bp.set_attribute('color', str(color))

    vehicle = world.spawn_actor(bp, spawn_points[0])
    print(f'\n[VEHICLE] {vehicle.type_id}')
    print(f'  Spawn: ({spawn_points[0].location.x:.1f}, '
          f'{spawn_points[0].location.y:.1f}, '
          f'{spawn_points[0].location.z:.1f})')
    print(f'  Role:  {role_name}')

    # Sensors
    sensors = []
    for s_idx, s_cfg in enumerate(config.get('sensors', [])):
        s_type = s_cfg.get('type')
        s_id = s_cfg.get('id', f'sensor_{s_idx}')
        sp = s_cfg.get('spawn_point', {})
        attrs = s_cfg.get('attributes', {})

        s_bp = bp_library.filter(s_type)[0]
        s_bp.set_attribute('ros_name', s_id)
        s_bp.set_attribute('role_name', s_id)
        for k, v in attrs.items():
            s_bp.set_attribute(str(k), str(v))

        tf = carla.Transform(
            location=carla.Location(
                x=sp.get('x', 0.0),
                y=-sp.get('y', 0.0),
                z=sp.get('z', 0.0)),
            rotation=carla.Rotation(
                roll=sp.get('roll', 0.0),
                pitch=-sp.get('pitch', 0.0),
                yaw=-sp.get('yaw', 0.0))
        )

        sensor = world.spawn_actor(s_bp, tf, attach_to=vehicle)
        sensors.append(sensor)

        if ros2:
            sensor.enable_for_ros()
            print(f'  [SENSOR] {s_id} ({s_type})')
            print(f'           ROS2: /carla/{role_name}/{s_id}')
        else:
            sensor.listen(lambda data: None)
            print(f'  [SENSOR] {s_id} ({s_type})')

    return vehicle, sensors, role_name


# =============================================================================
# Main
# =============================================================================

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Heading Spawner — spawns vehicle with LiDAR+IMU unit')
    argparser.add_argument('--config', default=None,
                           help='JSON config (default: carla_heading_config.json)')
    argparser.add_argument('--host', default='localhost')
    argparser.add_argument('-p', '--port', default=2000, type=int)
    argparser.add_argument('--ros2', action='store_true',
                           help='Enable ROS2 native topics')
    args = argparser.parse_args()

    if args.config is None:
        if ROS2_AVAILABLE:
            try:
                from ament_index_python.packages import get_package_share_directory
                pkg_share = get_package_share_directory('cart_controller')
                args.config = os.path.join(pkg_share, 'config', 'spawn_config.json')
            except Exception as e:
                print(f"[WARN] Could not find cart_controller share dir: {e}")
                args.config = str(Path(__file__).parent / 'carla_heading_config.json')
        else:
            args.config = str(Path(__file__).parent / 'carla_heading_config.json')

    # Connect
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()

    config = load_config(args.config)
    print(f'[OK] Loaded config: {args.config}')

    # Report sensor layout
    sensors_cfg = config.get('sensors', [])
    imu_found = any(s.get('type') == 'sensor.other.imu' for s in sensors_cfg)
    lidar_found = any('lidar' in s.get('type', '') for s in sensors_cfg)
    print(f'[INFO] Sensors: LiDAR={lidar_found}  IMU={imu_found} '
          f'(co-located at LiDAR mount)')

    vehicle = None
    sensors = []
    original_settings = None
    control_bridge = None

    try:
        # World settings
        world_cfg = config.get('world', {})
        target_map = world_cfg.get('map')
        if target_map:
            current_map = world.get_map().name.split('/')[-1]
            if current_map != target_map:
                print(f'[MAP] Loading {target_map}...')
                world = client.load_world(target_map)
                print(f'[MAP] {target_map} loaded')
            else:
                print(f'[MAP] Already on {target_map}')

        original_settings = world.get_settings()
        settings = world.get_settings()
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)

        settings.synchronous_mode = world_cfg.get('synchronous_mode', True)
        settings.fixed_delta_seconds = world_cfg.get('fixed_delta_seconds', 0.05)
        settings.no_rendering_mode = world_cfg.get('no_rendering_mode', False)
        world.apply_settings(settings)

        vehicle, sensors, role_name = spawn_actors(world, config, ros2=ROS2_AVAILABLE)

        constraints = config.get('constraints', {})
        print(f'\n[CONSTRAINTS]')
        for k, v in constraints.items():
            print(f'  {k}: {v}')

        # ROS2 control bridge
        if ROS2_AVAILABLE:
            try:
                rclpy.init()
                control_bridge = GameControlBridge(
                    vehicle, role_name=role_name, constraints=constraints)
                print(f'\n[ROS2] Heading Control Bridge active for "{role_name}"')
                print(f'[ROS2] IMU topic: /carla/{role_name}/imu')
                print(f'[ROS2] Wheel persistence: ENABLED (no snap-back)')
            except Exception as e:
                print(f'[WARN] Could not start ROS2 bridge: {e}')
                control_bridge = None

        print(f'\n{"=" * 60}')
        print(f' Heading spawner running (Ctrl+C to stop)')
        print(f' IMU topic:  /carla/{config.get("id", "hero")}/imu')
        print(f' LiDAR topic: /carla/{config.get("id", "hero")}/lidar')
        print(f'{"=" * 60}\n')

        frame = 0
        dt0 = datetime.now()

        while True:
            # Apply control before tick
            if control_bridge is not None:
                rclpy.spin_once(control_bridge, timeout_sec=0)
                control_bridge.apply_control()

            world.tick()
            frame += 1

            # Status line
            elapsed = (datetime.now() - dt0).total_seconds()
            fps = 1.0 / max(elapsed, 1e-6)
            vel = vehicle.get_velocity()
            speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            ctrl = vehicle.get_control()
            steer_deg = ctrl.steer * 70.0

            sys.stdout.write(
                f'\rFrame: {frame} | FPS: {fps:.0f} | '
                f'Speed: {speed:.1f} km/h | '
                f'Steer: {steer_deg:+.1f}° | '
                f'Throttle: {ctrl.throttle:.2f}    ')
            sys.stdout.flush()

            dt0 = datetime.now()
            time.sleep(0.005)

    except KeyboardInterrupt:
        print('\n\n--- Stopping heading spawner ---')

    finally:
        if control_bridge is not None:
            control_bridge.destroy_node()
            print('[OK] Control bridge destroyed')
        try:
            rclpy.shutdown()
        except Exception:
            pass

        for s in sensors:
            if s is not None:
                s.stop()
                s.destroy()
        if vehicle is not None:
            vehicle.destroy()
            print(f'[OK] Destroyed vehicle: {vehicle.type_id}')

        if original_settings is not None:
            world.apply_settings(original_settings)
            tm = client.get_trafficmanager(8000)
            tm.set_synchronous_mode(False)
            print('[OK] World settings restored')


if __name__ == '__main__':
    main()
