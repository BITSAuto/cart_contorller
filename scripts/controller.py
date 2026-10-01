#!/usr/bin/env python3
"""
Cart Controller — Low-Level Control Layer
Sole purpose: Enforce strict hardware constraints (30ms response + STOP).
Exposes simple pulse_right/pulse_left/hold API for any navigator script.
"""

import math
import re
import threading
import time
import numpy as np

try:
    import serial
except ImportError:
    serial = None
    print('[cart_controller] WARNING: pyserial not installed — cart serial output will be DISABLED')

try:
    from carla_msgs.msg import CarlaEgoVehicleControl
except ImportError:
    CarlaEgoVehicleControl = None

from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

PULSE_MS           = 35.0     # Enforced minimum hardware pulse!
MAX_STEER_DEG      = 15.0
DEG_PER_PULSE      = 0.45
STEER_CMD_SPEED    = 15.0

MOTOR_RIGHT        = 51
MOTOR_LEFT         = 49
MOTOR_STOP         = 50

MAX_SPEED_KMH      = 7.0
THROTTLE           = 0.4
TARGET_THROTTLE    = 75
WHEELBASE_M        = 1.8
IMU_EMA_ALPHA      = 0.3
SERIAL_PORT        = '/dev/ttyUSB0'
SERIAL_BAUD        = 9600

# Steering encoder (ESP32). Set ENCODER_PORT = SERIAL_PORT if the counts come
# back on the steering command link instead of a separate USB device.
ENCODER_PORT       = '/dev/ttyUSB1'
ENCODER_BAUD       = 115200
ENCODER_MAX_RIGHT  = -2500    # Count at full right lock
ENCODER_MAX_LEFT   = 2500     # Count at full left lock
ENCODER_MARGIN     = 150      # Stop pulsing this far before lock (~2 pulses of motor run-on)
ENCODER_TIMEOUT_S  = 0.5      # Older reading = encoder lost, steering pulses blocked
ENCODER_CENTER     = (ENCODER_MAX_LEFT + ENCODER_MAX_RIGHT) / 2.0
ENCODER_HALF_RANGE = (ENCODER_MAX_LEFT - ENCODER_MAX_RIGHT) / 2.0

def build_serial_command(
    a_val=0, b_throttle=50, c_val=0, d_steering=50,
    e_left_indicator=0, f_horn=0, g_light=0,
    h_right_indicator=0, i_brake=0, j_reverse=0
):
    return (
        f"*A{a_val}B{b_throttle}C{c_val}D{d_steering}E{e_left_indicator}"
        f"F{f_horn}G{g_light}H{h_right_indicator}I{i_brake}J{j_reverse}#"
    )

class MotorSteering:
    """Discrete Hardware Abstraction Layer.
    Never modify this logic. It enforces a strict lifecycle:
    IDLE -> 51/49 for 35ms -> 50 (STOP) -> IDLE.
    """
    _IDLE     = 'IDLE'
    _PULSING  = 'PULSING'
    _STOPPING = 'STOPPING'

    def __init__(self):
        self.motor = MOTOR_STOP
        self._phase = self._IDLE
        self._pulse_start_ms = 0.0
        self.simulated_angle = 0.0

    def tick(self):
        if self._phase == self._PULSING:
            now_ms = time.time() * 1000.0
            if (now_ms - self._pulse_start_ms) >= PULSE_MS:
                self.motor = MOTOR_STOP
                self._phase = self._STOPPING
        elif self._phase == self._STOPPING:
            self._phase = self._IDLE
        return self.motor

    @property
    def is_idle(self):
        return self._phase == self._IDLE

    def hold(self):
        self.motor = MOTOR_STOP
        self._phase = self._IDLE

    def pulse_right(self):
        if self._phase != self._IDLE: return False
        self.motor = MOTOR_RIGHT
        self._phase = self._PULSING
        self._pulse_start_ms = time.time() * 1000.0
        self.simulated_angle = min(MAX_STEER_DEG, self.simulated_angle + DEG_PER_PULSE)
        return True

    def pulse_left(self):
        if self._phase != self._IDLE: return False
        self.motor = MOTOR_LEFT
        self._phase = self._PULSING
        self._pulse_start_ms = time.time() * 1000.0
        self.simulated_angle = max(-MAX_STEER_DEG, self.simulated_angle - DEG_PER_PULSE)
        return True

class SteeringEncoder:
    """Background reader for the steering encoder count streamed over serial.
    Takes the first integer on each line, so '-1234' and 'Encoder: -1234' both work.
    """
    _NUM_RE = re.compile(r'(?<![\w.])[-+]?\d+(?![\w.])')

    def __init__(self, ser, logger):
        self.ser = ser
        self.logger = logger
        self.count = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        buf = b''
        while not self._stop.is_set():
            try:
                buf += self.ser.readline()
            except Exception as e:
                if not self._stop.is_set():
                    self.logger.error(f'Encoder serial read failed: {e}')
                return
            # readline() returns partial data on timeout; only parse complete lines
            if not buf.endswith(b'\n'):
                if len(buf) > 256: buf = b''
                continue
            match = self._NUM_RE.search(buf.decode('utf-8', errors='ignore'))
            buf = b''
            if match:
                self.count = int(match.group())
                self._stamp = time.time()
                self.logger.info(f'[Encoder] count={self.count}', throttle_duration_sec=5.0)

    @property
    def ok(self):
        return self.count is not None and (time.time() - self._stamp) < ENCODER_TIMEOUT_S

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=0.5)

def quat_to_yaw_deg(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw_rad) % 360.0

TOPICS = {
    'carla': { 'rgb': '/carla/hero/rgb/image', 'imu': '/carla/hero/imu', 'control': '/carla/hero/vehicle_control_cmd', 'steer': '/carla/hero/current_steer_angle' },
    'cart':  { 'rgb': '/bitsauto/rgb/image',   'imu': '/bitsauto/imu',   'control': None,                              'steer': '/bitsauto/current_steer_angle' },
}

class CartController:
    def __init__(self, node, mode='carla'):
        self.node = node
        self.mode = mode
        self.topics = TOPICS[mode]
        self.motor = MotorSteering()

        self.car_heading_deg = None
        self.smoothed_yaw_rate = 0.0
        self.actual_yaw_rate = 0.0
        self._imu_ok = False

        self.ser = None
        self.ctrl_pub = None
        self.encoder = None
        self._last_serial_time = 0.0

        self._setup_imu()
        self._setup_control_output()
        self._setup_encoder()

        # --- Debug: startup diagnostics ---
        self.node.get_logger().info(
            f'[Controller] ===== STARTUP SUMMARY =====\n'
            f'  Mode:      {mode.upper()}\n'
            f'  Serial:    {"OPEN on " + SERIAL_PORT if self.ser and self.ser.is_open else "NOT AVAILABLE"}\n'
            f'  Encoder:   {"ACTIVE on " + ENCODER_PORT if self.encoder else "NOT AVAILABLE"}\n'
            f'  CARLA pub: {"ACTIVE" if self.ctrl_pub else "N/A"}\n'
            f'  pyserial:  {"installed" if serial else "NOT INSTALLED"}'
        )

    def _setup_imu(self):
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
        self.node.create_subscription(Imu, self.topics['imu'], self._on_imu, qos)

    def _setup_control_output(self):
        if self.mode == 'carla' and CarlaEgoVehicleControl and self.topics['control']:
            self.ctrl_pub = self.node.create_publisher(CarlaEgoVehicleControl, self.topics['control'], 10)
        elif self.mode == 'cart' and serial is not None:
            try:
                self.ser = serial.Serial(port=SERIAL_PORT, baudrate=SERIAL_BAUD, timeout=0.05)
                self.node.get_logger().info(f'[Controller] Serial port OPENED: {SERIAL_PORT} @ {SERIAL_BAUD} baud')
            except Exception as e:
                self.node.get_logger().warn(f'Serial failed: {e}')

    def _setup_encoder(self):
        if self.mode != 'cart' or serial is None: return
        if ENCODER_PORT == SERIAL_PORT:
            if self.ser is None: return
            enc_ser = self.ser
        else:
            try:
                enc_ser = serial.Serial(port=ENCODER_PORT, baudrate=ENCODER_BAUD, timeout=0.1)
            except Exception as e:
                self.node.get_logger().warn(f'Encoder serial failed: {e} (steering pulses blocked)')
                return
        self.encoder = SteeringEncoder(enc_ser, self.node.get_logger())
        self.node.get_logger().info(f'[Controller] Encoder reader STARTED on {enc_ser.port} @ {enc_ser.baudrate} baud')

    def _on_imu(self, msg):
        try:
            self.car_heading_deg = quat_to_yaw_deg(msg.orientation)
            raw_yaw = -math.degrees(msg.angular_velocity.z)
            self.actual_yaw_rate = raw_yaw
            if not self._imu_ok:
                self.smoothed_yaw_rate = raw_yaw
                self._imu_ok = True
            else:
                self.smoothed_yaw_rate = IMU_EMA_ALPHA * raw_yaw + (1 - IMU_EMA_ALPHA) * self.smoothed_yaw_rate
        except Exception as e:
            self._imu_ok = False
            self.node.get_logger().error(f'[IMU] Processing FAILED: {e}', throttle_duration_sec=5.0)

    @property
    def imu_ok(self): return self._imu_ok

    @property
    def imu_steer_deg(self):
        if not self._imu_ok: return 0.0
        speed_ms = MAX_SPEED_KMH / 3.6
        if speed_ms < 0.3: return 0.0
        return math.degrees(math.atan2(math.radians(self.smoothed_yaw_rate) * WHEELBASE_M, speed_ms))

    @property
    def encoder_count(self):
        """Steering encoder count (+ = left), or None if stale. CARLA derives it from simulated_angle."""
        if self.mode == 'carla':
            return int(round(ENCODER_CENTER - self.motor.simulated_angle / MAX_STEER_DEG * ENCODER_HALF_RANGE))
        if self.encoder is None or not self.encoder.ok: return None
        return self.encoder.count

    @property
    def encoder_ok(self): return self.encoder_count is not None

    @property
    def encoder_steer_deg(self):
        """Measured wheel angle, right positive (same sign as simulated_angle)."""
        count = self.encoder_count
        if count is None: return None
        return -(count - ENCODER_CENTER) / ENCODER_HALF_RANGE * MAX_STEER_DEG

    def can_pulse_right(self):
        count = self.encoder_count
        return count is not None and count > ENCODER_MAX_RIGHT + ENCODER_MARGIN

    def can_pulse_left(self):
        count = self.encoder_count
        return count is not None and count < ENCODER_MAX_LEFT - ENCODER_MARGIN

    def pulse_right(self):
        if not self.can_pulse_right():
            self.node.get_logger().warn(
                f'[Steer] pulse_right BLOCKED | encoder_count={self.encoder_count} '
                f'encoder_present={"yes" if self.encoder else "NO"} '
                f'encoder_ok={self.encoder_ok}',
                throttle_duration_sec=2.0)
            return False
        return self.motor.pulse_right()

    def pulse_left(self):
        if not self.can_pulse_left():
            self.node.get_logger().warn(
                f'[Steer] pulse_left BLOCKED | encoder_count={self.encoder_count} '
                f'encoder_present={"yes" if self.encoder else "NO"} '
                f'encoder_ok={self.encoder_ok}',
                throttle_duration_sec=2.0)
            return False
        return self.motor.pulse_left()

    def tick(self):
        if self.mode == 'cart':
            if self.encoder_ok:
                self.motor.simulated_angle = self.encoder_steer_deg
            else:
                self.node.get_logger().warn('Steering encoder not reporting, pulses blocked', throttle_duration_sec=2.0)
        self.motor.tick()
        self._send_hw()

        # --- Debug: periodic state dump ---
        self.node.get_logger().info(
            f'[Tick] motor={self.motor.motor} phase={self.motor._phase} '
            f'sim_angle={self.motor.simulated_angle:+.1f}\u00b0 | '
            f'enc={self.encoder_count} enc_ok={self.encoder_ok} | '
            f'imu={self._imu_ok} yaw={self.smoothed_yaw_rate:+.1f}\u00b0/s | '
            f'serial={"open" if self.ser and self.ser.is_open else "CLOSED/NONE"}',
            throttle_duration_sec=3.0)

    def hold(self):
        self.motor.hold()
        self._send_hw()

    def _send_hw(self):
        if self.ctrl_pub and CarlaEgoVehicleControl:
            msg = CarlaEgoVehicleControl()
            msg.header = Header()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.steer = float(np.clip(self.motor.simulated_angle / MAX_STEER_DEG, -1.0, 1.0))
            msg.throttle = THROTTLE
            msg.brake = 0.0
            msg.hand_brake = False
            msg.reverse = False
            msg.gear = 1
            msg.manual_gear_shift = False
            self.ctrl_pub.publish(msg)

        if self.ser and self.ser.is_open:
            now = time.time()
            if (now - self._last_serial_time) >= 0.030:
                self._last_serial_time = now
                serial_cmd = build_serial_command(b_throttle=TARGET_THROTTLE, d_steering=self.motor.motor, i_brake=0)
                self.node.get_logger().debug(
                    f'[Serial TX] {serial_cmd}',
                    throttle_duration_sec=5.0)
                try:
                    self.ser.write(serial_cmd.encode())
                    # When the encoder shares this port its reader thread drains the replies
                    if self.encoder is None or self.encoder.ser is not self.ser:
                        while self.ser.readline(): pass
                except Exception as e:
                    self.node.get_logger().error(
                        f'[Serial] Write FAILED: {e}', throttle_duration_sec=2.0)
        elif self.mode == 'cart':
            self.node.get_logger().warn(
                f'[Serial] Cannot send to cart — serial port not available. '
                f'ser={"exists" if self.ser else "None"}, '
                f'is_open={self.ser.is_open if self.ser else "N/A"}, '
                f'pyserial={"installed" if serial else "NOT INSTALLED"}',
                throttle_duration_sec=5.0)

    def destroy(self):
        if self.encoder:
            self.encoder.stop()
            if self.encoder.ser is not self.ser:
                self.encoder.ser.close()
        if self.ser and self.ser.is_open:
            self.ser.write(build_serial_command(b_throttle=50, d_steering=50, i_brake=1).encode())
            time.sleep(0.05)
            self.ser.close()
