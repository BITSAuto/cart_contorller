# Cart Controller ROS 2 Package

> [!IMPORTANT]  
> When operating this package in simulation mode, the CARLA server must be explicitly executed with native ROS 2 compatibility enabled:
> ```bash
> ./CarlaUE4.sh --ros2
> ```

This package manages the autonomous steering of both the simulated CARLA vehicle and the physical hardware cart. The system is split strictly into two layers: a dumb hardware abstraction bridge (`controller.py`), and a high-level mathematical steering algorithm (`navigator.py`).

---

## 1. `controller.py` (Low-Level Hardware Bridge)
**Sole Purpose:** To safely wrap and enforce the strict physical timings and constraints of the real cart's steering actuator. It does no independent logic or decision-making. 

### Hardware Variables & Constraints (`controller.py`)
These variables define the unbreakable physical limits of the cart. They should rarely be changed as they map directly to hardware physics.

*   `PULSE_MS = 35.0`: The enforced minimum hardware pulse length. The real motor intrinsically takes 30ms to react to a signal. To safely clear this dead-time, any pulse requested by the navigator will be physically held active by the controller for exactly 35.0ms before being explicitly stopped.
*   `MAX_STEER_DEG = 15.0`: The maximum physical steering lock limits of the vehicle (left and right), used to clamp the simulation angle.
*   `DEG_PER_PULSE = 0.45`: A calibration constant representing how many literal degrees the physical wheels turn given one single 35ms burst. Only used to update the virtual `simulated_angle`.
*   `STEER_CMD_SPEED = 15.0`: The steering actuation velocity published to CARLA (`msg.steer`). It simulates the physical turning motion (0.45 deg per 35ms pulse) perfectly smoothly.
*   `MOTOR_RIGHT = 51`, `MOTOR_LEFT = 49`, `MOTOR_STOP = 50`: The raw ASCII code signals sent over the serial bus to the Arduino. Note the critical `MOTOR_STOP`: the real cart motor has infinite run-on and **does not stop by itself** until this `50` command is explicitly received.
*   `IMU_EMA_ALPHA = 0.3`: The Exponential Moving Average (EMA) smoothing factor used strictly on the raw IMU data stream to produce the `smoothed_yaw_rate`.
*   `ENCODER_PORT = '/dev/ttyUSB1'`, `ENCODER_BAUD = 115200`: Serial link of the ESP32 streaming the steering encoder count (one reading per line, e.g. `-1234` or `Encoder: -1234`). Set `ENCODER_PORT = SERIAL_PORT` if the counts arrive on the steering command link instead.
*   `ENCODER_MAX_RIGHT = -2500`, `ENCODER_MAX_LEFT = 2500`: Encoder counts at full right/left lock. Counts map linearly onto `±MAX_STEER_DEG` (one 0.45° pulse ≈ 75 counts).
*   `ENCODER_MARGIN = 150`: Pulses toward a lock are refused this many counts before it (~2 pulses), leaving room for the motor's run-on.
*   `ENCODER_TIMEOUT_S = 0.5`: If no encoder reading arrives within this window the encoder is treated as lost and **all steering pulses are blocked** (HUD shows `ENC: NO`).

### How `controller.py` Enforces Them
*   **The Discrete Pulse Modulator (`MotorSteering` class):** To fix the 30ms delay and infinite run-on, the controller restricts the navigator to simple `pulse_right()` and `pulse_left()` method calls. When called, the controller forces the motor to `51`/`49`. The moment 35.0ms elapses, the `tick()` function intercepts it and **forces a `50` STOP command**.
*   **The 30ms Serial Limiter:** In the `_send_hw()` function, a strict `0.030s` timeout is applied to serialize operations. This protects the physical bus from being spammed by rapid-fire PID logs.

---

## 2. `navigator.py` (High-Level Driving Logic)
**Sole Purpose:** To segment the road using YOLO computer vision and mathematically derive a safe, continuous steering effort utilizing a Cascade PID and Software PWM.

### Navigation Variables & Tuning (`navigator.py`)
These variables control *how* the cart drives and responds to curves. They are meant to be actively tuned over time.

**PID Parameters**
*   `K_P = 0.15` (Proportional Gain): Multiplies the raw optical error (`_composite_error`). The higher this is, the harder/faster the car tries to steer back to the center of the lane.
*   `K_I = 0.005` (Integral Gain): Scales the `_integral_error` accumulation. Highly useful for correcting any persistent asymmetrical drifting (e.g. if the physical cart naturally pulls left, the I-term builds up forcing constant micro-rights over time).
*   `K_D = 0.8` (Derivative Gain): Sourced entirely from the IMU (`actual_yaw_rate`). Provides heavy damping to stop the cart from oversteering. If the wheels are physically turning rapidly through a curve, this generates a strong counter-steer signal dynamically stopping the rotational momentum.

**Optical & Deadband Parameters**
*   `HEADING_DEADBAND = 1.5` (degrees): The micro-oscillation tolerance. If the car is perfectly aligned facing the road within +/- 1.5 degrees, the PID immediately zeroes itself and stops requesting pulses. This freezes the steering angle entirely and stops continuous left/right chattering jitter on long straights.
*   `PID_DEADBAND = 0.1`: The minimum duty cycle required to wake up the PWM. If the PID output magnitude is `< 0.1` (e.g., fractional noise), it is ignored (producing 0.0ms of ON-time).
*   `PWM_PERIOD_S = 0.150` (150ms): The length of the continuous evaluation block for the Software PWM. A duty cycle of 33% requests a single 50ms ON-time burst within this 150ms window.
*   `K_OFFSET = 0.05`: Maps the raw pixel offset from the center of the lane (`road_cx - img_cx`) down into a mock physical angular degree error so it can be linearly blended into the `_composite_error`.
*   `EMA_ALPHA = 0.15`: The aggressive smoothing factor applied exclusively on the computer vision YOLO segmentation output. Helps calm flickering bounding boxes or noisy road predictions at higher speeds before they reach the PID.

### The Driving Algorithm (Software PWM)
The Continuous Cascade PID naturally guarantees a bounded output between `[-1.0, 1.0]`. To apply this perfectly smooth mathematical signal to the extremely discrete hardware bridge (`controller.py`), the navigator implements a **Software PWM Cycle**:

1.  **Duty Cycle calculation:** If `_pid_output` = 0.50, the PID requests 50% duty cycle, generating 75ms of active turning time per 150ms block.
2.  **Hardware Dispatch:** During that 75ms "ON-time", `navigator.py` continuously spams the `pulse_right()` command to the hardware bridge.
3.  **Hardware execution:** Because the bridge inherently wraps every single `pulse_right()` command in an isolated, locked 35ms minimum execution block, spamming it for 75ms ends up generating exactly two continuous 35ms blocks back to back (70ms total). This perfectly emulates a human smoothly turning the wheel and then manually holding it.
4.  **Hold cycle:** For the remaining 75ms "OFF-time", the PWM explicitly commands the bridge to IDLE, executing the `50` Stop command.

to run everything first run 
carla sim
ros2 run cart_controller navigator.py
