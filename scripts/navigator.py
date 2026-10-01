#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import math
import time
import os
import argparse

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from controller import CartController, TOPICS

def _fix_qt_font_dir():
    # opencv-python points Qt at its bundled cv2/qt/fonts; if that folder is missing or empty
    # Qt floods the terminal with "QFontDatabase: Cannot find font directory" warnings.
    font_dir = os.environ.get('QT_QPA_FONTDIR')
    if not font_dir or (os.path.isdir(font_dir) and os.listdir(font_dir)): return
    for sys_dir in ('/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/TTF'):
        if os.path.isdir(sys_dir):
            os.environ['QT_QPA_FONTDIR'] = sys_dir
            return

_fix_qt_font_dir()

K_OFFSET           = 0.05
LOOKAHEAD_ROWS     = 10
EMA_ALPHA          = 0.15

K_P                = 0.13
K_I                = 0.005
K_D                = 0.6
PID_DEADBAND       = 0.1

HUD_H              = 50
COMPASS_R          = 55
COMPASS_MARGIN     = 10

PWM_PERIOD_S       = 0.150

def estimate_road_heading(mask, scan_rows=10):
    h, w = mask.shape[:2]
    horizon_y = int(h * 0.35)
    bottom_y = int(h * 0.85)
    step = max(1, (bottom_y - horizon_y) // scan_rows)
    
    for row_y in range(horizon_y, bottom_y, step):
        road_px = np.where(mask[row_y, :] > 127)[0]
        if len(road_px) >= 10:
            distant_cx = np.mean(road_px)
            return (distant_cx - (w / 2.0)) * 0.09
            
    return 0.0

def find_road_center(mask, scan_rows=5):
    h, w = mask.shape[:2]
    img_cx = w / 2.0
    row_positions = np.linspace(int(h * 0.80), int(h * 0.55), scan_rows, dtype=int)
    centers = []
    for row_y in row_positions:
        if row_y < 0 or row_y >= h: continue
        road_px = np.where(mask[row_y, :] > 127)[0]
        if len(road_px) < 10: continue
        mid = np.mean(road_px)
        centers.append(mid)
        
    if not centers: return None, None, None
    road_cx = float(np.mean(centers))
    return road_cx - img_cx, road_cx, img_cx

class CartNavigator(Node):
    def __init__(self, mode='carla'):
        super().__init__('cart_navigator')
        self.bridge = CvBridge()
        self.mode = mode
        model_path = os.path.expanduser('~/ros2_ws/src/road_segmentation/scripts/yolo11m-road-seg.pt')
        if YOLO and os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.get_logger().info('[Nav] YOLO loaded.')
        else:
            self.model = None

        self.ctrl = CartController(self, mode=mode)

        self._road_heading_deg = 0.0
        self._ema_offset       = 0.0
        self._ema_heading      = 0.0
        self._initialized_ema  = False

        self._composite_error  = 0.0
        self._integral_error   = 0.0
        self._pid_output       = 0.0
        self._prev_time        = None
        
        self._pwm_cycle_start  = 0.0

        rgb_topic = TOPICS[mode]['rgb']
        self.create_subscription(Image, rgb_topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info('[Nav] Started Software PWM Cascade PID Navigator.')
        self.get_logger().info(f'[Nav] Waiting for image to publish on topic: {rgb_topic} ...')
        
        self._received_first_image = False

    def _on_image(self, msg):
        if not self._received_first_image:
            self.get_logger().info(f'[Nav] Successfully received first image from {TOPICS[self.mode]["rgb"]}!')
            self._received_first_image = True

        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            mask = self._segment(img)

            if mask is None:
                self.ctrl.hold()
                return

            road_heading_raw = estimate_road_heading(mask, LOOKAHEAD_ROWS)
            offset, road_cx, img_cx = find_road_center(mask)

            if offset is not None and road_heading_raw is not None:
                if not self._initialized_ema:
                    self._ema_offset = offset
                    self._ema_heading = road_heading_raw
                    self._initialized_ema = True
                else:
                    self._ema_offset = EMA_ALPHA * offset + (1 - EMA_ALPHA) * self._ema_offset
                    self._ema_heading = EMA_ALPHA * road_heading_raw + (1 - EMA_ALPHA) * self._ema_heading

                self._road_heading_deg = self._ema_heading
                self._composite_error = self._road_heading_deg + (K_OFFSET * self._ema_offset)

            now = time.time()
            if self._pwm_cycle_start == 0.0:
                self._pwm_cycle_start = now
                
            dt = now - self._prev_time if self._prev_time else 0.1
            if dt <= 0 or dt > 1.0: dt = 0.1
            self._prev_time = now

            if abs(self._composite_error) < 1.5:
                self._pid_output = 0.0
                self._integral_error *= 0.95
            else:
                # Anti-windup: don't integrate toward a steering lock the encoder guard is blocking
                can_steer = self.ctrl.can_pulse_right() if self._composite_error > 0 else self.ctrl.can_pulse_left()
                if can_steer:
                    self._integral_error += self._composite_error * dt
                    self._integral_error = max(-20.0, min(20.0, self._integral_error))

                eff_yaw = self.ctrl.smoothed_yaw_rate if abs(self.ctrl.smoothed_yaw_rate) > 0.5 else 0.0

                p_term = K_P * self._composite_error
                i_term = K_I * self._integral_error
                d_term = -K_D * eff_yaw

                self._pid_output = p_term + i_term + d_term

            time_in_cycle = now - self._pwm_cycle_start
            if time_in_cycle >= PWM_PERIOD_S:
                self._pwm_cycle_start = now
                time_in_cycle = 0.0
                
            duty_cycle = min(1.0, max(0.0, abs(self._pid_output)))
            
            if duty_cycle < PID_DEADBAND:
                on_time = 0.0
            else:
                on_time = max(0.035, duty_cycle * PWM_PERIOD_S)

            if time_in_cycle < on_time:
                if self.ctrl.motor.is_idle:
                    # Encoder-guarded: refused near ±2500 lock or when the encoder is stale
                    if self._pid_output > 0:
                        self.ctrl.pulse_right()
                    else:
                        self.ctrl.pulse_left()
            else:
                pass

            self.ctrl.tick()

            # --- Debug: periodic navigation state ---
            self.get_logger().info(
                f'[Nav] error={self._composite_error:+.1f}° pid={self._pid_output:+.2f} '
                f'duty={duty_cycle:.2f} on_time={on_time:.3f}s | '
                f'motor={self.ctrl.motor.motor} enc_ok={self.ctrl.encoder_ok} '
                f'imu_ok={self.ctrl.imu_ok} heading={"%.1f" % self.ctrl.car_heading_deg if self.ctrl.car_heading_deg is not None else "N/A"}',
                throttle_duration_sec=3.0)

            self._draw(img, mask, offset, road_cx, img_cx)

        except Exception as e:
            self.get_logger().error(f'Image error: {e}')

    def _segment(self, img):
        if self.model is None: return None
        results = self.model.predict(img, conf=0.85, verbose=False)[0]
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if results.masks is not None:
            for m in results.masks.data:
                m_np = m.cpu().numpy().astype(np.uint8)
                m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = cv2.bitwise_or(mask, m_np * 255)
            return mask
        return None

    def _draw(self, img, mask, offset, road_cx, img_cx):
        vis = img.copy()
        h, w = vis.shape[:2]

        if mask is not None:
            overlay = np.zeros_like(vis)
            overlay[mask > 127] = (0, 180, 60)
            vis = cv2.addWeighted(vis, 1.0, overlay, 0.35, 0)

        icx = int(w / 2)
        top = int(h * 0.4)
        bot = int(h * 0.88)
        cv2.line(vis, (icx, top), (icx, bot), (255, 255, 255), 1)
        if road_cx is not None:
            cv2.line(vis, (int(road_cx), top), (int(road_cx), bot), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = HUD_H + 30

        cv2.putText(vis, f'Offset: {offset:+.0f} px' if offset is not None else 'No road',
                    (10, y), font, 0.50, (255, 255, 255), 1); y += 22
        cv2.putText(vis, f'Error: {self._composite_error:+.1f} deg',
                    (10, y), font, 0.50, (0, 255, 255), 1); y += 22
        cv2.putText(vis, f'PID Duty: {self._pid_output:+.2f}',
                    (10, y), font, 0.50, (200, 200, 255), 1); y += 22
        cv2.putText(vis, f'IMU Steer: {self.ctrl.imu_steer_deg:+.1f} deg',
                    (10, y), font, 0.50, (200, 200, 255), 1); y += 22
        enc = self.ctrl.encoder_count
        if enc is None:
            enc_str, enc_col = 'Enc Steer: ---', (100, 100, 100)
        else:
            lock = '' if self.ctrl.can_pulse_right() and self.ctrl.can_pulse_left() else '  LIMIT'
            enc_str, enc_col = f'Enc Steer: {self.ctrl.encoder_steer_deg:+.1f} deg ({enc:+d}){lock}', (200, 255, 200)
        cv2.putText(vis, enc_str, (10, y), font, 0.50, enc_col, 1); y += 22
        cv2.putText(vis, f'Yaw Rate: {self.ctrl.smoothed_yaw_rate:+.1f} d/s',
                    (10, y), font, 0.50, (0, 255, 0), 1); y += 22

        cv2.putText(vis, f'Mode: {self.mode.upper()}',
                    (10, y), font, 0.50, (0, 200, 255) if self.mode == 'carla' else (255, 200, 0), 1)

        self._draw_heading_strip(vis, w)
        self._draw_compass(vis, w)

        cv2.imshow('Software PWM Cascade PID', vis)
        cv2.waitKey(1)

    def _draw_heading_strip(self, vis, w):
        font = cv2.FONT_HERSHEY_SIMPLEX
        strip = vis[0:HUD_H, 0:w].copy()
        cv2.rectangle(strip, (0, 0), (w, HUD_H), (20, 20, 20), -1)
        vis[0:HUD_H, 0:w] = cv2.addWeighted(vis[0:HUD_H, 0:w], 0.3, strip, 0.7, 0)
        for x_sep in [w // 3, 2 * w // 3]: cv2.line(vis, (x_sep, 5), (x_sep, HUD_H - 5), (80, 80, 80), 1)
        if self.ctrl.car_heading_deg is not None:
            if not hasattr(self, '_initial_heading'): self._initial_heading = self.ctrl.car_heading_deg
            rel = -((self.ctrl.car_heading_deg - self._initial_heading + 180) % 360 - 180)
            car_str, car_color = f'Car: {rel:+.1f} deg', (255, 255, 255)
        else:
            car_str, car_color = 'Car: ---', (100, 100, 100)
        cv2.putText(vis, car_str, (10, 32), font, 0.58, car_color, 1, cv2.LINE_AA)
        cv2.putText(vis, f'Road: {self._road_heading_deg:+.1f} deg', (w // 3 + 15, 32), font, 0.58, (0, 255, 120), 1, cv2.LINE_AA)
        imu_str = 'IMU: OK' if self.ctrl.imu_ok else 'IMU: NO'
        imu_col = (0, 220, 0) if self.ctrl.imu_ok else (0, 0, 220)
        cv2.putText(vis, imu_str, (2 * w // 3 + 15, 32), font, 0.58, imu_col, 1, cv2.LINE_AA)
        enc_str = 'ENC: OK' if self.ctrl.encoder_ok else 'ENC: NO'
        enc_col = (0, 220, 0) if self.ctrl.encoder_ok else (0, 0, 220)
        cv2.putText(vis, enc_str, (2 * w // 3 + 135, 32), font, 0.58, enc_col, 1, cv2.LINE_AA)

    def _draw_compass(self, vis, w):
        cx, cy = w - COMPASS_R - COMPASS_MARGIN - 5, HUD_H + COMPASS_R + COMPASS_MARGIN
        cv2.circle(vis, (cx, cy), COMPASS_R + 2, (30, 30, 30), -1)
        cv2.circle(vis, (cx, cy), COMPASS_R, (60, 60, 60), 1)
        def draw_arrow(angle_deg, color, thickness=2, tip_len=0.35):
            rad = math.radians(angle_deg - 90)
            tip_y = int(cy + (COMPASS_R - 8) * math.sin(rad))
            cv2.arrowedLine(vis, (int(cx - (COMPASS_R // 3) * math.cos(rad)), int(cy - (COMPASS_R // 3) * math.sin(rad))), (int(cx + (COMPASS_R - 8) * math.cos(rad)), tip_y), color, thickness, tipLength=tip_len, line_type=cv2.LINE_AA)
        if self._road_heading_deg is not None: draw_arrow(self._road_heading_deg, (0, 220, 80), thickness=2)
        draw_arrow(0.0, (200, 200, 200), thickness=3)
        cv2.circle(vis, (cx, cy), 4, (255, 255, 0), -1)
        leg_y = cy + COMPASS_R + 15
        cv2.putText(vis, 'Car', (cx - COMPASS_R, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(vis, 'Road', (cx + 5, leg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 80), 1)

    def destroy_node(self):
        self.ctrl.destroy()
        super().destroy_node()

def main():
    parser = argparse.ArgumentParser(description='Cart Software PWM Cascade PID Navigator')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--carla', action='store_true', help='CARLA mode')
    group.add_argument('--cart', action='store_true', help='Cart mode')
    args, remaining = parser.parse_known_args()
    rclpy.init(args=remaining)
    node = CartNavigator(mode='carla' if args.carla else 'cart')
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
