#!/usr/bin/env python3

import serial
import time

# --- Configuration ---
# Replace with your ESP32's actual serial port
SERIAL_PORT = '/dev/ttyUSB0' 
# 115200 is the standard default for ESP32, change if your sketch uses a different rate (e.g., 9600)
BAUD_RATE = 115200           

def main():
    try:
        # Open the serial port
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Successfully connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
        print("Listening for data... (Press Ctrl+C to stop)")

        # Continuously read from the serial port
        while True:
            if ser.in_waiting > 0:
                # Read the line, decode the bytes to a string, and strip trailing newlines
                incoming_data = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if incoming_data:
                    print(f"ESP32: {incoming_data}")
                    
            # Small sleep to prevent the loop from maxing out CPU usage
            time.sleep(0.01)

    except serial.SerialException as e:
        print(f"\n[Error] Could not open serial port {SERIAL_PORT}.")
        print(f"Details: {e}")
        print("Make sure the port is correct and not open in the Arduino IDE Serial Monitor.")
    except KeyboardInterrupt:
        print("\n[Info] Script interrupted by user. Exiting...")
    finally:
        # Ensure the serial port is properly closed on exit
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("[Info] Serial port closed.")

if __name__ == '__main__':
    main()