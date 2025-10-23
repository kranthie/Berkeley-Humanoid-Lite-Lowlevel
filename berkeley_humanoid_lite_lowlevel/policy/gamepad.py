# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.

"""
Gamepad Controller Module for Berkeley Humanoid Lite

This module implements UDP-based controllers for the Berkeley Humanoid Lite robot,
supporting both gamepad and keyboard input devices. It handles command broadcasting
over UDP for robot control modes and movement velocities.
"""

import threading
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from inputs import get_gamepad, devices


class XInputEntry:
    """
    Constants for gamepad button and axis mappings.

    This class defines the standard mapping for various gamepad controls,
    including analog sticks, triggers, d-pad, and buttons.
    """
    AXIS_X_L = "ABS_X"
    AXIS_Y_L = "ABS_Y"
    AXIS_TRIGGER_L = "ABS_Z"
    AXIS_X_R = "ABS_RX"
    AXIS_Y_R = "ABS_RY"
    AXIS_TRIGGER_R = "ABS_RZ"

    BTN_HAT_X = "ABS_HAT0X"
    BTN_HAT_Y = "ABS_HAT0Y"

    BTN_A = "BTN_SOUTH"
    BTN_B = "BTN_EAST"
    BTN_X = "BTN_NORTH"
    BTN_Y = "BTN_WEST"
    BTN_BUMPER_L = "BTN_TL"
    BTN_BUMPER_R = "BTN_TR"
    BTN_THUMB_L = "BTN_THUMBL"
    BTN_THUMB_R = "BTN_THUMBR"
    BTN_BACK = "BTN_SELECT"
    BTN_START = "BTN_START"


@dataclass
class ControllerProfile:
    """Profile defining how to normalize axis values for different controller types."""
    name: str
    center_value: float  # Center position value (0 for XInput, 128 for DualSense)
    max_range: float     # Maximum deviation from center (32768 for XInput, 128 for DualSense)
    invert: bool = True  # Whether to invert the Y axis (standard for most controllers)

    def normalize_axis(self, raw_value: Optional[int]) -> float:
        """Normalize raw axis value to range [-1.0, 1.0]."""
        if raw_value is None:
            return 0.0

        normalized = (raw_value - self.center_value) / self.max_range
        if self.invert:
            normalized = -normalized
        return normalized


# Controller profiles for different gamepad types
CONTROLLER_PROFILES = {
    'dualsense': ControllerProfile(
        name='Sony DualSense (PS5)',
        center_value=128.0,
        max_range=128.0,
        invert=True
    ),
    'xinput': ControllerProfile(
        name='XInput/Xbox Controller',
        center_value=0.0,
        max_range=32768.0,
        invert=True
    ),
}


class Se2Gamepad:
    def __init__(self,
                 stick_sensitivity: float = 1.0,
                 dead_zone: float = 0.01,
                 debug: bool = False,
                 ) -> None:
        self.stick_sensitivity = stick_sensitivity
        self.dead_zone = dead_zone
        self.debug = debug

        self._stopped = threading.Event()
        self._run_forever_thread = None

        self.reset()

        self.commands = {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "velocity_yaw": 0.0,
            "mode_switch": 0,
        }

        # Detect and configure controller
        self._gamepad_device, self._controller_profile = self._detect_controller()
        print(f"Using controller: {self._controller_profile.name}")

    def _detect_controller(self) -> Tuple[Optional[object], ControllerProfile]:
        """Detect connected controller and return device and appropriate profile."""
        # Try to find a known controller type
        for device in devices:
            device_name = device.name.lower()

            # Check for DualSense/PS5 controller
            if ('dualsense' in device_name or 'sony' in device_name):
                # Exclude motion sensors and touchpad
                if 'motion' not in device_name and 'touchpad' not in device_name:
                    print(f"Detected gamepad: {device.name}")
                    return device, CONTROLLER_PROFILES['dualsense']

            # Check for Xbox/XInput controllers
            if 'xbox' in device_name or 'xinput' in device_name:
                print(f"Detected gamepad: {device.name}")
                return device, CONTROLLER_PROFILES['xinput']

        # No specific device found, use default with XInput profile
        print("No specific controller detected, using default gamepad with XInput profile")
        return None, CONTROLLER_PROFILES['xinput']

    def reset(self) -> None:
        self._states = {key: 0 for key in XInputEntry.__dict__.values()}

    def stop(self) -> None:
        print("Gamepad stopping...")
        self._stopped.set()
        # self._run_forever_thread.join()

    def run(self) -> None:
        self._run_forever_thread = threading.Thread(target=self.run_forever)
        self._run_forever_thread.start()

    def run_forever(self) -> None:
        while not self._stopped.is_set():
            self.advance()

    def advance(self) -> None:
        try:
            # Use specific device if found, otherwise use default
            if self._gamepad_device:
                events = self._gamepad_device.read()
            else:
                events = get_gamepad()

            # Update all events from the joystick
            for event in events:
                self._states[event.code] = event.state

            self._update_command_buffer()
        except Exception as e:
            # Silently ignore gamepad errors (device not found, disconnected, etc.)
            if self.debug:
                print(f"Gamepad error: {e}")
            pass

    def _normalize_and_apply_deadzone(self, raw_value: Optional[int]) -> float:
        """Normalize axis value using controller profile and apply dead zone."""
        normalized = self._controller_profile.normalize_axis(raw_value)

        # Apply dead zone
        if abs(normalized) < self.dead_zone:
            normalized = 0.0

        return normalized * self.stick_sensitivity

    def _update_command_buffer(self) -> Dict[str, float]:
        """Update command buffer with current gamepad state."""
        # Read raw axis values
        raw_velocity_x = self._states.get(XInputEntry.AXIS_Y_L)
        raw_velocity_y = self._states.get(XInputEntry.AXIS_X_R)
        raw_velocity_yaw = self._states.get(XInputEntry.AXIS_X_L)

        # Normalize and apply dead zones using controller profile
        self.commands["velocity_x"] = self._normalize_and_apply_deadzone(raw_velocity_x)
        self.commands["velocity_y"] = self._normalize_and_apply_deadzone(raw_velocity_y)
        self.commands["velocity_yaw"] = self._normalize_and_apply_deadzone(raw_velocity_yaw)

        # Determine mode switch based on button combinations
        mode_switch = 0

        # Enter RL control mode (A + Right Bumper)
        if self._states.get(XInputEntry.BTN_A) and self._states.get(XInputEntry.BTN_BUMPER_R):
            mode_switch = 3

        # Enter init mode (A + Left Bumper)
        if self._states.get(XInputEntry.BTN_A) and self._states.get(XInputEntry.BTN_BUMPER_L):
            mode_switch = 2

        # Enter idle mode (X or Left/Right Thumbstick press)
        if self._states.get(XInputEntry.BTN_X) or self._states.get(XInputEntry.BTN_THUMB_L) or self._states.get(XInputEntry.BTN_THUMB_R):
            mode_switch = 1

        self.commands["mode_switch"] = mode_switch


if __name__ == "__main__":
    command_controller = Se2Gamepad()
    command_controller.run()

    try:
        while True:
            print(f"""{command_controller.commands.get("velocity_x"):.2f}, {command_controller.commands.get("velocity_y"):.2f}, {command_controller.commands.get("velocity_yaw"):.2f}""")
            pass
    except KeyboardInterrupt:
        print("Keyboard interrupt")

    command_controller.stop()
