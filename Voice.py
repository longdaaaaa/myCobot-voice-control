#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Voice-Controlled MyCobot 320 Robot Control Script (Wi-Fi, GPT-4o-mini, Coordinate Change Mode)
-----------------------------------------------------------------------------------------------
Workflow:
  1. Import required libraries.
  2. Connect to the robot via Wi-Fi using MyCobot320Socket.
  3. Set the gripper to pass-through mode.
  4. Move the robot to the zero position, then to the fixed starting coordinates.
  5. Wait for user input. When Enter is pressed, record 3 seconds of audio and transcribe it using Whisper.
  6. Send the transcribed command to GPT-4o-mini with a prompt that instructs it to return either:
       - A coordinate change (a list of three numbers, e.g. [0, 40, 0]), which is added to the current robot coordinates.
       - A single number ("0" or "1") for gripper control (0 = open, 1 = close).
  7. Execute the command accordingly.
  8. When the user types "q" to quit, return the robot to the zero position.
"""

import time
import numpy as np
import sounddevice as sd
import whisper
import openai
from pymycobot import MyCobot320Socket

# -------------------------------
# Configuration Parameters
# -------------------------------
SAMPLE_RATE = 16000         # 16 kHz (Whisper expects 16kHz audio)
DURATION = 3                # Recording duration in seconds for each command
MODEL_SIZE = "base"         # Whisper model size: "base", "small", "medium", or "large"

# OpenAI API settings
OPENAI_API_KEY = ""
OPENAI_PROJECT = ""
openai.api_key = OPENAI_API_KEY
openai.organization = None
openai.project = OPENAI_PROJECT

# Robot connection settings (Wi-Fi)
ROBOT_IP = "192.168.43.94"   # Replace with your robot's IP address
ROBOT_PORT = 9000           # Default port for MyCobot320Socket

# Initial positioning:
ZERO_ANGLES = [0, 0, 0, 0, 0, 0]  # Zero position (all joints zero)
START_COORDS = [-329.1, 104.6, 179.1, -179.46, -6.69, 95.57]  # Fixed starting coordinates

# Movement speed settings
MOVE_SPEED = 30            # Speed for coordinate movement
GRIPPER_SPEED = 100        # Speed for gripper commands

# Original fixed orientation for coordinate changes
ORIGINAL_ORIENTATION = [-179.46, -6.69, 95.57]

# Updated prompt template
COMMAND_PROMPT_TEMPLATE = (
    "You will receive a command converted from my voice. "
    "You are allowed to generate only one command at a time. "
    "If you received a command telling you move to a certain direction, "
    "or just containing a direction information, "
    "you need to generate a coordination change like: "
    "[100, 0, 0] "
    "this three number refer to x,y,z direction change, "
    "For example, if you received a command like 'go left 40', "
    "you have to generate a coordination change of [0, 40, 0], then send the coordination change only. "
    "Left and right directions are on y (left is positive, right is negative); "
    "forward and back directions are on x (forward is positive, back is negative); "
    "up and down directions are on z (up is positive, down is negative). "
    "Besides this, when receiving a command telling you to control the gripper, "
    "you send a number like '1' only; "
    "when receiving a command similar to opening the gripper, "
    "you should send '0'; "
    "when receiving a command similar to closing the gripper, "
    "you should send '1'. "
    "And the command is: {}. "
)

# -------------------------------

def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    """Record audio from the microphone and return a 1D NumPy array."""
    try:
        print(f"Recording audio for {duration} seconds...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print("Error recording audio:", e)
        return None

def transcribe_audio(audio, fs=SAMPLE_RATE):
    """Transcribe the given audio using OpenAI Whisper."""
    try:
        print("Transcribing audio...")
        model = whisper.load_model(MODEL_SIZE)
        result = model.transcribe(audio, fp16=False, language="en")
        text = result.get("text", "").strip()
        print("Transcribed text:", text)
        return text
    except Exception as e:
        print("Error during transcription:", e)
        return ""

def interpret_command(command_text):
    """
    Use GPT-4o-mini to convert the transcribed voice command into a command change.
    This function uses the updated OpenAI API syntax.
    """
    try:
        prompt = COMMAND_PROMPT_TEMPLATE.format(command_text)
        print("Sending prompt to GPT-4o-mini:")
        print(prompt)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an command generator, only generate command as I defined."},
                {"role": "user", "content": prompt}
            ],
            timeout=20
        )
        generated_command = response["choices"][0]["message"]["content"].strip()
        print("GPT-4o-mini generated command:", generated_command)
        return generated_command
    except Exception as e:
        print("Error during GPT-4o-mini interpretation:", e)
        return None

def parse_and_apply_command(cmd_str, mc):
    """
    Parse the generated command string and apply it to the robot.
    - If the command is a coordinate change (a list of 3 numbers), then:
         1. Retrieve the current coordinates from the robot.
         2. Update x, y, z by adding the coordinate change to the current x, y, z.
         3. Use ORIGINAL_ORIENTATION for the new target (do not update the orientation).
         4. Send the new coordinates using send_coords().
    - If the command is a single number ("0" or "1"), then control the gripper.
    """
    try:
        if cmd_str.startswith("[") and cmd_str.endswith("]"):
            # Clean up the string to convert any en dash/em dash to a standard hyphen.
            coord_str = cmd_str.strip("[]").replace("–", "-").replace("—", "-")
            parts = [float(x.strip()) for x in coord_str.split(",")]
            if len(parts) != 3:
                print("Invalid coordinate change length:", parts)
                return
            # Get current coordinates (only x, y, z) from the robot
            current_coords = mc.get_coords()  # returns [x, y, z, rx, ry, rz]
            print("Current robot coordinates:", current_coords)
            new_coords = [
                current_coords[0] + parts[0],
                current_coords[1] + parts[1],
                current_coords[2] + parts[2]
            ] + ORIGINAL_ORIENTATION
            print("New target coordinates:", new_coords)
            mc.send_coords(new_coords, MOVE_SPEED, 1)
        else:
            # Assume it's a gripper command (0 or 1)
            cmd_val = int(cmd_str)
            if cmd_val == 0:
                print("Opening gripper...")
                mc.set_gripper_state(0, GRIPPER_SPEED)
            elif cmd_val == 1:
                print("Closing gripper...")
                mc.set_gripper_state(1, GRIPPER_SPEED)
            else:
                print("Unrecognized gripper command:", cmd_val)
    except Exception as e:
        print("Error parsing or applying command:", e)

def main():
    # Step 1: Connect to the robot via Wi-Fi using MyCobot320Socket (pymycobot)
    try:
        print("Connecting to MyCobot 320 via Wi-Fi...")
        mc = MyCobot320Socket(ROBOT_IP, ROBOT_PORT)
        time.sleep(1)
        mc.focus_all_servos()
        time.sleep(1)
    except Exception as e:
        print("Error initializing robot:", e)
        return

    # Step 2: Set the gripper to pass-through mode
    try:
        print("Setting gripper to pass-through mode...")
        ret_mode = mc.set_gripper_mode(0)
        print("Gripper mode set, return code:", ret_mode)
    except Exception as e:
        print("Error setting gripper mode:", e)
        return
    time.sleep(1)

    # Step 3: Move robot to zero position, then to the fixed starting coordinates
    try:
        print("Moving robot to zero position...")
        mc.send_angles(ZERO_ANGLES, MOVE_SPEED)
        time.sleep(3)
        print("Moving robot to starting coordinates:", START_COORDS)
        mc.send_coords(START_COORDS, MOVE_SPEED, 1)
        time.sleep(3)
    except Exception as e:
        print("Error during initial positioning:", e)
        return

    # Main loop: process voice commands until user quits.
    while True:
        user_input = input("\nPress Enter to record a voice command, or type 'q' to quit: ").strip()
        if user_input.lower() == "q":
            print("Exiting voice control loop.")
            break

        # Step 4: Record audio and transcribe it.
        audio = record_audio(duration=DURATION, fs=SAMPLE_RATE)
        if audio is None:
            print("Audio recording failed. Try again.")
            continue
        text_command = transcribe_audio(audio, fs=SAMPLE_RATE)
        if len(text_command) < 3:
            print("Transcribed command is too short or unclear. Please try again.")
            continue

        # Step 5: Interpret the command using GPT-4o-mini
        generated_cmd = interpret_command(text_command)
        if not generated_cmd:
            print("Failed to generate a robot command. Try again.")
            continue

        # Step 6: Parse and apply the generated command.
        parse_and_apply_command(generated_cmd, mc)
        time.sleep(2)

    # After exiting the loop, return the robot to the zero position.
    try:
        print("Returning robot to zero position...")
        mc.send_angles(ZERO_ANGLES, MOVE_SPEED)
        time.sleep(3)
    except Exception as e:
        print("Error returning to zero position:", e)

if __name__ == "__main__":
    main()
