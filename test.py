#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Script: Voice-to-MyCobot Command Construction (No Robot Movement)
This script records audio from the microphone, transcribes it using Whisper,
and then uses GPT‑4 via the updated OpenAI API to convert the voice command
into a MyCobot320Socket API command. Finally, it prints the generated command.
"""

import time
import numpy as np
import sounddevice as sd
import whisper
import openai

# -------------------------------
# Configuration Parameters
# -------------------------------
SAMPLE_RATE = 16000         # Sampling rate in Hz (Whisper expects 16 kHz)
DURATION = 3                # Recording duration in seconds
MODEL_SIZE = "base"         # Whisper model size: "base", "small", "medium", or "large"

# Set your OpenAI API key and project ID.
OPENAI_API_KEY = ""
OPENAI_PROJECT = ""

# Set these as global properties for the OpenAI package.
openai.api_key = OPENAI_API_KEY
openai.organization = None  # Set this if you have an organization ID; otherwise, leave as None.
openai.project = OPENAI_PROJECT

COMMAND_PROMPT_TEMPLATE = (
    "You will receive a command converted from my voice. "
    "You are allowed to generate only one command at a time. "
    "If you received a command telling you move to a certain direction,"
    "or just containing a direction information,"
    "you need to generate a coordination like:"
    "[-329.1, 104.6, 179.1]"
    "this three number refer to x,y,z direction,"
    "and [-329.1, 104.6, 179.1] is the starting point,"
    "for example, if you received a command like  'go left 40'"
    "you have to add 40 to y, then send the coordination only"
    "which is [-329.1, 144.6, 179.1]"
    "left and right direction are on y, left is Positive direction, right is Negative direction"
    "forward and back direction are on x, back is Positive direction, forward is Negative direction"
    "up and down direction are on z, up is Positive direction, down is Negative direction"
    "Besides this, when received a command telling you to control the graper,"
    "you send a number like '1' only"
    "when receiving a command similar to opening the gripper,"
    "you should send '0' "
    "when receiving a command similar to closing the gripper,"
    "you should send '1' "
    "and the command is: {}. "
)
# -------------------------------

def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    """Record audio from the microphone and return a 1D NumPy array."""
    try:
        print(f"Recording audio for {duration} seconds...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()
    except Exception as e:
        print("Error recording audio:", e)
        return None

def transcribe_audio(audio, fs=SAMPLE_RATE):
    """Transcribe the given audio using Whisper."""
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
    Use GPT-4 to convert the transcribed voice command into a MyCobot API command.
    This function uses the updated OpenAI API syntax.
    """
    try:
        prompt = COMMAND_PROMPT_TEMPLATE.format(command_text)
        print("Sending prompt to GPT-4:")
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
        print("GPT-4 generated command:", generated_command)
        return generated_command
    except Exception as e:
        print("Error during GPT-4 interpretation:", e)
        return None

def main():
    # Step 1: Record audio from the microphone
    audio = record_audio()
    if audio is None:
        print("Audio recording failed. Exiting.")
        return

    # Step 2: Transcribe audio using Whisper
    text_command = transcribe_audio(audio)
    if len(text_command) < 3:
        print("Transcribed command is too short or unclear. Please try again.")
        return

    # Step 3: Interpret the command using GPT-4
    robot_command = interpret_command(text_command)
    if not robot_command:
        print("Failed to generate a robot command. Exiting.")
        return

    # Step 4: Print out the final result for testing purposes
    print("\nFinal constructed MyCobot command:")
    print(robot_command)

if __name__ == "__main__":
    main()