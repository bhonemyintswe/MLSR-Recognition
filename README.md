# Gesture Recognition with Gradio

![Demo](demo.gif)

## Overview

This project demonstrates a real-time gesture recognition system using deep learning and the Gradio library. The system can classify hand gestures captured from a video stream and provide text labels for the recognized gestures. It utilizes a pre-trained deep learning model to make predictions based on keypoint features extracted from the video frames.

## Table of Contents

- [Dependencies](#dependencies)

## Dependencies

Before running the project, ensure you have the following dependencies installed:

- Gradio: 3.44.4
- TensorFlow: 2.13.0
- NumPy: 1.22.4
- MediaPipe: 0.10.5
- OpenCV: 4.8.1

You can install these dependencies using the `requirements.txt` file provided in this repository.


```bash
# Clone the repository
git clone https://github.com/bhonemyintswe/MSLR.git

# Navigate to the project directory
cd MSLR

# Install required packages
pip install -r requirements.txt
