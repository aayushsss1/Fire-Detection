# Fire Detection Android App with MobileNet and TensorFlow Lite

This repository contains the source code and resources for a Fire Detection Android application that utilizes the MobileNet model, optimized and compressed using TensorFlow Lite. The app's primary goal is to identify the presence of fire within video frames and provide real-time feedback to the user.

## Overview

Detecting fires quickly and accurately is crucial for preventing and mitigating potential disasters. This Android app leverages machine learning and computer vision techniques to identify fires within video frames, providing valuable information to users in real-time.

The core of this application is the MobileNet model, which has been optimized and compressed using TensorFlow Lite. This ensures efficient performance on mobile devices without sacrificing accuracy. The app captures video frames from the device's camera and passes them through the model for inference.

## Requisites

- Android Studio
- TensorFlow Lite

## Model

MobileNet, a family of efficient convolutional neural network models, is an excellent choice for object detection tasks like fire detection in images or video frames. Its lightweight architecture, depthwise separable convolutions, and TensorFlow Lite compatibility make it well-suited for resource-constrained mobile and embedded devices. By fine-tuning a MobileNet model on fire-related data, it can quickly and accurately identify fires, enabling real-time fire detection applications that play a crucial role in enhancing safety and preventing disasters.

## Usage

1. Clone the GitHub Repository and run the `FireDetection.ipynb` notebook to generate the tensorflow lite model.

```bash
git clone https://github.com/aayushsss1/Fire-Detection.git
```
2. Open the project in Android Studio.
3. Build and run the app on your Android device or emulator.

## Output 

The Final Android Application !

<div align = "center">
<kbd>
<img src="https://user-images.githubusercontent.com/51130346/180652664-92ccd58a-7411-4751-b52f-1d79c8119bd3.jpg" alt="alt text" width="200" height="400"/>
</kbd>
</div>