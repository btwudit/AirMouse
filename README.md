# AirMouse

AirMouse is a computer vision project. It lets you move your computers cursor with your hand, in front of a webcam. The project uses Python, OpenCV and hand tracking, so you can try out mouse controls without holding a mouse.

I made this project to play around with making daily computer use feel a bit more natural.

## What you need

- A Windows PC or laptop

- A webcam (a USB webcam you plug in works too)

- Python **3.9 to 3.14**

- A working internet connection the time you install the packages

- Install all the packages listed in the requirment.txt

## Install and set it up

### 1. Download the project

Clone this repository. Download it as a ZIP file. Then open the project folder in VS Code or your terminal.

### 2. Create an environment

In the project folder, run:

```powershell

py -3.12 -m venv venv

```

You can swap `3.12` for another Python version you have installed, from 3.9 to 3.14.

### 3. Activate it

For Windows PowerShell:

```powershell

attached in the main.py as commment in the line

```

When it is active, you should see `(venv)` at the start of your terminal line.

### 4. Install the required packages

Run this inside the activated environment:

```bash

python -m pip install --upgrade pip

python -m pip install opencv-python mediapipe pyautogui numpy

pip install requirment.txt

```

These packages handle webcam input, hand tracking, cursor control and image processing.

### 5. Run AirMouse

Run the Python file. For example, if your main file is called `main.py`:

```bash

python main.py

```

If your main script has a name, use that filename instead.

## A setup tips

- Plug in your webcam before you start the program.

- This project is built around a webcam that sits upright, so keep the camera in the position you used when setting up the project.

- Make sure your hand is easy to see and the room has decent lighting.

- Close the camera. Any other program that might already be using the webcam.

## Built with

- Python

-

- MediaPipe

- PyAutoGUI

- NumPy
