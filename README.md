# AirMouse

AirMouse is a small computer-vision project that lets you control your computer's cursor using your hand in front of a webcam. It uses Python, OpenCV and hand tracking, so you can try mouse controls without holding a physical mouse.
I built this project to experiment with making everyday computer interaction a little more natural.
## What you need
- A Windows PC or laptop
- A webcam (an external USB webcam works too)
- Python **3.9 to 3.14**
- A working internet connection for the first-time package installation
- install all the packages in the requirment.txt
## Install and set it up
### 1. Download the project
Clone this repository or download it as a ZIP file, then open the project folder in VS Code or your terminal.
### 2. Create a virtual environment
In the project folder, run:
```powershell
py -3.12 -m venv venv
```
You can replace `3.12` with another installed Python version from 3.9 to 3.14.
### 3. Activate it

For Windows PowerShell:

```powershell
attached in the main.py as commment in the first line
```
When it is active, you should see `(venv)` at the start of your terminal line.
### 4. Install the required packages
Run this inside the activated environment:

```bash
python -m pip install --upgrade pip
python -m pip install opencv-python mediapipe pyautogui numpy
pip install requirment.txt
```
These packages are used for webcam input, hand tracking, cursor control and image processing.
### 5. Run AirMouse
Run the main Python file. For example, if your main file is called `main.py`:
```bash
python main.py
```
If your main script has a different name, use that filename instead.
## A few setup tips
- Connect your webcam before starting the program.
- This project is designed around a vertically positioned webcam, so keep the camera in the same orientation used when setting up the project.
- Make sure your hand is clearly visible and the room has decent lighting.
- Close the camera app or any other program that may already be using the webcam.
## Built with
- Python
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy
