### Vision-Mouse-AI

This script enables control of the computer mouse using hand gestures detected through a webcam. It utilizes the following technologies:

- **OpenCV** (`cv2`): For capturing video from the webcam and processing frames.
- **MediaPipe** (`mediapipe`): To detect hand landmarks in real-time.
- **PyAutoGUI**: For controlling the mouse cursor and simulating clicks.

#### Requirements
- Python 3.x
- Libraries: `cv2`, `mediapipe`, `pyautogui`

#### Setup and Usage
1. **Install Dependencies:**

pip install opencv-python mediapipe pyautogui

2. **Run the Script:**


3. **Hand Gestures:**
- **Right Hand:**
  - **Cursor Movement:** Use the central point between wrist and middle finger MCP.
  - **Left Click:** Bring thumb and index finger close (within 50 pixels).
- **Left Hand:**
  - **Right Click:** Bring thumb and index finger close (within 50 pixels).

4. **Keyboard Commands:**
- Press **'q'** to exit the program.

#### Functionality
- The script initializes the webcam and detects hand landmarks using MediaPipe.
- It calculates the central point of the hand for mouse cursor control.
- Hand gestures are interpreted to perform left and right clicks using PyAutoGUI.
- Optionally, it displays the webcam feed with overlaid hand landmarks.

#### Notes
- Adjust `min_detection_confidence` in `mp.solutions.hands.Hands` for better landmark detection as per lighting conditions.
- Ensure proper lighting and clear hand visibility for optimal performance.


