import cv2
import mediapipe as mp
import pyautogui

# Initialize the camera and hand detector
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Get the screen size for mouse control
screen_width, screen_height = pyautogui.size()

show_video = True

while True:
    success, frame = cap.read()
    if not success:
        break  # If the frame is not successfully captured, exit the loop

    # Flip the frame and convert it to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks and classification
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if show_video:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate the central point of the hand for cursor movement
            landmarks = hand_landmarks.landmark
            wrist = landmarks[mp_hands.HandLandmark.WRIST]
            middle_finger_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            central_x = int(((wrist.x + middle_finger_mcp.x) / 2) * screen_width)
            central_y = int(((wrist.y + middle_finger_mcp.y) / 2) * screen_height)

            # Get the coordinates of the index finger tip and the thumb tip for click detection
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_x = index_tip.x * screen_width
            index_y = index_tip.y * screen_height
            thumb_x = thumb_tip.x * screen_width
            thumb_y = thumb_tip.y * screen_height
            distance = ((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2) ** 0.5

            # Use central point for cursor movement, maintain clicking logic
            label = handedness.classification[0].label
            if label == "Right":
                pyautogui.moveTo(central_x, central_y, duration=0.1)  # Move the cursor to the central point
                if distance < 50:  # Threshold for click action
                    pyautogui.click()
                    pyautogui.sleep(0.2)  # Prevent multiple clicks in a short time

            # Left hand for right-click
            elif label == "Left" and distance < 50:
                pyautogui.rightClick()
                pyautogui.sleep(0.2)  # Prevent multiple clicks

    # Show the frame
    if show_video:
        cv2.imshow("Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()
