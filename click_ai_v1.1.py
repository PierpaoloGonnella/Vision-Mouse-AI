import cv2
import mediapipe as mp
import pyautogui
import math

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)

drawing_utils = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
index_y = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Get the x, y coordinates of the tip of the index finger
            index_x = int(
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                * frame_width
            )
            index_y = int(
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                * frame_height
            )

            # Get the x, y coordinates of the tip of the thumb
            thumb_x = int(
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame_width
            )
            thumb_y = int(
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                * frame_height
            )

            # Calculate the distance between the index finger tip and thumb tip
            distance = math.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            # If the distance is less than a certain threshold, trigger a click
            if distance < 50:
                pyautogui.click()
                pyautogui.sleep(0.2)

            # Move the mouse cursor to the location of the index finger tip
            # pyautogui.moveTo(index_x, index_y)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
