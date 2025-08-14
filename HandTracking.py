import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


while True:

    success, frame = cap.read()
    if success:
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        result = hand.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, c = frame.shape
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                print(index_finger_tip)
                pixel_x = int(index_finger_tip.x * w)
                pixel_y = int(index_finger_tip.y * h)
                cv2.circle(frame, (pixel_x, pixel_y), 10, (255, 0, 255), cv2.FILLED)
        cv2.imshow("capture image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()

