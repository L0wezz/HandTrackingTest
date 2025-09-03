import mediapipe as mp
import cv2
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import image_format_pb2


base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils


index_finger_history = []
drawing_enabled = False  
last_time = time.time()  
delay = 0.2 

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
                pixel_x = int(index_finger_tip.x * w)
                pixel_y = int(index_finger_tip.y * h)
                coord = (pixel_x, pixel_y)
                
                RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGBframe)
                recognition_result = recognizer.recognize(mp_image)

                if recognition_result.gestures and recognition_result.gestures[0]:
                    top_gesture = recognition_result.gestures[0][0]
                    print(top_gesture)
                else:
                    print("No gestures detected.")

        else:
            print("No hands detected.")
        

        current_time = time.time()
        if drawing_enabled and (current_time - last_time) > delay and len(index_finger_history) < 100:
            index_finger_history.append(coord)
            last_time = current_time 
        
        for i in range(1, len(index_finger_history)):
            cv2.line(frame, index_finger_history[i - 1], index_finger_history[i], (0, 255, 0), 2)

        cv2.imshow("capture image", frame)
        
       
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('d'):
            drawing_enabled = not drawing_enabled
        elif key == ord('c'):
            index_finger_history = []
        elif key == ord('p'):
            print(index_finger_history)   

cap.release()
cv2.destroyAllWindows()
