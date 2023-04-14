import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
import pickle
import cv2 as cv
import googletrans
from googletrans import Translator

# Initialize translator
translator = Translator()

def image_processed(hand_img):
    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        data = str(data)
        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])


# Load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)


# Define languages
languages = {
    'en': 'english',
    'ta': 'tamil',
    'hi': 'hindi'
}

# Ask user for language choice
while True:
    lang_choice = input('Enter language choice (en for English, ta for Tamil, hi for Hindi): ')
    if lang_choice in languages:
        break
    print('Invalid choice, please try again.')

# Get language name
lang_name = languages[lang_choice]

# Main loop
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    data = image_processed(frame)
    data = np.array(data)

    # Predict gesture using model
    y_pred = svm.predict(data.reshape(-1,63))

    # Translate output based on language choice
    output_text = str(y_pred[0])
    translated = translator.translate(output_text, dest=lang_choice).text

    print(translated)
    

    # Add text to frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 100)
    fontScale = 3
    color = (255, 0, 0)
    thickness = 5
    frame = cv2.putText(frame, translated, org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
