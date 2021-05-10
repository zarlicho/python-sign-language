import mediapipe as mp 
import cv2
import numpy as np
import csv
import os

from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score 
import pickle

with open('sibi.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic 

lock1 = 0
lock2 = 0
lock3 = 0
lock4 = 0

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        results = holistic.process(image)
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        try:
            handR = results.right_hand_landmarks.landmark
            handR_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in handR]).flatten())
            row = handR_row
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            xc = body_language_class.split(' ')[0]

            if xc == "B":
                lock1 += 1
            elif xc == "U":
                lock2 += 1
            elif xc == "K":
                lock3 += 1
            elif xc == "A":
                lock4 += 1

            if lock1 > 2 and lock2 >2 and lock3 > 2 and lock4 >2:
                print("terbuka")
                
            cv2.putText(image, body_language_class.split(' ')[0]
                    ,(90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass    
        cv2.imshow('Raw Webcam Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
