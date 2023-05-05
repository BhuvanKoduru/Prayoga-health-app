import base64
import numpy as np
# import tensorflow as tf
#import keras.models
import cv2
import re
import sys
import os
import glob
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) #First
    b = np.array(b) #Mid
    c = np.array(c) #End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def downdog():
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
            img = cv2.imread("Downward-Facing-Dog-Pose_Andrew-Clark_2400x1350.jpeg.webp", cv2.IMREAD_COLOR)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

            #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            #print(results.pose_landmarks)

            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass
            #print(len(landmarks)) 33

            #Downward facing dog pose
            #Angle between right_hip(24), right_ankle(28), right_knee(26) and same goes to left(23,27,25)
            right_hip = [landmarks[24].x, landmarks[24].y]
            right_ankle = [landmarks[28].x, landmarks[28].y]
            right_knee = [landmarks[26].x, landmarks[26].y]

            left_hip = [landmarks[23].x, landmarks[23].y]
            left_ankle = [landmarks[27].x, landmarks[27].y]
            left_knee = [landmarks[25].x, landmarks[25].y]

            right_shoulder = [landmarks[12].x, landmarks[12].y]
            left_shoulder = [landmarks[11].x, landmarks[11].y]

            right_elbow = [landmarks[14].x, landmarks[14].y]
            right_wrist = [landmarks[16].x, landmarks[16].y]

            left_elbow = [landmarks[13].x, landmarks[13].y]
            left_wrist = [landmarks[15].x, landmarks[15].y]

            Rightangle1 = calculate_angle(right_hip, right_knee, right_ankle)
            Rightangle2 = calculate_angle(right_knee, right_hip, right_shoulder)
            Rightangle3 = calculate_angle(right_shoulder, right_elbow, right_wrist)

            Leftangle1 = calculate_angle(left_hip, left_knee, left_ankle)
            Leftangle2 = calculate_angle(left_knee, left_hip, left_shoulder)
            Leftangle3 = calculate_angle(left_shoulder, left_elbow, left_wrist)

            downdogPose = {
                #  'name': 'downwardFacingDogPose',
                 'r1':Rightangle1,
                 'r2':Rightangle2,
                 'r3':Rightangle3,
                 'l1':Leftangle1,
                 'l2':Leftangle2,
                 'l3':Leftangle3
            }
            #print(downdogPose)
            return downdogPose
# dictDog = downdog()
# print(dictDog)

def mountain():
     with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
            img = cv2.imread("Mountain-Pose-–-Tadasana.jpg", cv2.IMREAD_COLOR)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

            #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            #print(results.pose_landmarks)

            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass
            right_shoulder = [landmarks[12].x, landmarks[12].y]
            left_shoulder = [landmarks[11].x, landmarks[11].y]

            right_elbow = [landmarks[14].x, landmarks[14].y]
            right_wrist = [landmarks[16].x, landmarks[16].y]

            left_elbow = [landmarks[13].x, landmarks[13].y]
            left_wrist = [landmarks[15].x, landmarks[15].y]


            right_hip = [landmarks[24].x, landmarks[24].y]
            right_ankle = [landmarks[28].x, landmarks[28].y]
            right_knee = [landmarks[26].x, landmarks[26].y]

            left_hip = [landmarks[23].x, landmarks[23].y]
            left_ankle = [landmarks[27].x, landmarks[27].y]
            left_knee = [landmarks[25].x, landmarks[25].y]

            right_shoulder = [landmarks[12].x, landmarks[12].y]
            left_shoulder = [landmarks[11].x, landmarks[11].y]

            Rightangle1 = calculate_angle(right_hip, right_knee, right_ankle)
            Rightangle2 = calculate_angle(right_knee, right_hip, right_shoulder)
            Rightangle3 = calculate_angle(right_shoulder, right_elbow, right_wrist)

            Leftangle1 = calculate_angle(left_hip, left_knee, left_ankle)
            Leftangle2 = calculate_angle(left_knee, left_hip, left_shoulder)
            Leftangle3 = calculate_angle(left_shoulder, left_elbow, left_wrist)

            mountainPose = {
                #  'name': 'mountainPose',
                 'r1':Rightangle1,
                 'r2':Rightangle2,
                 'r3':Rightangle3,
                 'l1':Leftangle1,
                 'l2':Leftangle2,
                 'l3':Leftangle3
            }
            print(mountainPose)
            return mountainPose
     
def upwardDog():
     with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
            img = cv2.imread("How-To-Do-Upward-Facing-Dog-Pose.jpg", cv2.IMREAD_COLOR)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

            #mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            #print(results.pose_landmarks)

            try:
                landmarks = results.pose_landmarks.landmark
            except:
                pass
            right_shoulder = [landmarks[12].x, landmarks[12].y]
            left_shoulder = [landmarks[11].x, landmarks[11].y]

            right_elbow = [landmarks[14].x, landmarks[14].y]
            right_wrist = [landmarks[16].x, landmarks[16].y]

            left_elbow = [landmarks[13].x, landmarks[13].y]
            left_wrist = [landmarks[15].x, landmarks[15].y]


            right_hip = [landmarks[24].x, landmarks[24].y]
            right_ankle = [landmarks[28].x, landmarks[28].y]
            right_knee = [landmarks[26].x, landmarks[26].y]

            left_hip = [landmarks[23].x, landmarks[23].y]
            left_ankle = [landmarks[27].x, landmarks[27].y]
            left_knee = [landmarks[25].x, landmarks[25].y]

            right_shoulder = [landmarks[12].x, landmarks[12].y]
            left_shoulder = [landmarks[11].x, landmarks[11].y]

            Rightangle1 = calculate_angle(right_hip, right_knee, right_ankle)
            Rightangle2 = calculate_angle(right_knee, right_hip, right_shoulder)
            Rightangle3 = calculate_angle(right_shoulder, right_elbow, right_wrist)

            Leftangle1 = calculate_angle(left_hip, left_knee, left_ankle)
            Leftangle2 = calculate_angle(left_knee, left_hip, left_shoulder)
            Leftangle3 = calculate_angle(left_shoulder, left_elbow, left_wrist)

            updogPose = {
                #  'name': 'updogPose',
                 'r1':Rightangle1,
                 'r2':Rightangle2,
                 'r3':Rightangle3,
                 'l1':Leftangle1,
                 'l2':Leftangle2,
                 'l3':Leftangle3
            }
            #print(updogPose)
            return updogPose

