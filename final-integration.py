import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
import vlc
import train as train
import sys, webbrowser, datetime, time

def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10])+euclideanDist(mouth[4], mouth[8]))/(2*euclideanDist(mouth[0], mouth[6])))

def getFaceDirection(shape, size):
    image_points = np.array([
                                shape[33],    # Nose tip
                                shape[8],     # Chin
                                shape[45],    # Left eye left corner
                                shape[36],    # Right eye right corne
                                shape[54],    # Left Mouth corner
                                shape[48]     # Right mouth corner
                            ], dtype="double")

    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner

                            ])

    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return(translation_vector[1][0])

def euclideanDist(a, b):
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))

#EAR -> Eye Aspect ratio
def ear(eye):
    return ((euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3])))

# open_avg = train.getAvg()
# close_avg = train.getAvg()

#alert = vlc.MediaPlayer('focus.mp3')

fps = 10
frame_thresh_0 = 15*fps/3
frame_thresh_1 = 10*fps/3
frame_thresh_2 = 8*fps/3
frame_thresh_3 = 5*fps/3

close_thresh = 0.3#(close_avg+open_avg)/2.0
eye_closed_timer = 0
yawning = 0
level1_counter = 0
level2_counter = 0
level3_counter = 0
level4_counter = 0
level1_flag = 1
level2_flag = 1
level3_flag = 1
level4_flag = 1

# print(close_thresh)

capture = cv2.VideoCapture(-1)
avgEAR = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

while(True):
    #input()
    time.sleep(1.0/fps)
    try:
        ret, frame = capture.read()
        size = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame
        rects = detector(gray, 0)
        if(len(rects)):
            shape = face_utils.shape_to_np(predictor(gray, rects[0]))
            leftEye = shape[leStart:leEnd]
            rightEye = shape[reStart:reEnd]
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            #print("Mouth Open Ratio", yawn(shape[mStart:mEnd]))
            leftEAR = ear(leftEye) #Get the left eye aspect ratio
            rightEAR = ear(rightEye) #Get the right eye aspect ratio
            avgEAR = (leftEAR+rightEAR)/2.0
            #print("Eye Aspect Ratio", avgEAR)
            #print(eye_closed_timer, level1_counter, level2_counter, level3_counter, level4_counter)
            eyeContourColor = (255, 255, 255)

            if(yawn(shape[mStart:mEnd]) > 0.55):
                #cv2.putText(gray, "Yawn Detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                if not yawning:
                    print("Yawn Detected")
                yawning = 1
                cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
                cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)

            yawning = 0

            if(avgEAR < close_thresh):
                eye_closed_timer += 1
                eyeContourColor = (0,255,255)
                if eye_closed_timer >= frame_thresh_0:
                    eyeContourColor = (0, 0, 255)
                    cv2.putText(gray, "Drowsiness Level 4", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                    #alert.play()
                    if(level4_flag):
                        level4_flag = 0
                        level4_counter += 1
                elif eye_closed_timer >= frame_thresh_1:
                    eyeContourColor = (0, 0, 255)
                    cv2.putText(gray, "Drowsiness Level 3", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                    #alert.play()
                    if(level3_flag):
                        level3_flag = 0
                        level3_counter += 1
                elif eye_closed_timer >= frame_thresh_2:
                    eyeContourColor = (255, 0, 0)
                    cv2.putText(gray, "Drowsiness Level 2", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                    #alert.play()
                    if(level2_flag):
                        level2_flag = 0
                        level2_counter += 1
                elif (yawning or eye_closed_timer >= frame_thresh_3):
                    eyeContourColor = (147, 20, 255)
                    cv2.putText(gray, "Drowsiness Level 1", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                    #alert.play()
                    if(level1_flag):
                        level1_flag = 0
                        level1_counter += 1

            elif(avgEAR > close_thresh and eye_closed_timer):
                #alert.stop()
                yawning=0
                level1_flag = 1
                level2_flag = 1
                level3_flag = 1
                level4_flag = 1
                eye_closed_timer=0

            if level4_counter >= 1:
                capture.release()
                cv2.destroyAllWindows()
                exit(1) # do something to stop the vehicle, maybe alert agency and police
            if level3_counter >= 3:
                capture.release()
                cv2.destroyAllWindows()
                exit(2) # take a break
            if level2_counter >= 5:
                capture.release()
                cv2.destroyAllWindows()
                exit(3) # report tired, less concentration


            cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
            cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)

        if(avgEAR>close_thresh):
            pass
            #alert.stop()
        cv2.imshow('Driver', gray)
        if(cv2.waitKey(1)==27):
            break
    except:
        continue
