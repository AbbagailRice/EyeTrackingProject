#this file handles the webcam and preps frames for mediapipe
#it opens the camera creates the camera matrix and converts frames for face tracking

import cv2
import mediapipe as mp
import numpy as np
import time

from config import CAMERA_WIDTH,CAMERA_HEIGHT


#open webcam
def open_camera():

    cap =cv2.VideoCapture(0)

    #set webcam resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    #check if webcam opened
    if not cap.isOpened():
        print("could not open webcam")
        return None

    return cap


#create camera matrix
def create_camera_matrix(width,height):

    focal_length =width

    camera_matrix =np.array([
        [focal_length,0,width/2],
        [0,focal_length,height/2],
        [0,0,1]
    ],dtype =np.float64)

    return camera_matrix


#create distortion coefficients
def create_distortion_coefficients():

    #assume no lens distortion
    distortion_coefficients =np.zeros((4,1))

    return distortion_coefficients


#convert frame into mediapipe image
def frame_to_mediapipe(frame):

    #convert frame from bgr to rgb
    rgb_frame =cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    #convert frame into mediapipe image
    mp_image =mp.Image(
        image_format =mp.ImageFormat.SRGB,
        data =rgb_frame
    )

    return mp_image


#create timestamp in ms
def create_timestamp(start_time):

    timestamp_ms =int(
        (time.time()-start_time)*1000
    )

    return timestamp_ms