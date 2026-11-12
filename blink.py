#this file detects blinks from eyelid landmarks
#calibration uses it to skip frames where the eyes are closed or squinting
 
import numpy as np
from config import *
 
#ratio of eyelid gap to eye width for one eye
def get_eye_ratio(
    face_landmarks,
    width,
    height,
    top,
    bottom,
    left,
    right
):
 
    def point(index):
 
        return np.array([
            face_landmarks[index].x*width,
            face_landmarks[index].y*height
        ])
 
    eye_width =np.linalg.norm(
        point(left)-point(right)
    )
 
    if eye_width<0.000001:
        return 0.0
 
    eye_gap =np.linalg.norm(
        point(top)-point(bottom)
    )
 
    return eye_gap/eye_width
 
#get left and right eye openness
def get_eye_openness(
    face_landmarks,
    width,
    height
):
 
    left_ratio =get_eye_ratio(
        face_landmarks,
        width,
        height,
        LEFT_EYE_TOP,
        LEFT_EYE_BOTTOM,
        LEFT_EYE_LEFT,
        LEFT_EYE_RIGHT
    )
 
    right_ratio =get_eye_ratio(
        face_landmarks,
        width,
        height,
        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM,
        RIGHT_EYE_LEFT,
        RIGHT_EYE_RIGHT
    )
 
    return left_ratio,right_ratio
 
#true when both eyes are open enough to trust
def eyes_open(
    face_landmarks,
    width,
    height
):
 
    left_ratio,right_ratio =get_eye_openness(
        face_landmarks,
        width,
        height
    )
 
    return (
        left_ratio>BLINK_RATIO_THRESHOLD and
        right_ratio>BLINK_RATIO_THRESHOLD
    )