#this file handles eye center calibration and monitor calibration
#it maps calibrated gaze yaw and pitch values to positions on the screen

import numpy as np
from config import BASE_EYE_RADIUS, EYE_CALIBRATION_FRAMES

#calibrated eye centers
left_eye_center_local =None
right_eye_center_local =None

#face scale when eye centers were calibrated
eye_calibration_face_scale =None

#samples collected during an active calibration (none when idle)
eye_samples =None

#start collecting frames for eye center calibration
def start_eye_calibration():
 
    global eye_samples
 
    eye_samples ={
        "left": [],
        "right": [],
        "scale": []
    }
 
    print("\ncollecting eye calibration frames, look at the camera")
 
#true while frames are being collected
def is_eye_calibrating():
    return eye_samples is not None
 
#number of good frames collected so far
def get_eye_calibration_progress():
    if eye_samples is None:
        return 0
 
    return len(eye_samples["scale"])
 
#add one frame to the eye calibration
#returns true on the frame the calibration finishes
def update_eye_calibration(
    face_origin,
    face_basis,
    left_iris_3d,
    right_iris_3d,
    current_face_scale,
    eyes_are_open
):
 
    if eye_samples is None:
        return False
 
    #skip blink and squint frames
    if not eyes_are_open:
        return False
 
    #convert iris positions into local face coordinates
    left_iris_local =face_basis.T@(
        left_iris_3d-face_origin
    )
 
    right_iris_local =face_basis.T@(
        right_iris_3d-face_origin
    )
 
    #camera direction in camera coordinates
    camera_direction =np.array([
        0.0,
        0.0,
        1.0
    ])
 
    #convert camera direction into local face coordinates
    camera_direction_local =face_basis.T@(
        camera_direction
    )
 
    camera_direction_local =(
        camera_direction_local/
        np.linalg.norm(camera_direction_local)
    )
 
    #move behind iris to approx eyeball center for this frame
    eye_samples["left"].append(
        left_iris_local+
        BASE_EYE_RADIUS*
        camera_direction_local
    )
 
    eye_samples["right"].append(
        right_iris_local+
        BASE_EYE_RADIUS*
        camera_direction_local
    )
 
    eye_samples["scale"].append(
        current_face_scale
    )
 
    if len(eye_samples["scale"])<EYE_CALIBRATION_FRAMES:
        return False
 
    finish_eye_calibration()
 
    return True
 
#combine collected frames into the final eye centers
def finish_eye_calibration():
 
    global left_eye_center_local
    global right_eye_center_local
    global eye_calibration_face_scale
    global eye_samples
 
    left_samples =np.array(
        eye_samples["left"]
    )
 
    right_samples =np.array(
        eye_samples["right"]
    )
 
    #median ignores outlier frames
    left_eye_center_local =np.median(
        left_samples,
        axis =0
    )
 
    right_eye_center_local =np.median(
        right_samples,
        axis =0
    )
 
    #save face scale at calibration
    eye_calibration_face_scale =float(
        np.median(
            eye_samples["scale"]
        )
    )
 
    #median absolute deviation will show how steady the frames were
    left_spread =np.median(
        np.abs(left_samples-left_eye_center_local),
        axis =0
    )
 
    right_spread =np.median(
        np.abs(right_samples-right_eye_center_local),
        axis =0
    )
 
    eye_samples =None
 
    print("\neye sphere centers calibrated")
 
    print(
        "left eye:",
        left_eye_center_local
    )
 
    print(
        "right eye:",
        right_eye_center_local
    )
 
    print(
        "face scale:",
        eye_calibration_face_scale
    )
 
    print(
        "frame spread (px) left/right:",
        left_spread,
        right_spread
    )

#get calibrated eye centers
def get_eye_centers():

    return (
        left_eye_center_local,
        right_eye_center_local,
        eye_calibration_face_scale
    )
