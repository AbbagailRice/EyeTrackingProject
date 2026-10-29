#this file handles eye center calibration and monitor calibration
#it maps calibrated gaze yaw and pitch values to positions on the screen

import numpy as np
from config import BASE_EYE_RADIUS

#calibrated eye centers
left_eye_center_local =None
right_eye_center_local =None

#face scale when eye centers were calibrated
eye_calibration_face_scale =None

#calibrate eye sphere centers
def calibrate_eye_centers(
    face_origin,
    face_basis,
    left_iris_3d,
    right_iris_3d,
    current_face_scale
):

    global left_eye_center_local
    global right_eye_center_local
    global eye_calibration_face_scale

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

    #move behind iris to approximate eyeball center
    left_eye_center_local =(
        left_iris_local+
        BASE_EYE_RADIUS*
        camera_direction_local
    )

    right_eye_center_local =(
        right_iris_local+
        BASE_EYE_RADIUS*
        camera_direction_local
    )

    #save face scale at calibration
    eye_calibration_face_scale =current_face_scale

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

#get calibrated eye centers
def get_eye_centers():

    return (
        left_eye_center_local,
        right_eye_center_local,
        eye_calibration_face_scale
    )
