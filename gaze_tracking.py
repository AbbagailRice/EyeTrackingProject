#this file calculates eye position iris position and gaze direction
#it creates left right and combined gaze rays and converts gaze into yaw and pitch angles

import numpy as np
import math

from collections import deque

from config import *
from face_tracking import landmark_to_3d


#store recent combined gaze directions
combined_gaze_directions =deque(
    maxlen =GAZE_FILTER_LENGTH
)


#get eye and iris information
def get_eye_data(
    face_landmarks,
    width,
    height
):

    #get iris centers
    left_iris =face_landmarks[
        LEFT_IRIS[0]
    ]

    right_iris =face_landmarks[
        RIGHT_IRIS[0]
    ]

    #convert iris landmarks into 3d points
    left_iris_3d =landmark_to_3d(
        left_iris,
        width,
        height
    )

    right_iris_3d =landmark_to_3d(
        right_iris,
        width,
        height
    )

    return {
        "left_iris": left_iris,
        "right_iris": right_iris,
        "left_iris_3d": left_iris_3d,
        "right_iris_3d": right_iris_3d
    }


#convert combined gaze direction into yaw and pitch
def calculate_gaze_angles(
    gaze_direction
):

    #camera looking straight forward
    reference_forward =np.array([
        0.0,
        0.0,
        -1.0
    ])

    #normalize gaze direction
    gaze_direction =(
        gaze_direction/
        np.linalg.norm(gaze_direction)
    )

    #project gaze onto horizontal xz plane
    xz_projection =np.array([
        gaze_direction[0],
        0.0,
        gaze_direction[2]
    ])

    xz_length =np.linalg.norm(
        xz_projection
    )

    if xz_length>0.000001:

        xz_projection =(
            xz_projection/
            xz_length
        )

        yaw_rad =math.acos(
            np.clip(
                np.dot(
                    reference_forward,
                    xz_projection
                ),
                -1.0,
                1.0
            )
        )

        #set horizontal direction
        if gaze_direction[0]<0:
            yaw_rad =-yaw_rad

    else:
        yaw_rad =0.0

    #project gaze onto vertical yz plane
    yz_projection =np.array([
        0.0,
        gaze_direction[1],
        gaze_direction[2]
    ])

    yz_length =np.linalg.norm(
        yz_projection
    )

    if yz_length>0.000001:

        yz_projection =(
            yz_projection/
            yz_length
        )

        pitch_rad =math.acos(
            np.clip(
                np.dot(
                    reference_forward,
                    yz_projection
                ),
                -1.0,
                1.0
            )
        )

        #set vertical direction
        if gaze_direction[1]>0:
            pitch_rad =-pitch_rad

    else:
        pitch_rad =0.0

    #convert to degrees
    yaw =np.degrees(
        yaw_rad
    )

    pitch =np.degrees(
        pitch_rad
    )

    return yaw,pitch


#calculate gaze from calibrated eye spheres
def calculate_gaze(
    face_origin,
    face_basis,
    eye_data,
    left_eye_center_local,
    right_eye_center_local,
    calibration_face_scale,
    current_face_scale
):

    #check if eye spheres are calibrated
    if (
        left_eye_center_local is None or
        right_eye_center_local is None or
        calibration_face_scale is None
    ):
        return None

    #calculate distance scale adjustment
    if calibration_face_scale>0:

        scale_ratio =(
            current_face_scale/
            calibration_face_scale
        )

    else:
        scale_ratio =1.0

    #scale saved eye offsets
    left_scaled_offset =(
        left_eye_center_local*
        scale_ratio
    )

    right_scaled_offset =(
        right_eye_center_local*
        scale_ratio
    )

    #reconstruct eye sphere centers
    left_eye_center_3d =face_origin+(
        face_basis@
        left_scaled_offset
    )

    right_eye_center_3d =face_origin+(
        face_basis@
        right_scaled_offset
    )

    #calculate left gaze direction
    left_gaze_3d =(
        eye_data["left_iris_3d"]-
        left_eye_center_3d
    )

    #calculate right gaze direction
    right_gaze_3d =(
        eye_data["right_iris_3d"]-
        right_eye_center_3d
    )

    #check gaze vectors
    if (
        np.linalg.norm(left_gaze_3d)<0.000001 or
        np.linalg.norm(right_gaze_3d)<0.000001
    ):
        return None

    #normalize left gaze
    left_gaze_3d =(
        left_gaze_3d/
        np.linalg.norm(left_gaze_3d)
    )

    #normalize right gaze
    right_gaze_3d =(
        right_gaze_3d/
        np.linalg.norm(right_gaze_3d)
    )

    #combine both gaze directions
    raw_combined_gaze =(
        left_gaze_3d+
        right_gaze_3d
    )/2

    #normalize combined direction
    raw_combined_gaze =(
        raw_combined_gaze/
        np.linalg.norm(raw_combined_gaze)
    )

    #store combined gaze for smoothing
    combined_gaze_directions.append(
        raw_combined_gaze
    )

    #average recent combined directions
    combined_gaze_3d =np.mean(
        combined_gaze_directions,
        axis =0
    )

    #normalize smoothed gaze
    combined_gaze_3d =(
        combined_gaze_3d/
        np.linalg.norm(combined_gaze_3d)
    )

    #calculate yaw and pitch
    smooth_yaw,smooth_pitch =calculate_gaze_angles(
        combined_gaze_3d
    )

    #find midpoint between eyes
    combined_eye_center_3d =(
        left_eye_center_3d+
        right_eye_center_3d
    )/2

    #convert eye centers to pixels
    left_eye_center_2d =(
        int(left_eye_center_3d[0]),
        int(left_eye_center_3d[1])
    )

    right_eye_center_2d =(
        int(right_eye_center_3d[0]),
        int(right_eye_center_3d[1])
    )

    combined_eye_center_2d =(
        int(combined_eye_center_3d[0]),
        int(combined_eye_center_3d[1])
    )

    #calculate left gaze endpoint
    left_ray_end =(
        int(
            left_eye_center_3d[0]+
            left_gaze_3d[0]*
            GAZE_RAY_LENGTH
        ),
        int(
            left_eye_center_3d[1]+
            left_gaze_3d[1]*
            GAZE_RAY_LENGTH
        )
    )

    #calculate right gaze endpoint
    right_ray_end =(
        int(
            right_eye_center_3d[0]+
            right_gaze_3d[0]*
            GAZE_RAY_LENGTH
        ),
        int(
            right_eye_center_3d[1]+
            right_gaze_3d[1]*
            GAZE_RAY_LENGTH
        )
    )

    #calculate combined gaze endpoint
    combined_ray_end =(
        int(
            combined_eye_center_3d[0]+
            combined_gaze_3d[0]*
            GAZE_RAY_LENGTH
        ),
        int(
            combined_eye_center_3d[1]+
            combined_gaze_3d[1]*
            GAZE_RAY_LENGTH
        )
    )

    return {
        "left_eye_center_3d": left_eye_center_3d,
        "right_eye_center_3d": right_eye_center_3d,
        "combined_eye_center_3d": combined_eye_center_3d,

        "left_eye_center_2d": left_eye_center_2d,
        "right_eye_center_2d": right_eye_center_2d,
        "combined_eye_center_2d": combined_eye_center_2d,

        "left_gaze_3d": left_gaze_3d,
        "right_gaze_3d": right_gaze_3d,
        "combined_gaze_3d": combined_gaze_3d,

        "left_ray_end": left_ray_end,
        "right_ray_end": right_ray_end,
        "combined_ray_end": combined_ray_end,

        "smooth_yaw": smooth_yaw,
        "smooth_pitch": smooth_pitch,

        "scale_ratio": scale_ratio
    }