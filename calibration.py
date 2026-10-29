#this file handles eye center calibration and monitor calibration
#it maps calibrated gaze yaw and pitch values to positions on the screen

import numpy as np
from config import BASE_EYE_RADIUS

#calibrated eye centers
left_eye_center_local =None
right_eye_center_local =None

#face scale when eye centers were calibrated
eye_calibration_face_scale =None

#store monitor calibration points
monitor_calibration ={}


#monitor mapping models
monitor_x_model =None
monitor_y_model =None


#monitor status
monitor_ready =False


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

    #reset monitor calibration after eye calibration
    reset_monitor_calibration()


#get calibrated eye centers
def get_eye_centers():

    return (
        left_eye_center_local,
        right_eye_center_local,
        eye_calibration_face_scale
    )


#save monitor calibration point
def save_monitor_point(
    key,
    smooth_yaw,
    smooth_pitch
):

    if key =="1":

        monitor_calibration["top_left"] =(
            smooth_yaw,
            smooth_pitch
        )

        print(
            "saved monitor top-left:",
            monitor_calibration["top_left"]
        )

    elif key =="2":

        monitor_calibration["top_right"] =(
            smooth_yaw,
            smooth_pitch
        )

        print(
            "saved monitor top-right:",
            monitor_calibration["top_right"]
        )

    elif key =="3":

        monitor_calibration["bottom_left"] =(
            smooth_yaw,
            smooth_pitch
        )

        print(
            "saved monitor bottom-left:",
            monitor_calibration["bottom_left"]
        )

    elif key =="4":

        monitor_calibration["bottom_right"] =(
            smooth_yaw,
            smooth_pitch
        )

        print(
            "saved monitor bottom-right:",
            monitor_calibration["bottom_right"]
        )

    elif key =="5":

        monitor_calibration["center"] =(
            smooth_yaw,
            smooth_pitch
        )

        print(
            "saved monitor center:",
            monitor_calibration["center"]
        )

    #check if monitor calibration is complete
    check_monitor_calibration()


#build monitor mapping from gaze angles
def build_monitor_mapping():

    global monitor_x_model
    global monitor_y_model

    #create calibration input values
    gaze_samples =np.array([
        [
            monitor_calibration["top_left"][0],
            monitor_calibration["top_left"][1],
            1.0
        ],
        [
            monitor_calibration["top_right"][0],
            monitor_calibration["top_right"][1],
            1.0
        ],
        [
            monitor_calibration["bottom_left"][0],
            monitor_calibration["bottom_left"][1],
            1.0
        ],
        [
            monitor_calibration["bottom_right"][0],
            monitor_calibration["bottom_right"][1],
            1.0
        ],
        [
            monitor_calibration["center"][0],
            monitor_calibration["center"][1],
            1.0
        ]
    ],dtype =np.float64)

    #known horizontal monitor positions
    screen_x =np.array([
        0.0,
        1.0,
        0.0,
        1.0,
        0.5
    ],dtype =np.float64)

    #known vertical monitor positions
    screen_y =np.array([
        0.0,
        0.0,
        1.0,
        1.0,
        0.5
    ],dtype =np.float64)

    #calculate horizontal mapping
    monitor_x_model,_residuals,_rank,_values =np.linalg.lstsq(
        gaze_samples,
        screen_x,
        rcond =None
    )

    #calculate vertical mapping
    monitor_y_model,_residuals,_rank,_values =np.linalg.lstsq(
        gaze_samples,
        screen_y,
        rcond =None
    )


#check if monitor calibration is complete
def check_monitor_calibration():

    global monitor_ready

    if (
        "top_left" in monitor_calibration and
        "top_right" in monitor_calibration and
        "bottom_left" in monitor_calibration and
        "bottom_right" in monitor_calibration and
        "center" in monitor_calibration
    ):

        build_monitor_mapping()

        monitor_ready =True

        print("\nvirtual monitor calibrated")


#map gaze angles to screen
def map_gaze_to_screen(
    smooth_yaw,
    smooth_pitch,
    screen_width,
    screen_height
):

    if not monitor_ready:
        return None

    #create current gaze sample
    gaze_sample =np.array([
        smooth_yaw,
        smooth_pitch,
        1.0
    ])

    #predict normalized monitor position
    monitor_u =np.dot(
        gaze_sample,
        monitor_x_model
    )

    monitor_v =np.dot(
        gaze_sample,
        monitor_y_model
    )

    #convert into screen pixels
    screen_gaze_x =int(
        np.clip(
            monitor_u,
            0.0,
            1.0
        )*
        (screen_width-1)
    )

    screen_gaze_y =int(
        np.clip(
            monitor_v,
            0.0,
            1.0
        )*
        (screen_height-1)
    )

    return (
        screen_gaze_x,
        screen_gaze_y
    )


#check monitor status
def is_monitor_ready():

    return monitor_ready


#reset monitor calibration
def reset_monitor_calibration():

    global monitor_calibration
    global monitor_x_model
    global monitor_y_model
    global monitor_ready

    monitor_calibration ={}

    monitor_x_model =None
    monitor_y_model =None

    monitor_ready =False