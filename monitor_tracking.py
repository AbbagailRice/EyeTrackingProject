#this file creates a virtual monitor plane and tracks where the gaze ray hits it
#it converts the 3d gaze intersection into a normalized monitor and screen coordinates 
#idea from:https://github.com/JEOresearch/EyeTracker/blob/main/Webcam3DTracker/MonitorTracking.py

import numpy as np

from config import *
from face_tracking import landmark_to_3d

#virtual monitor information
monitor_corners =None
monitor_center =None
monitor_normal =None

#monitor status
monitor_ready =False

#scale used to create monitor
units_per_cm =None

#store calibration samples
monitor_calibration ={}

#screen correction models
screen_x_model =None
screen_y_model =None

#normalize vector
def normalize(vector):

    length =np.linalg.norm(vector)

    if length<0.000001:
        return vector

    return vector/length


#estimate world units per centimeter
def calculate_units_per_cm(
    face_landmarks,
    width,
    height
):

    #get chin and forehead points
    chin =landmark_to_3d(
        face_landmarks[CHIN],
        width,
        height
    )

    forehead =landmark_to_3d(
        face_landmarks[FOREHEAD],
        width,
        height
    )

    #calculate face height
    face_height =np.linalg.norm(
        forehead-chin
    )

    #approximate chin to forehead as 15 cm
    units_per_cm =face_height/15.0

    return units_per_cm


#create virtual monitor
def calibrate_monitor(
    gaze_origin,
    gaze_direction,
    face_landmarks,
    width,
    height
):

    global monitor_corners
    global monitor_center
    global monitor_normal
    global monitor_ready
    global units_per_cm
    global monitor_calibration
    global screen_x_model
    global screen_y_model

    #calculate scale
    units_per_cm =calculate_units_per_cm(
        face_landmarks,
        width,
        height
    )

    #normalize center gaze direction
    monitor_normal =normalize(
        gaze_direction.copy()
    )

    #convert monitor distance into world units
    monitor_distance =(
        MONITOR_DISTANCE_CM*
        units_per_cm
    )

    #place monitor center directly along center gaze
    monitor_center =(
        gaze_origin+
        monitor_normal*
        monitor_distance
    )

    #world up direction
    world_up =np.array([
        0.0,
        -1.0,
        0.0
    ])

    #calculate monitor right direction
    monitor_right =np.cross(
        world_up,
        monitor_normal
    )

    monitor_right =normalize(
        monitor_right
    )

    #calculate monitor up direction
    monitor_up =np.cross(
        monitor_normal,
        monitor_right
    )

    monitor_up =normalize(
        monitor_up
    )

    #convert monitor dimensions into world units
    half_width =(
        MONITOR_WIDTH_CM*
        0.5*
        units_per_cm
    )

    half_height =(
        MONITOR_HEIGHT_CM*
        0.5*
        units_per_cm
    )

    #top-left
    point_0 =(
        monitor_center-
        monitor_right*half_width+
        monitor_up*half_height
    )

    #top-right
    point_1 =(
        monitor_center+
        monitor_right*half_width+
        monitor_up*half_height
    )

    #bottom-right
    point_2 =(
        monitor_center+
        monitor_right*half_width-
        monitor_up*half_height
    )

    #bottom-left
    point_3 =(
        monitor_center-
        monitor_right*half_width-
        monitor_up*half_height
    )

    #store monitor corners
    monitor_corners =[
        point_0,
        point_1,
        point_2,
        point_3
    ]

    #reset screen calibration
    monitor_calibration ={}
    screen_x_model =None
    screen_y_model =None

    monitor_ready =True

    #save center calibration
    monitor_calibration["center"] =(
        0.5,
        0.5
    )

    print("\nvirtual monitor calibrated")
    print("units per cm:",units_per_cm)
    print("monitor center:",monitor_center)

#find where gaze ray hits monitor
def get_gaze_intersection(
    gaze_origin,
    gaze_direction
):

    if not monitor_ready:
        return None

    #normalize gaze direction
    gaze_direction =normalize(
        gaze_direction
    )

    #calculate ray plane relationship
    denominator =np.dot(
        monitor_normal,
        gaze_direction
    )

    #gaze ray is almost parallel to monitor
    if abs(denominator)<0.000001:
        return None

    #calculate intersection distance
    distance =np.dot(
        monitor_normal,
        monitor_center-gaze_origin
    )/denominator

    #intersection is behind the eyes
    if distance<=0:
        return None

    #calculate intersection point
    intersection =(
        gaze_origin+
        gaze_direction*distance
    )

    return intersection

#convert intersection into monitor coordinates
def get_monitor_coordinates(
    intersection
):

    if (
        not monitor_ready or
        intersection is None
    ):
        return None

    #get monitor corners
    point_0 =monitor_corners[0]
    point_1 =monitor_corners[1]
    point_3 =monitor_corners[3]

    #horizontal monitor vector
    horizontal =point_1-point_0

    #vertical monitor vector
    vertical =point_3-point_0

    #intersection relative to top-left
    relative =intersection-point_0

    horizontal_length =np.dot(
        horizontal,
        horizontal
    )

    vertical_length =np.dot(
        vertical,
        vertical
    )

    if (
        horizontal_length<0.000001 or
        vertical_length<0.000001
    ):
        return None

    #normalized horizontal position
    monitor_x =np.dot(
        relative,
        horizontal
    )/horizontal_length

    #normalized vertical position
    monitor_y =np.dot(
        relative,
        vertical
    )/vertical_length

    return monitor_x,monitor_y

#save one screen calibration point
def save_calibration_point(
    key,
    gaze_origin,
    gaze_direction
):

    if not monitor_ready:
        print("calibrate monitor center first")
        return

    #find current raw gaze position
    intersection =get_gaze_intersection(
        gaze_origin,
        gaze_direction
    )

    raw_position =get_monitor_coordinates(
        intersection
    )

    if raw_position is None:
        print("could not save calibration point")
        return

    raw_x,raw_y =raw_position

    #save raw gaze position and known target
    if key =="1":

        monitor_calibration["top_left"] =(
            raw_x,
            raw_y,
            0.05,
            0.05
        )

        print(
            "saved top-left:",
            raw_x,
            raw_y
        )

    elif key =="2":

        monitor_calibration["top_right"] =(
            raw_x,
            raw_y,
            0.95,
            0.05
        )

        print(
            "saved top-right:",
            raw_x,
            raw_y
        )

    elif key =="3":

        monitor_calibration["bottom_left"] =(
            raw_x,
            raw_y,
            0.05,
            0.95
        )

        print(
            "saved bottom-left:",
            raw_x,
            raw_y
        )

    elif key =="4":

        monitor_calibration["bottom_right"] =(
            raw_x,
            raw_y,
            0.95,
            0.95
        )

        print(
            "saved bottom-right:",
            raw_x,
            raw_y
        )

    #check if all calibration points exist
    if (
        "top_left" in monitor_calibration and
        "top_right" in monitor_calibration and
        "bottom_left" in monitor_calibration and
        "bottom_right" in monitor_calibration
    ):

        build_screen_mapping()


#build correction from raw monitor coordinates
def build_screen_mapping():

    global screen_x_model
    global screen_y_model

    #get calibration points
    points =[
        monitor_calibration["top_left"],
        monitor_calibration["top_right"],
        monitor_calibration["bottom_left"],
        monitor_calibration["bottom_right"]
    ]

    #raw monitor coordinates
    raw_samples =np.array([
        [points[0][0],points[0][1],1.0],
        [points[1][0],points[1][1],1.0],
        [points[2][0],points[2][1],1.0],
        [points[3][0],points[3][1],1.0],

        #center is known from monitor calibration
        [0.5,0.5,1.0]
    ],dtype =np.float64)

    #known screen x positions
    target_x =np.array([
        0.05,
        0.95,
        0.05,
        0.95,
        0.5
    ],dtype =np.float64)

    #known screen y positions
    target_y =np.array([
        0.05,
        0.05,
        0.95,
        0.95,
        0.5
    ],dtype =np.float64)

    #fit correction model
    screen_x_model,_residuals,_rank,_values =np.linalg.lstsq(
        raw_samples,
        target_x,
        rcond =None
    )

    screen_y_model,_residuals,_rank,_values =np.linalg.lstsq(
        raw_samples,
        target_y,
        rcond =None
    )

    print("\n5-point screen calibration complete")

#map gaze ray to screen
def map_gaze_to_screen(
    gaze_origin,
    gaze_direction,
    screen_width,
    screen_height
):

    if not monitor_ready:
        return None

    #find monitor intersection
    intersection =get_gaze_intersection(
        gaze_origin,
        gaze_direction
    )

    if intersection is None:
        return None

    #convert intersection to monitor position
    monitor_position =get_monitor_coordinates(
        intersection
    )

    if monitor_position is None:
        return None

    raw_x,raw_y =monitor_position

    #use calibration correction if available
    if (
        screen_x_model is not None and
        screen_y_model is not None
    ):

        sample =np.array([
            raw_x,
            raw_y,
            1.0
        ])

        monitor_x =np.dot(
            sample,
            screen_x_model
        )

        monitor_y =np.dot(
            sample,
            screen_y_model
        )

    else:

        monitor_x =raw_x
        monitor_y =raw_y

    #convert normalized monitor position into screen pixels
    screen_x =int(
        np.clip(
            monitor_x,
            0.0,
            1.0
        )*
        (screen_width-1)
    )

    screen_y =int(
        np.clip(
            monitor_y,
            0.0,
            1.0
        )*
        (screen_height-1)
    )

    return screen_x,screen_y

#check monitor status
def is_monitor_ready():
    return monitor_ready

#reset monitor
def reset_monitor():

    global monitor_corners
    global monitor_center
    global monitor_normal
    global monitor_ready
    global units_per_cm
    global monitor_calibration
    global screen_x_model
    global screen_y_model

    monitor_corners =None
    monitor_center =None
    monitor_normal =None
    units_per_cm =None

    monitor_calibration ={}

    screen_x_model =None
    screen_y_model =None

    monitor_ready =False