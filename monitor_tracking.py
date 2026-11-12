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

#current corner calibration collection
pending_calibration_key =None

#store calibration samples
monitor_calibration ={}

#screen correction models
screen_x_model =None
screen_y_model =None

#current corner calibration collection
pending_calibration_key =None

#store temporary corner samples
calibration_samples =[]

#frames collected for the virtual monitor (none when idle)
monitor_samples =None

#current visual calibration target
calibration_target ="center"

#corner calibration order
calibration_order =[
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]
 
CALIBRATION_TARGETS ={
    "1": ("center",0.50,0.50),
    "2": ("top_left",0.10,0.10),
    "3": ("top_center",0.50,0.10),
    "4": ("top_right",0.90,0.10),
    "5": ("middle_left",0.10,0.50),
    "6": ("middle_right",0.90,0.50),
    "7": ("bottom_left",0.10,0.90),
    "8": ("bottom_center",0.50,0.90),
    "9": ("bottom_right",0.90,0.90)
}
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

#start collecting samples for one calibration point
def start_calibration_point(key):

    global pending_calibration_key
    global calibration_samples

    if not monitor_ready:
        return

    #check valid calibration key
    if key not in CALIBRATION_TARGETS:
        return

    target_name,_target_x,_target_y =CALIBRATION_TARGETS[key]

    #only allow currently displayed target
    if calibration_target !=target_name:
        return

    pending_calibration_key =key
    calibration_samples =[]

#start collecting frames for the virtual monitor
#the user should look at the center of the screen and hold still
def start_monitor_calibration():
 
    global monitor_samples
 
    monitor_samples ={
        "origin": [],
        "direction": [],
        "units": []
    }
 
#true while monitor frames are being collected
def is_monitor_calibrating():
    return monitor_samples is not None
 
 
#number of good monitor frames collected so far
def get_monitor_calibration_progress():
    if monitor_samples is None:
        return 0
 
    return len(monitor_samples["units"])

#add one frame to the monitor calibration
#returns true on the frame the monitor is created
def update_monitor_calibration(
    gaze_data,
    face_landmarks,
    width,
    height,
    eyes_are_open
):
 
    if monitor_samples is None:
        return False
 
    #need calibrated eyes and open eyes
    if gaze_data is None or not eyes_are_open:
        return False
 
    monitor_samples["origin"].append(
        np.array(gaze_data["combined_eye_center_3d"])
    )
 
    monitor_samples["direction"].append(
        np.array(gaze_data["combined_gaze_3d"])
    )
 
    monitor_samples["units"].append(
        calculate_units_per_cm(
            face_landmarks,
            width,
            height
        )
    )
 
    if len(monitor_samples["units"])<MONITOR_CALIBRATION_FRAMES:
        return False
 
    #median of every frame instead of one single frame
    origin =np.median(
        np.array(monitor_samples["origin"]),
        axis =0
    )
 
    direction =normalize(
        np.median(
            np.array(monitor_samples["direction"]),
            axis =0
        )
    )
 
    scale =float(
        np.median(
            monitor_samples["units"]
        )
    )
 
    calibrate_monitor(
        origin,
        direction,
        scale
    )
 
    return True
 
#create virtual monitor
def calibrate_monitor(
    gaze_origin,
    gaze_direction,
    scale_units_per_cm
):
 
    global monitor_corners
    global monitor_center
    global monitor_normal
    global monitor_ready
    global units_per_cm
    global monitor_calibration
    global screen_x_model
    global screen_y_model
    global calibration_target
    global monitor_samples
    global pending_calibration_key
    global calibration_samples

    #store scale
    units_per_cm =scale_units_per_cm

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

    pending_calibration_key =None
    calibration_samples =[]
 
    monitor_samples =None

    monitor_ready =True

    #save center point from monitor calibration
    monitor_calibration["5"] =(
        0.5,
        0.5,
        0.5,
        0.5
    )
    
    #show next target
    calibration_target ="center"

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

#collect one frame for active corner calibration
def collect_calibration_sample(
    raw_x,
    raw_y
):

    global pending_calibration_key
    global calibration_samples
    global calibration_target

    #nothing currently being calibrated
    if pending_calibration_key is None:
        return

    #store current raw point
    calibration_samples.append(
        (
            raw_x,
            raw_y
        )
    )

    #wait until enough frames are collected
    if len(calibration_samples)<CALIBRATION_SAMPLE_COUNT:
        return

    #convert samples into array
    samples =np.array(
        calibration_samples,
        dtype =np.float64
    )

    #use median to reduce effect of jumps
    average_x =np.median(
        samples[:,0]
    )

    average_y =np.median(
        samples[:,1]
    )

    key =pending_calibration_key

    target_name,target_x,target_y =CALIBRATION_TARGETS[key]

    #save calibration point
    monitor_calibration[key] =(
        average_x,
        average_y,
        target_x,
        target_y
    )

    print(
        "saved",
        target_name,
        ":",
        average_x,
        average_y
    )

    #find current point in calibration order
    current_index =calibration_order.index(
        key
    )

    #move to next visual target
    if current_index<len(calibration_order)-1:

        next_key =calibration_order[
            current_index+1
        ]

        calibration_target =CALIBRATION_TARGETS[
            next_key
        ][0]

    else:

        calibration_target =None

    #reset temporary samples
    pending_calibration_key =None
    calibration_samples =[]

    #build mapping after all 9 points
    if len(monitor_calibration) ==9:

        build_screen_mapping()


#build quadratic correction from raw monitor coordinates
def build_screen_mapping():

    global screen_x_model
    global screen_y_model

    raw_samples =[]
    target_x =[]
    target_y =[]

    #all points used by calibration math
    mapping_order =[
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9"
    ]

    #build calibration arrays
    for key in mapping_order:

        raw_x,raw_y,screen_x,screen_y =monitor_calibration[
            key
        ]

        #quadratic calibration features
        raw_samples.append([
            raw_x,
            raw_y,
            raw_x*raw_y,
            raw_x*raw_x,
            raw_y*raw_y,
            1.0
        ])

        target_x.append(
            screen_x
        )

        target_y.append(
            screen_y
        )

    raw_samples =np.array(
        raw_samples,
        dtype =np.float64
    )

    target_x =np.array(
        target_x,
        dtype =np.float64
    )

    target_y =np.array(
        target_y,
        dtype =np.float64
    )

    #fit horizontal correction
    screen_x_model,_residuals,_rank,_values =np.linalg.lstsq(
        raw_samples,
        target_x,
        rcond =None
    )

    #fit vertical correction
    screen_y_model,_residuals,_rank,_values =np.linalg.lstsq(
        raw_samples,
        target_y,
        rcond =None
    )

    print("\n9-point screen calibration complete")

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

    #collect calibration data if a corner is active
    collect_calibration_sample(
        raw_x,
        raw_y
    )

    #use calibration correction if available
    if (
        screen_x_model is not None and
        screen_y_model is not None
    ):

        sample =np.array([
            raw_x,
            raw_y,
            raw_x*raw_y,
            raw_x*raw_x,
            raw_y*raw_y,
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
    global pending_calibration_key
    global calibration_samples
    global calibration_target
    global monitor_samples

    monitor_corners =None
    monitor_center =None
    monitor_normal =None
    units_per_cm =None

    monitor_calibration ={}

    screen_x_model =None
    screen_y_model =None

    monitor_ready =False

    pending_calibration_key =None
    calibration_samples =[]
    monitor_samples =None

    calibration_target ="center"


#get current calibration target
def get_calibration_target():
    return calibration_target


#get calibration sample progress
def get_calibration_progress():
    if pending_calibration_key is None:
        return 0

    return len(calibration_samples)