#this file creates the project display and draws tracking information
#it draws the quadrants webcam preview head axes eye circles gaze rays and gaze position

import cv2
import numpy as np
import tkinter as tk

from config import CAM_WIDTH,CAM_HEIGHT


WINDOW_NAME ="eye tracking project"


#get screen size
def get_screen_size():

    root =tk.Tk()

    screen_width =root.winfo_screenwidth()
    screen_height =root.winfo_screenheight()

    root.destroy()

    return screen_width,screen_height


#create fullscreen window
def setup_window():

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )


#draw tracking information on webcam
def draw_tracking(
    frame,
    eye_data,
    gaze_data,
    head_pose
):

    #draw iris center points
    if eye_data is not None:

        height,width,_ =frame.shape

        left_iris =eye_data["left_iris"]
        right_iris =eye_data["right_iris"]

        left_x =int(
            left_iris.x*width
        )

        left_y =int(
            left_iris.y*height
        )

        right_x =int(
            right_iris.x*width
        )

        right_y =int(
            right_iris.y*height
        )

        cv2.circle(
            frame,
            (left_x,left_y),
            3,
            (255,255,255),
            -1
        )

        cv2.circle(
            frame,
            (right_x,right_y),
            3,
            (255,255,255),
            -1
        )

    #draw gaze information
    if gaze_data is not None:

        #draw calibrated left eye center as circle
        cv2.circle(
            frame,
            gaze_data["left_eye_center_2d"],
            14,
            (255,255,0),
            2
        )

        #draw calibrated right eye center as circle
        cv2.circle(
            frame,
            gaze_data["right_eye_center_2d"],
            14,
            (0,255,255),
            2
        )

        #draw left gaze ray
        cv2.line(
            frame,
            gaze_data["left_eye_center_2d"],
            gaze_data["left_ray_end"],
            (255,255,0),
            3
        )

        #draw right gaze ray
        cv2.line(
            frame,
            gaze_data["right_eye_center_2d"],
            gaze_data["right_ray_end"],
            (0,255,255),
            3
        )

        #draw combined gaze ray
        cv2.line(
            frame,
            gaze_data["combined_eye_center_2d"],
            gaze_data["combined_ray_end"],
            (255,255,255),
            4
        )

    #draw head coordinate system
    if head_pose is not None:

        #draw x axis
        cv2.line(
            frame,
            head_pose["nose_point"],
            head_pose["x_axis"],
            (0,0,255),
            3
        )

        #draw y axis
        cv2.line(
            frame,
            head_pose["nose_point"],
            head_pose["y_axis"],
            (0,255,0),
            3
        )

        #draw z axis
        cv2.line(
            frame,
            head_pose["nose_point"],
            head_pose["z_axis"],
            (255,0,0),
            3
        )

        #label axes
        cv2.putText(
            frame,
            "x",
            head_pose["x_axis"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

        cv2.putText(
            frame,
            "y",
            head_pose["y_axis"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            "z",
            head_pose["z_axis"],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )


#create main display
def create_display(
    frame,
    screen_width,
    screen_height,
    gaze_data,
    screen_gaze,
    monitor_ready
):

    display =np.zeros(
        (
            screen_height,
            screen_width,
            3
        ),
        dtype =np.uint8
    )

    #find center of display
    center_x =screen_width//2
    center_y =screen_height//2

    #draw quadrant colors
    cv2.rectangle(
        display,
        (0,0),
        (center_x,center_y),
        (0,0,255),
        -1
    )

    cv2.rectangle(
        display,
        (center_x,0),
        (screen_width,center_y),
        (0,255,0),
        -1
    )

    cv2.rectangle(
        display,
        (0,center_y),
        (center_x,screen_height),
        (255,0,0),
        -1
    )

    cv2.rectangle(
        display,
        (center_x,center_y),
        (screen_width,screen_height),
        (0,255,255),
        -1
    )

    #draw quadrant borders
    cv2.rectangle(
        display,
        (0,0),
        (center_x,center_y),
        (255,255,255),
        3
    )

    cv2.rectangle(
        display,
        (center_x,0),
        (screen_width-1,center_y),
        (255,255,255),
        3
    )

    cv2.rectangle(
        display,
        (0,center_y),
        (center_x,screen_height-1),
        (255,255,255),
        3
    )

    cv2.rectangle(
        display,
        (center_x,center_y),
        (screen_width-1,screen_height-1),
        (255,255,255),
        3
    )

    #draw quadrant numbers
    cv2.putText(
        display,
        "1",
        (center_x//2-20,center_y//2),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255,255,255),
        4
    )

    cv2.putText(
        display,
        "2",
        (
            center_x+center_x//2-20,
            center_y//2
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255,255,255),
        4
    )

    cv2.putText(
        display,
        "3",
        (
            center_x//2-20,
            center_y+center_y//2
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255,255,255),
        4
    )

    cv2.putText(
        display,
        "4",
        (
            center_x+center_x//2-20,
            center_y+center_y//2
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0,0,0),
        4
    )

    #show gaze angles
    if gaze_data is not None:

        cv2.putText(
            display,
            f"yaw: {gaze_data['smooth_yaw']:.2f}",
            (20,screen_height-70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        cv2.putText(
            display,
            f"pitch: {gaze_data['smooth_pitch']:.2f}",
            (20,screen_height-30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

    #show monitor calibration status
    if monitor_ready:

        cv2.putText(
            display,
            "monitor calibrated",
            (20,screen_height-110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

    #resize webcam
    small_frame =cv2.resize(
        frame,
        (
            CAM_WIDTH,
            CAM_HEIGHT
        )
    )

    #place webcam in top left
    display[
        0:CAM_HEIGHT,
        0:CAM_WIDTH
    ] =small_frame

    #draw mapped gaze position
    if screen_gaze is not None:

        cv2.circle(
            display,
            screen_gaze,
            20,
            (255,255,255),
            4
        )

    return display


#show display
def show_display(display):

    cv2.imshow(
        WINDOW_NAME,
        display
    )