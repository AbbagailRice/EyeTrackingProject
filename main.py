#this file runs the eye tracking project and connects the project modules
#it opens the webcam tracks the face calculates gaze handles calibration and displays the results
#press q to close the webcam and end the program

#keys:
#  c: calibrate eye centers (look at the camera lens, hold still)
#  v: calibrate virtual monitor (look at the screen center, hold still)
#  1: measure the center point (after v), then press 2 3 4 5 6 7 8 9 in order as each target appears
#  r: rebuild the reference face shape
#  q: quit

import cv2
import time

import camera
import face_tracking
import gaze_tracking
import calibration
import display
import monitor_tracking
import keyboard
import blink

from config import EYE_CALIBRATION_FRAMES, MONITOR_CALIBRATION_FRAMES, FACE_REFERENCE_FRAMES

#get screen size
SCREEN_WIDTH,SCREEN_HEIGHT =display.get_screen_size()

#build keyboard layout
keyboard_keys =keyboard.create_keyboard(
    SCREEN_WIDTH,
    SCREEN_HEIGHT
)

#create face landmarker
landmarker =face_tracking.create_face_landmarker()

#open webcam
cap =camera.open_camera()

if cap is None:
    exit()

#timestamp
start_time =time.time()

#create fullscreen window
display.setup_window()

#keep webcam running
while True:
    ret, frame =cap.read()

    #check if frame was read
    if not ret:
        print("could not read frame")
        break
    
    #flip webcam horizontally
    frame =cv2.flip(frame,1)

    #get frame size
    height,width,_ =frame.shape
    
    #create camera matrix
    camera_matrix =camera.create_camera_matrix(
        width,
        height
    )

    #assume no lens distortion
    distortion_coefficients =camera.create_distortion_coefficients()

    #convert frame into mediapipe image
    mp_image =camera.frame_to_mediapipe(frame)

    #create timestamp in ms
    timestamp_ms =camera.create_timestamp(start_time)

    #detect facial landmarks
    result =landmarker.detect_for_video(mp_image,timestamp_ms)
    #default frame data
    eye_data =None
    gaze_data =None
    head_pose =None
    face_origin =None
    face_basis =None
    face_landmarks =None
    current_face_scale =None
    eyes_are_open =False

    gaze_key =None

    #calculate gaze and head position
    if result.face_landmarks:

        face_landmarks =result.face_landmarks[0]
        
        #fit the face coordinate system to the rigid landmarks
        face_origin,face_basis,current_face_scale =face_tracking.create_face_coordinate_system(
            face_landmarks,
            width,
            height
        )

        if face_basis is not None:

            #get eye and iris information
            eye_data =gaze_tracking.get_eye_data(
                face_landmarks,
                width,
                height
            )

            #check for blinks
            eyes_are_open =blink.eyes_open(
                face_landmarks,
                width,
                height
            )

            #get calibrated eye sphere information
            (left_eye_center_local,
                right_eye_center_local,
                calibration_face_scale
            ) =calibration.get_eye_centers()

            #calculate gaze
            gaze_data =gaze_tracking.calculate_gaze(
                face_origin,
                face_basis,
                eye_data,
                left_eye_center_local,
                right_eye_center_local,
                calibration_face_scale,
                current_face_scale
            )

            #add this frame to eye calibration if it is running
            eye_calibration_done =calibration.update_eye_calibration(
                face_origin,
                face_basis,
                eye_data["left_iris_3d"],
                eye_data["right_iris_3d"],
                current_face_scale,
                eyes_are_open
            )

            #clear gaze history if eye calibration just finished
            if eye_calibration_done:
                gaze_tracking.combined_gaze_directions.clear()

            #add this frame to monitor calibration if it is running
            monitor_tracking.update_monitor_calibration(
                gaze_data,
                face_landmarks,
                width,
                height,
                eyes_are_open
            )

            #calculate head pose
            head_pose =face_tracking.calculate_head_pose(
                face_landmarks,
                width,
                height,
                camera_matrix,
                distortion_coefficients
            )

            #draw tracking on webcam
            display.draw_tracking(
                frame,
                eye_data,
                gaze_data,
                head_pose
            )

    #default mapped gaze
    screen_gaze =None

    #map gaze ray to virtual monitor
    if gaze_data is not None:

        screen_gaze =monitor_tracking.map_gaze_to_screen(
            gaze_data["combined_eye_center_3d"],
            gaze_data["combined_gaze_3d"],
            SCREEN_WIDTH,
            SCREEN_HEIGHT
        )
    
    #find key under gaze
    gaze_key =keyboard.get_gaze_key(
        keyboard_keys,
        screen_gaze
    )

    #create main display
    main_display =display.create_display(
        frame,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        gaze_data,
        screen_gaze,
        monitor_tracking.is_monitor_ready(),
        monitor_tracking.get_calibration_target(),
        monitor_tracking.get_calibration_progress(),
        keyboard_keys,
        gaze_key
    )

    #show display
    display.show_display(
        main_display
    )

    #check keyboard input
    key =cv2.waitKey(1)&0xFF

    #calibrate eye centers
    if key ==ord("c"):

        if (
            face_origin is not None and
            face_basis is not None and
            eye_data is not None
        ):

            calibration.start_eye_calibration()

    #start screen calibration point
    elif key in [
        ord("1"),
        ord("2"),
        ord("3"),
        ord("4"),
        ord("5"),
        ord("6"),
        ord("7"),
        ord("8"),
        ord("9")
    ]:

        if gaze_data is not None:

            calibration_key =chr(
                key
            )

            monitor_tracking.start_calibration_point(
                calibration_key
            )

    #calibrate virtual monitor
    elif key ==ord("v"):

        if gaze_data is not None:

            monitor_tracking.start_monitor_calibration()

    #rebuild reference face shape
    elif key ==ord("r"):
 
        face_tracking.reset_face_reference()

    #press q to quit
    elif key ==ord("q"):
        break


#release webcam
cap.release()

#close mediapipe
landmarker.close()

#close all windows
cv2.destroyAllWindows()