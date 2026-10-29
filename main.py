#this file runs the eye tracking project and connects the project modules
#it opens the webcam tracks the face calculates gaze handles calibration and displays the results
#press q to close the webcam and end the program

import cv2
import time

import camera
import face_tracking
import gaze_tracking
import calibration
import display
import monitor_tracking

#get screen size
SCREEN_WIDTH,SCREEN_HEIGHT =display.get_screen_size()


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

    #calculate gaze and head position
    if result.face_landmarks:

        face_landmarks =result.face_landmarks[0]
        
        #create mediapipe face coordinate system
        face_origin,face_basis =face_tracking.create_face_coordinate_system(
            face_landmarks,
            width,
            height
        )

        #calculate current face scale
        current_face_scale =face_tracking.calculate_face_scale(
            face_landmarks,
            width,
            height
        )

        #get eye and iris information
        eye_data =gaze_tracking.get_eye_data(
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

    #create main display
    main_display =display.create_display(
        frame,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        gaze_data,
        screen_gaze,
        monitor_tracking.is_monitor_ready()
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

            calibration.calibrate_eye_centers(
                face_origin,
                face_basis,
                eye_data["left_iris_3d"],
                eye_data["right_iris_3d"],
                current_face_scale
            )

    #save monitor top-left
    elif key ==ord("1"):

        if gaze_data is not None:

            monitor_tracking.save_calibration_point(
                "1",
                gaze_data["combined_eye_center_3d"],
                gaze_data["combined_gaze_3d"]
            )

    #save monitor top-right
    elif key ==ord("2"):

        if gaze_data is not None:

            monitor_tracking.save_calibration_point(
                "2",
                gaze_data["combined_eye_center_3d"],
                gaze_data["combined_gaze_3d"]
            )

    #save monitor bottom-left
    elif key ==ord("3"):

        if gaze_data is not None:

            monitor_tracking.save_calibration_point(
                "3",
                gaze_data["combined_eye_center_3d"],
                gaze_data["combined_gaze_3d"]
            )

    #save monitor bottom-right
    elif key ==ord("4"):

        if gaze_data is not None:

            monitor_tracking.save_calibration_point(
                "4",
                gaze_data["combined_eye_center_3d"],
                gaze_data["combined_gaze_3d"]
            )

    #calibrate virtual monitor
    elif key ==ord("m"):

        if (
            gaze_data is not None and
            face_landmarks is not None
        ):

            monitor_tracking.calibrate_monitor(
                gaze_data["combined_eye_center_3d"],
                gaze_data["combined_gaze_3d"],
                face_landmarks,
                width,
                height
            )

    #press q to quit
    elif key ==ord("q"):
        break


#release webcam
cap.release()

#close mediapipe
landmarker.close()

#close all windows
cv2.destroyAllWindows()