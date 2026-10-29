#this file opens the webcam and tracks the users eyes and irises using mediapipe
#it calculates the iris position relative to each eye
#it prints horizontal and vertical gaze ratios for testing
#press q to close the webcam and end the program

import cv2
import mediapipe as mp
import time #this is because mp requires a timestamp in ms for video processing (VIDEO MODE)
import numpy as np
import tkinter as tk


#left iris landmarks
LEFT_IRIS =[468,469,470,471,472]

#right iris landmarks
RIGHT_IRIS =[473,474,475,476,477]

#left eye reference landmarks
LEFT_EYE_LEFT =33
LEFT_EYE_RIGHT =133
LEFT_EYE_TOP =159
LEFT_EYE_BOTTOM =145

#right eye reference landmarks
RIGHT_EYE_LEFT =362
RIGHT_EYE_RIGHT =263
RIGHT_EYE_TOP =386
RIGHT_EYE_BOTTOM =374

#smoothing settings
SMOOTHING_FRAMES =5

#webcam display size
CAM_WIDTH =300
CAM_HEIGHT =220

#get screen size
root =tk.Tk()
SCREEN_WIDTH =root.winfo_screenwidth()
SCREEN_HEIGHT =root.winfo_screenheight()
root.destroy()

#store calibration points
calibration_points ={}

#store recent gaze values
horizontal_values =[]
vertical_values =[]

#model file path
model_path ="face_landmarker.task"

#setup mediapipe face landmarker
base_options =mp.tasks.BaseOptions(model_asset_path =model_path)

#other
options =mp.tasks.vision.FaceLandmarkerOptions(
    base_options =base_options,
    running_mode =mp.tasks.vision.RunningMode.VIDEO,
    num_faces =1
)

#create face landmarker
landmarker =mp.tasks.vision.FaceLandmarker.create_from_options(options)

#open webcam
cap =cv2.VideoCapture(0)

#set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

#check if webcam opened
if not cap.isOpened():
    print("could not open webcam")
    exit()

#TIMESTAMP
start_time =time.time()

#create fullscreen window
cv2.namedWindow("eye tracking project",cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "eye tracking project",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

#keep webcam running
while True:
    ret, frame =cap.read()

    #check if frame was read
    if not ret:
        print("could not read frame")
        break
    
    #flip webcam horizontally
    frame =cv2.flip(frame,1)

    #convert frame from bgr to rgb
    rgb_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    #convert frame into mediapipe image
    mp_image =mp.Image(
        image_format =mp.ImageFormat.SRGB,
        data =rgb_frame
    )

    #create timestamp in ms
    timestamp_ms =int((time.time()-start_time)*1000)

    #detect facial landmarks
    result =landmarker.detect_for_video(mp_image,timestamp_ms)

     #calculate gaze position
    if result.face_landmarks:
        height,width,_ =frame.shape
        face_landmarks =result.face_landmarks[0]

        #get iris centers
        left_iris =face_landmarks[LEFT_IRIS[0]]
        right_iris =face_landmarks[RIGHT_IRIS[0]]

        #get left eye landmarks
        left_eye_left =face_landmarks[LEFT_EYE_LEFT]
        left_eye_right =face_landmarks[LEFT_EYE_RIGHT]
        left_eye_top =face_landmarks[LEFT_EYE_TOP]
        left_eye_bottom =face_landmarks[LEFT_EYE_BOTTOM]

        #get right eye landmarks
        right_eye_left =face_landmarks[RIGHT_EYE_LEFT]
        right_eye_right =face_landmarks[RIGHT_EYE_RIGHT]
        right_eye_top =face_landmarks[RIGHT_EYE_TOP]
        right_eye_bottom =face_landmarks[RIGHT_EYE_BOTTOM]

        #calculate left eye ratios
        left_horizontal =(left_iris.x-left_eye_left.x)/(left_eye_right.x-left_eye_left.x)
        left_vertical =(left_iris.y-left_eye_top.y)/(left_eye_bottom.y-left_eye_top.y)

        #calculate right eye ratios
        right_horizontal =(right_iris.x-right_eye_left.x)/(right_eye_right.x-right_eye_left.x)
        right_vertical =(right_iris.y-right_eye_top.y)/(right_eye_bottom.y-right_eye_top.y)

        #average both eyes
        horizontal_ratio =(left_horizontal+right_horizontal)/2
        vertical_ratio =(left_vertical+right_vertical)/2

        #add current values to smoothing lists
        horizontal_values.append(horizontal_ratio)
        vertical_values.append(vertical_ratio)

        #keep only recent values
        if len(horizontal_values)>SMOOTHING_FRAMES:
            horizontal_values.pop(0)

        if len(vertical_values)>SMOOTHING_FRAMES:
            vertical_values.pop(0)

        #average recent values
        smooth_horizontal =sum(horizontal_values)/len(horizontal_values)
        smooth_vertical =sum(vertical_values)/len(vertical_values)

        #calculate quadrant from calibration
        quadrant =0

        if len(calibration_points) ==4:

            #get saved calibration points
            point_1 =calibration_points[1]
            point_2 =calibration_points[2]
            point_3 =calibration_points[3]
            point_4 =calibration_points[4]

            #average left and right horizontal values
            left_horizontal =(point_1[0]+point_3[0])/2
            right_horizontal =(point_2[0]+point_4[0])/2

            #find horizontal middle
            horizontal_middle =(left_horizontal+right_horizontal)/2

            #find vertical middle for left side
            left_vertical_middle =(point_1[1]+point_3[1])/2

            #find vertical middle for right side
            right_vertical_middle =(point_2[1]+point_4[1])/2

            #determine left side
            if smooth_horizontal <horizontal_middle:

                #determine quadrant 1 or 3
                if smooth_vertical >left_vertical_middle:
                    quadrant =1
                else:
                    quadrant =3

            #determine right side
            else:

                #determine quadrant 2 or 4
                if smooth_vertical >right_vertical_middle:
                    quadrant =2
                else:
                    quadrant =4


        #draw iris center points
        left_x =int(left_iris.x*width)
        left_y =int(left_iris.y*height)

        right_x =int(right_iris.x*width)
        right_y =int(right_iris.y*height)

        cv2.circle(frame,(left_x,left_y),3,(255,255,255),-1)
        cv2.circle(frame,(right_x,right_y),3,(255,255,255),-1)

    #create main display
    display =np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3),dtype =np.uint8)

    #find center of display
    center_x =SCREEN_WIDTH//2
    center_y =SCREEN_HEIGHT//2

    #draw quadrant colors
    cv2.rectangle(display,(0,0),(center_x,center_y),(0,0,255),-1)
    cv2.rectangle(display,(center_x,0),(SCREEN_WIDTH,center_y),(0,255,0),-1)
    cv2.rectangle(display,(0,center_y),(center_x,SCREEN_HEIGHT),(255,0,0),-1)
    cv2.rectangle(display,(center_x,center_y),(SCREEN_WIDTH,SCREEN_HEIGHT),(0,255,255),-1)

    #show horizontal value
    cv2.putText(
        display,
        f"horizontal: {smooth_horizontal:.3f}",
        (20,SCREEN_HEIGHT-70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    #show vertical value
    cv2.putText(
        display,
        f"vertical: {smooth_vertical:.3f}",
        (20,SCREEN_HEIGHT-30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    #show calibration middle values
    if len(calibration_points) ==4:
        cv2.putText(
            display,
            f"h middle: {horizontal_middle:.3f}",
            (20,SCREEN_HEIGHT-190),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        cv2.putText(
            display,
            f"left v middle: {left_vertical_middle:.3f}",
            (20,SCREEN_HEIGHT-150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        cv2.putText(
            display,
            f"right v middle: {right_vertical_middle:.3f}",
            (20,SCREEN_HEIGHT-110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

    #draw quadrant borders
    cv2.rectangle(display,(0,0),(center_x,center_y),(255,255,255),3)
    cv2.rectangle(display,(center_x,0),(SCREEN_WIDTH-1,center_y),(255,255,255),3)
    cv2.rectangle(display,(0,center_y),(center_x,SCREEN_HEIGHT-1),(255,255,255),3)
    cv2.rectangle(display,(center_x,center_y),(SCREEN_WIDTH-1,SCREEN_HEIGHT-1),(255,255,255),3)

    #set default number thickness
    thickness_1 =4
    thickness_2 =4
    thickness_3 =4
    thickness_4 =4

    #set default number sizeq
    size_1 =2
    size_2 =2
    size_3 =2
    size_4 =2

    #make selected quadrant number bold
    if quadrant ==1:
        thickness_1 =10
        size_1 =3
    elif quadrant ==2:
        thickness_2 =10
        size_2 =3
    elif quadrant ==3:
        thickness_3 =10
        size_3 =3
    elif quadrant ==4:
        thickness_4 =10
        size_4 =3

    #draw quadrant numbers
    cv2.putText(
        display,
        "1",
        (center_x//2-20,center_y//2),
        cv2.FONT_HERSHEY_SIMPLEX,
        size_1,
        (255,255,255),
        thickness_1
    )

    cv2.putText(
        display,
        "2",
        (center_x+center_x//2-20,center_y//2),
        cv2.FONT_HERSHEY_SIMPLEX,
        size_2,
        (255,255,255),
        thickness_2
    )

    cv2.putText(
        display,
        "3",
        (center_x//2-20,center_y+center_y//2),
        cv2.FONT_HERSHEY_SIMPLEX,
        size_3,
        (255,255,255),
        thickness_3
    )

    cv2.putText(
        display,
        "4",
        (center_x+center_x//2-20,center_y+center_y//2),
        cv2.FONT_HERSHEY_SIMPLEX,
        size_4,
        (0,0,0),
        thickness_4
    )

    #resize webcam
    small_frame =cv2.resize(frame,(CAM_WIDTH,CAM_HEIGHT))

    #place webcam in top left
    display[0:CAM_HEIGHT,0:CAM_WIDTH] =small_frame

    #show display
    cv2.imshow("eye tracking project",display)

    #check keyboard input
    key =cv2.waitKey(1)&0xFF

    #save calibration point
    if key ==ord("1"):
        calibration_points[1] =(smooth_horizontal,smooth_vertical)
        print("saved quadrant 1:",calibration_points[1])

    elif key ==ord("2"):
        calibration_points[2] =(smooth_horizontal,smooth_vertical)
        print("saved quadrant 2:",calibration_points[2])

    elif key ==ord("3"):
        calibration_points[3] =(smooth_horizontal,smooth_vertical)
        print("saved quadrant 3:",calibration_points[3])

    elif key ==ord("4"):
        calibration_points[4] =(smooth_horizontal,smooth_vertical)
        print("saved quadrant 4:",calibration_points[4])

    #press q to quit
    elif key ==ord("q"):
        break


#release webcam
cap.release()

#close mediapipe
landmarker.close()

#close all windows
cv2.destroyAllWindows()