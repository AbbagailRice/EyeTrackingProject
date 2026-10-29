#this file opens the webcam and tracks the users eyes and irises using mediapipe and open cv
#it creates a face coordinate system using stable facial landmarks
#it draws x y and z axes that move with the users head

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

#head pose landmarks
NOSE_TIP =1
CHIN =152
LEFT_EYE_OUTER =33
RIGHT_EYE_OUTER =263
LEFT_MOUTH =61
RIGHT_MOUTH =291

#smoothing settings
SMOOTHING_FRAMES =5

#webcam display size
CAM_WIDTH =500
CAM_HEIGHT =350

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

#default gaze values
smooth_horizontal =0.0
smooth_vertical =0.0
quadrant =0

#3d reference face points
#should approx the shape of a normal human face
face_model_points =np.array([
    (0.0,0.0,0.0),
    (0.0,-63.6,-12.5),
    (-43.3,32.7,-26.0),
    (43.3,32.7,-26.0),
    (-28.9,-28.9,-24.1),
    (28.9,-28.9,-24.1)
],dtype =np.float64)


#gaze ray length
GAZE_RAY_LENGTH =120.0

#calibrated eye centers in local face coordinates
left_eye_center_local =None
right_eye_center_local =None

#current mediapipe face coordinate system
face_origin =None
face_basis =None

#store current pose information
pose_ready =False
current_left_eye_pixel =None
current_right_eye_pixel =None
rotation_matrix =None
translation_vector =None

#store current iris pixels
current_left_iris_pixel =None
current_right_iris_pixel =None

#3d coordinate axis points
axis_points =np.array([
    (50.0,0.0,0.0),
    (0.0,50.0,0.0),
    (0.0,0.0,50.0)
],dtype =np.float64)

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


#convert mediapipe landmark into 3d pixel space
def landmark_to_3d(landmark,width,height):
    return np.array([
        landmark.x*width,
        landmark.y*height,
        landmark.z*width
    ],dtype =np.float64)


#create local coordinate system from stable face landmarks
def create_face_coordinate_system(face_landmarks,width,height):

    #get stable face points
    nose =landmark_to_3d(
        face_landmarks[NOSE_TIP],
        width,
        height
    )

    left_outer =landmark_to_3d(
        face_landmarks[LEFT_EYE_OUTER],
        width,
        height
    )

    right_outer =landmark_to_3d(
        face_landmarks[RIGHT_EYE_OUTER],
        width,
        height
    )

    chin =landmark_to_3d(
        face_landmarks[CHIN],
        width,
        height
    )

    #create horizontal face axis
    x_axis =right_outer-left_outer
    x_axis =x_axis/np.linalg.norm(x_axis)

    #temporary vertical direction
    y_temp =chin-nose
    y_temp =y_temp/np.linalg.norm(y_temp)

    #create forward face axis
    z_axis =np.cross(x_axis,y_temp)
    z_axis =z_axis/np.linalg.norm(z_axis)

    #rebuild vertical axis
    y_axis =np.cross(z_axis,x_axis)
    y_axis =y_axis/np.linalg.norm(y_axis)

    #combine axes into rotation matrix
    basis =np.column_stack((
        x_axis,
        y_axis,
        z_axis
    ))

    return nose,basis

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
    focal_length =width

    camera_matrix =np.array([
        [focal_length,0,width/2],
        [0,focal_length,height/2],
        [0,0,1]
    ],dtype =np.float64)

    #assume no lens distortion
    distortion_coefficients =np.zeros((4,1))

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

     #calculate gaze and head position
    if result.face_landmarks:
    
        face_landmarks =result.face_landmarks[0]

        #create mediapipe face coordinate system
        face_origin,face_basis =create_face_coordinate_system(
            face_landmarks,
            width,
            height
        )

        #get iris centers
        left_iris =face_landmarks[LEFT_IRIS[0]]
        right_iris =face_landmarks[RIGHT_IRIS[0]]

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

        #calculate observed left eye center
        left_eye_center_x =(
            left_eye_left.x+
            left_eye_right.x+
            left_eye_top.x+
            left_eye_bottom.x
        )/4

        left_eye_center_y =(
            left_eye_left.y+
            left_eye_right.y+
            left_eye_top.y+
            left_eye_bottom.y
        )/4

        #calculate observed right eye center
        right_eye_center_x =(
            right_eye_left.x+
            right_eye_right.x+
            right_eye_top.x+
            right_eye_bottom.x
        )/4

        right_eye_center_y =(
            right_eye_left.y+
            right_eye_right.y+
            right_eye_top.y+
            right_eye_bottom.y
        )/4

        #calculate observed eye centers in 3d
        left_eye_observed_3d =(
            landmark_to_3d(left_eye_left,width,height)+
            landmark_to_3d(left_eye_right,width,height)+
            landmark_to_3d(left_eye_top,width,height)+
            landmark_to_3d(left_eye_bottom,width,height)
        )/4

        right_eye_observed_3d =(
            landmark_to_3d(right_eye_left,width,height)+
            landmark_to_3d(right_eye_right,width,height)+
            landmark_to_3d(right_eye_top,width,height)+
            landmark_to_3d(right_eye_bottom,width,height)
        )/4

        #reconstruct calibrated eye centers
        if left_eye_center_local is not None and right_eye_center_local is not None:

            left_eye_center_3d =face_origin+(
                face_basis@left_eye_center_local
            )

            right_eye_center_3d =face_origin+(
                face_basis@right_eye_center_local
            )

            #convert iris positions into local face coordinates
            left_iris_local =face_basis.T@(
                left_iris_3d-face_origin
            )

            right_iris_local =face_basis.T@(
                right_iris_3d-face_origin
            )

            #calculate gaze directions relative to face
            left_gaze_local =left_iris_local-left_eye_center_local
            right_gaze_local =right_iris_local-right_eye_center_local

            #normalize gaze directions
            left_gaze_local =left_gaze_local/np.linalg.norm(left_gaze_local)
            right_gaze_local =right_gaze_local/np.linalg.norm(right_gaze_local)

            #convert gaze directions back into camera coordinates
            left_gaze_3d =face_basis@left_gaze_local
            right_gaze_3d =face_basis@right_gaze_local

            #convert observed centers to pixels
            left_eye_center_2d =(
                int(left_eye_center_3d[0]),
                int(left_eye_center_3d[1])
            )

            right_eye_center_2d =(
                int(right_eye_center_3d[0]),
                int(right_eye_center_3d[1])
            )

            #ray display length
            ray_length =100

            #calculate gaze endpoints
            left_ray_end =(
                int(left_eye_center_3d[0]+left_gaze_3d[0]*ray_length),
                int(left_eye_center_3d[1]+left_gaze_3d[1]*ray_length)
            )

            right_ray_end =(
                int(right_eye_center_3d[0]+right_gaze_3d[0]*ray_length),
                int(right_eye_center_3d[1]+right_gaze_3d[1]*ray_length)
            )

            #draw eye circles
            cv2.circle(
                frame,
                left_eye_center_2d,
                14,
                (255,255,0),
                2
            )

            cv2.circle(
                frame,
                right_eye_center_2d,
                14,
                (0,255,255),
                2
            )

            #draw gaze rays
            cv2.line(
                frame,
                left_eye_center_2d,
                left_ray_end,
                (255,255,0),
                3
            )

            cv2.line(
                frame,
                right_eye_center_2d,
                right_ray_end,
                (0,255,255),
                3
            )

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

        #get head pose image points
        image_points =np.array([
            (
                face_landmarks[NOSE_TIP].x*width,
                face_landmarks[NOSE_TIP].y*height
            ),
            (
                face_landmarks[CHIN].x*width,
                face_landmarks[CHIN].y*height
            ),
            (
                face_landmarks[LEFT_EYE_OUTER].x*width,
                face_landmarks[LEFT_EYE_OUTER].y*height
            ),
            (
                face_landmarks[RIGHT_EYE_OUTER].x*width,
                face_landmarks[RIGHT_EYE_OUTER].y*height
            ),
            (
                face_landmarks[LEFT_MOUTH].x*width,
                face_landmarks[LEFT_MOUTH].y*height
            ),
            (
                face_landmarks[RIGHT_MOUTH].x*width,
                face_landmarks[RIGHT_MOUTH].y*height
            )
        ],dtype =np.float64)

        #estimate head rotation and position
        success,rotation_vector,translation_vector =cv2.solvePnP(
            face_model_points,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags =cv2.SOLVEPNP_ITERATIVE
        )

        #draw head coordinate system
        if success:
            #convert rotation vector into rotation matrix
            rotation_matrix,_ =cv2.Rodrigues(rotation_vector)

            #head pose is ready
            pose_ready =True

            #project 3d axis points onto webcam
            projected_axis,_ =cv2.projectPoints(
                axis_points,
                rotation_vector,
                translation_vector,
                camera_matrix,
                distortion_coefficients
            )

            #get nose point
            nose_x =int(image_points[0][0])
            nose_y =int(image_points[0][1])

            nose_point =(nose_x,nose_y)

            #get projected axis points
            x_axis =tuple(projected_axis[0].ravel().astype(int))
            y_axis =tuple(projected_axis[1].ravel().astype(int))
            z_axis =tuple(projected_axis[2].ravel().astype(int))

            #draw x axis
            cv2.line(
                frame,
                nose_point,
                x_axis,
                (0,0,255),
                3
            )

            #draw y axis
            cv2.line(
                frame,
                nose_point,
                y_axis,
                (0,255,0),
                3
            )

            #draw z axis
            cv2.line(
                frame,
                nose_point,
                z_axis,
                (255,0,0),
                3
            )

            #label axes
            cv2.putText(
                frame,
                "x",
                x_axis,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

            cv2.putText(
                frame,
                "y",
                y_axis,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                "z",
                z_axis,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2
            )

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

    #calibrate eye centers
    if key ==ord("c"):

        if face_origin is not None and face_basis is not None:

            #convert observed eye centers into local face coordinates
            left_eye_center_local =face_basis.T@(
                left_eye_observed_3d-face_origin
            )

            right_eye_center_local =face_basis.T@(
                right_eye_observed_3d-face_origin
            )

            print("\neye centers calibrated")
            print("left eye:",left_eye_center_local)
            print("right eye:",right_eye_center_local)

    #save calibration point
    elif key ==ord("1"):
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