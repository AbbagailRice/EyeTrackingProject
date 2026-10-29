#this file handles face tracking and head position using mediapipe and open cv
#it creates the local face coordinate system and calculates the head pose axes

import cv2
import mediapipe as mp
import numpy as np

from config import *


#create face landmarker
def create_face_landmarker():

    #setup mediapipe face landmarker
    base_options =mp.tasks.BaseOptions(
        model_asset_path =MODEL_PATH
    )

    options =mp.tasks.vision.FaceLandmarkerOptions(
        base_options =base_options,
        running_mode =mp.tasks.vision.RunningMode.VIDEO,
        num_faces =1
    )

    landmarker =mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )

    return landmarker


#convert mediapipe landmark into 3d pixel space
def landmark_to_3d(landmark,width,height):

    return np.array([
        landmark.x*width,
        landmark.y*height,
        landmark.z*width
    ],dtype =np.float64)

#calculate scale from stable nose landmarks
def calculate_face_scale(
    face_landmarks,
    width,
    height
):

    #convert nose landmarks into 3d points
    nose_points =np.array([
        landmark_to_3d(
            face_landmarks[index],
            width,
            height
        )
        for index in NOSE_INDICES
    ])

    #calculate average distance between points
    total_distance =0.0
    count =0

    for i in range(len(nose_points)):

        for j in range(i+1,len(nose_points)):

            distance =np.linalg.norm(
                nose_points[i]-
                nose_points[j]
            )

            total_distance +=distance
            count +=1

    #avoid division by zero
    if count ==0:
        return 1.0

    return total_distance/count


#create local coordinate system from stable face landmarks
def create_face_coordinate_system(
    face_landmarks,
    width,
    height
):

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
    z_axis =np.cross(
        x_axis,
        y_temp
    )

    z_axis =z_axis/np.linalg.norm(z_axis)

    #rebuild vertical axis
    y_axis =np.cross(
        z_axis,
        x_axis
    )

    y_axis =y_axis/np.linalg.norm(y_axis)

    #combine axes into rotation matrix
    face_basis =np.column_stack((
        x_axis,
        y_axis,
        z_axis
    ))

    return nose,face_basis


#calculate head pose for debug display
def calculate_head_pose(
    face_landmarks,
    width,
    height,
    camera_matrix,
    distortion_coefficients
):

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
        FACE_MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags =cv2.SOLVEPNP_ITERATIVE
    )

    #return nothing if head pose failed
    if not success:
        return None

    #project 3d axis points onto webcam
    projected_axis,_ =cv2.projectPoints(
        AXIS_POINTS,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion_coefficients
    )

    #get nose point
    nose_point =(
        int(image_points[0][0]),
        int(image_points[0][1])
    )

    #get projected axis points
    x_axis =tuple(
        projected_axis[0].ravel().astype(int)
    )

    y_axis =tuple(
        projected_axis[1].ravel().astype(int)
    )

    z_axis =tuple(
        projected_axis[2].ravel().astype(int)
    )

    return {
        "nose_point": nose_point,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis
    }