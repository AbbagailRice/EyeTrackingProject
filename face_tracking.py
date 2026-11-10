#this file handles face tracking and head position using mediapipe and open cv
#it creates the local face coordinate system and calculates the head pose axes

import cv2
import mediapipe as mp
import numpy as np

from config import *

#reference face basis used to prevent pca axis flipping
face_basis_reference =None

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

#get stable nose points in 3d
def get_nose_points(
    face_landmarks,
    width,
    height
):

    nose_points =np.array([
        landmark_to_3d(
            face_landmarks[index],
            width,
            height
        )
        for index in NOSE_INDICES
    ],dtype =np.float64)

    return nose_points

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


#create stable local coordinate system from nose landmarks
def create_face_coordinate_system(
    face_landmarks,
    width,
    height
):

    global face_basis_reference

    #get stable nose region landmarks
    nose_points =get_nose_points(
        face_landmarks,
        width,
        height
    )

    #use average nose position as face origin
    face_origin =np.mean(
        nose_points,
        axis =0
    )

    #center nose points around origin
    centered_points =(
        nose_points-
        face_origin
    )

    #calculate covariance matrix
    covariance =np.cov(
        centered_points.T
    )

    #calculate pca axes
    eigenvalues,eigenvectors =np.linalg.eigh(
        covariance
    )

    #sort largest eigenvalue first
    order =np.argsort(
        eigenvalues
    )[::-1]

    face_basis =eigenvectors[
        :,
        order
    ]

    #make coordinate system right handed
    if np.linalg.det(face_basis)<0:
        face_basis[:,2] *=-1

    #save first stable orientation as reference
    if face_basis_reference is None:

        face_basis_reference =face_basis.copy()

    else:

        #prevent axes from flipping between frames
        for axis in range(3):

            if np.dot(
                face_basis[:,axis],
                face_basis_reference[:,axis]
            )<0:

                face_basis[:,axis] *=-1

        #make sure basis is still right handed
        if np.linalg.det(face_basis)<0:
            face_basis[:,2] *=-1

        #slowly update reference orientation
        face_basis_reference =(
            0.95*face_basis_reference+
            0.05*face_basis
        )

        #restore orthogonal axes
        u,_values,vh =np.linalg.svd(
            face_basis_reference
        )

        face_basis_reference =u@vh

    return face_origin,face_basis

#calculate scale from stable nose landmarks
def calculate_face_scale(
    face_landmarks,
    width,
    height
):

    #get nose points
    nose_points =get_nose_points(
        face_landmarks,
        width,
        height
    )

    total_distance =0.0
    count =0

    #calculate average pairwise distance
    for i in range(len(nose_points)):

        for j in range(
            i+1,
            len(nose_points)
        ):

            distance =np.linalg.norm(
                nose_points[i]-
                nose_points[j]
            )

            total_distance +=distance
            count +=1

    if count ==0:
        return 1.0

    return total_distance/count

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