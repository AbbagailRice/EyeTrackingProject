#this file handles face tracking and head position using mediapipe and open cv
#it creates the local face coordinate system and calculates the head pose axes

import cv2
import mediapipe as mp
import numpy as np

from config import *

#reference rigid landmark shape
face_reference_points =None

#frames collected while building the reference
reference_buffer =[]

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
 
#get rigid face landmarks in 3d
def get_rigid_points(
    face_landmarks,
    width,
    height
):
 
    return np.array([
        landmark_to_3d(
            face_landmarks[index],
            width,
            height
        )
        for index in RIGID_FACE_INDICES
    ],dtype =np.float64)

#find scale rotation and translation so that
#target is about the scale*rotationATsource+translation
def fit_similarity(
    source,
    target
):
 
    count =len(source)
 
    source_center =source.mean(axis =0)
    target_center =target.mean(axis =0)
 
    source_centered =source-source_center
    target_centered =target-target_center
 
    #cross covariance between the two point sets
    covariance =(
        target_centered.T@
        source_centered
    )/count
 
    u,singular_values,vh =np.linalg.svd(
        covariance
    )
 
    #fix any reflection so rotation is always right handed
    sign =np.ones(3)
 
    if np.linalg.det(u)*np.linalg.det(vh)<0:
        sign[2] =-1.0
 
    rotation =u@np.diag(sign)@vh
 
    source_variance =(
        source_centered**2
    ).sum()/count
 
    if source_variance<0.000001:
        scale =1.0
 
    else:
        scale =(
            singular_values*sign
        ).sum()/source_variance
 
    translation =(
        target_center-
        scale*rotation@source_center
    )
 
    return scale,rotation,translation
 
#fit similarity then drop the worst landmarks and fit again
def fit_similarity_robust(
    source,
    target
):
 
    scale,rotation,translation =fit_similarity(
        source,
        target
    )
 
    predicted =(
        scale*(source@rotation.T)+
        translation
    )
 
    residuals =np.linalg.norm(
        predicted-target,
        axis =1
    )
 
    keep_count =max(
        6,
        int(len(source)*POSE_INLIER_FRACTION)
    )
 
    keep =np.argsort(
        residuals
    )[:keep_count]
 
    return fit_similarity(
        source[keep],
        target[keep]
    )
 
#collect frames and build the reference face shape
def update_face_reference(points):
 
    global face_reference_points
 
    reference_buffer.append(points)
 
    if len(reference_buffer)<FACE_REFERENCE_FRAMES:
        return
 
    #align every frame to the first one
    anchor =reference_buffer[0]
    aligned =[anchor]
 
    for frame_points in reference_buffer[1:]:
 
        scale,rotation,translation =fit_similarity_robust(
            frame_points,
            anchor
        )
 
        aligned.append(
            scale*(frame_points@rotation.T)+
            translation
        )
 
    #median shape rejects bad frames
    median_shape =np.median(
        np.array(aligned),
        axis =0
    )
 
    #center reference so the fitted translation is the face origin
    face_reference_points =(
        median_shape-
        median_shape.mean(axis =0)
    )
 
    reference_buffer.clear()
 
    print("\nface reference shape ready")
 
 
#true once the reference shape exists
def is_face_reference_ready():
 
    return face_reference_points is not None
 
 
#how many frames of the reference have been collected
def get_face_reference_progress():
 
    return len(reference_buffer)
 
 
#reset the reference shape and start over
def reset_face_reference():
 
    global face_reference_points
 
    face_reference_points =None
    reference_buffer.clear()
 

#create stable local coordinate system from rigid landmarks
#returns face origin, face basis (rotation) and face scale
#returns none vals while the reference shape is still being built
def create_face_coordinate_system(
    face_landmarks,
    width,
    height
):

    #get stable nose region landmarks
    points =get_rigid_points(
        face_landmarks,
        width,
        height
    )

    if face_reference_points is None:
 
        update_face_reference(points)
 
        return None,None,None
 
    scale,rotation,translation =fit_similarity_robust(
        face_reference_points,
        points
    )
 
    #reference is centered so translation is the face origin
    return translation,rotation,scale

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