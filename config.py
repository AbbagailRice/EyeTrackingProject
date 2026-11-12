#this file stores the settings and constants used by the eye tracking project
#it includes mediapipe landmarks camera settings gaze settings and head pose points

import numpy as np


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

#nose landmarks used to calculate face scale !!!(no longer used for the face frame)!!!
NOSE_INDICES =[
    4,45,275,220,440,1,5,51,281,44,274,241,
    461,125,354,218,438,195,167,393,165,391,
    3,248
]

#rigid landmarks for fitting the face coordinate system
#forehead, nose bridge, nose, eye corners, temples and cheek edges
RIGID_FACE_INDICES =[
    10,151,9,8,168,6,197,195,5,4,1,
    33,133,362,263,130,359,
    127,356,234,454,
    21,251,54,284
]

#frames used for the reference face shape at startup
FACE_REFERENCE_FRAMES =30
 
#fraction of rigid landmarks kept after dropping the worst fitting ones
POSE_INLIER_FRACTION =0.8
 
#frames collected for eye center calibration and monitor calibration
EYE_CALIBRATION_FRAMES =45
MONITOR_CALIBRATION_FRAMES =45
 
#eye openness ratio (eyelid gap/eye width) below this counts as a blink
BLINK_RATIO_THRESHOLD =0.20

#corner calibration settings
CALIBRATION_SAMPLE_COUNT =15

#gaze smoothing settings
GAZE_FILTER_LENGTH =8

#gaze ray length
GAZE_RAY_LENGTH =180.0

#approx eye sphere radius
BASE_EYE_RADIUS =20.0

#webcam display size
CAM_WIDTH =350
CAM_HEIGHT =200

#webcam resolution
CAMERA_WIDTH =1280
CAMERA_HEIGHT =720


#model file path
MODEL_PATH ="face_landmarker.task"


#3d reference face points
#should approx the shape of a normal human face
FACE_MODEL_POINTS =np.array([
    (0.0,0.0,0.0),
    (0.0,-63.6,-12.5),
    (-43.3,32.7,-26.0),
    (43.3,32.7,-26.0),
    (-28.9,-28.9,-24.1),
    (28.9,-28.9,-24.1)
],dtype =np.float64)


#3d coordinate axis points
AXIS_POINTS =np.array([
    (50.0,0.0,0.0),
    (0.0,50.0,0.0),
    (0.0,0.0,50.0)
],dtype =np.float64)

#forehead landmark
FOREHEAD =10

#virtual monitor dimensions
MONITOR_WIDTH_CM =60.0
MONITOR_HEIGHT_CM =40.0

#virtual monitor distance from eyes
MONITOR_DISTANCE_CM =50.0