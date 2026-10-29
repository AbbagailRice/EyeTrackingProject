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

#nose landmarks used to calculate face scale
NOSE_INDICES =[
    4,45,275,220,440,1,5,51,281,44,274,241,
    461,125,354,218,438,195,167,393,165,391,
    3,248
]

#gaze smoothing settings
GAZE_FILTER_LENGTH =8

#gaze ray length
GAZE_RAY_LENGTH =180.0

#approx eye sphere radius
BASE_EYE_RADIUS =20.0

#webcam display size
CAM_WIDTH =500
CAM_HEIGHT =350

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