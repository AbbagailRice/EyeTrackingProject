#this file manages the on-screen gaze keyboard
#it creates key positions and finds which key the user is looking at

import math

from config import SNAP_DISTANCE_PX, SNAP_HYSTERESIS_PX, SNAP_HYSTERESIS_PX, CAM_HEIGHT

#where the gaze is snapped to the nearest key
current_snap_label =None

#keyboard layout
KEYBOARD_ROWS =[
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M"],
    ["SPACE","BACK"]
]

#if its not listed its just 1
KEY_WEIGHTS ={
    "SPACE": 4.0,
    "BACK": 1.5
}

#key size settings
KEY_HEIGHT_FRACTION =0.125
KEY_GAP =8
ROW_GAP =40

#space kept free below the keyboard
BOTTOM_MARGIN =95

#text box settings (the box sits just above the keyboard)
TEXT_BOX_HEIGHT_FRACTION =0.07
TEXT_BOX_GAP =12

#find the top of the keyboard
def get_keyboard_top(
    screen_height
):
    row_height =int(
        screen_height*KEY_HEIGHT_FRACTION
    )

    keyboard_height =(
        len(KEYBOARD_ROWS)*row_height+
        (len(KEYBOARD_ROWS)-1)*ROW_GAP
    )

    return (
        screen_height-
        BOTTOM_MARGIN-
        keyboard_height
    )

#get the rectangle of the text box above the keyboard
def get_text_box(
    screen_width,
    screen_height
):
    keyboard_width =int(
        screen_width*0.90
    )

    x1 =int(
        (screen_width-keyboard_width)/2
    )

    x2 =x1+keyboard_width
    y2 =(
        get_keyboard_top(screen_height)-
        TEXT_BOX_GAP
    )

    y1 =(
        y2-
        int(screen_height*TEXT_BOX_HEIGHT_FRACTION)
    )

    #keep the box clear of the webcam preview in the top left
    y1 =max(
        y1,
        CAM_HEIGHT+8
    )

    return x1,y1,x2,y2

#create keyboard key rectangles
def create_keyboard(
    screen_width,
    screen_height
):

    keys =[]

    #keyboard dimensions
    keyboard_width =int(
        screen_width*0.90
    )

    keyboard_top =get_keyboard_top(
    screen_height
)

    row_height =int(
        screen_height*KEY_HEIGHT_FRACTION
    )

    key_gap =KEY_GAP
    row_gap =ROW_GAP

    #create letter rows
    for row_index,row in enumerate(
        KEYBOARD_ROWS
    ):

        #calculate the width of each key based on its weight
        weights =[
            KEY_WEIGHTS.get(label,1.0)
            for label in row
        ]

        available_width =(
            keyboard_width-
            key_gap*(len(row)-1)
        )

        key_widths =[
            int(available_width*weight/sum(weights))
            for weight in weights
        ]

        row_width =(
            sum(key_widths)+
            key_gap*(len(row)-1)
        )

        start_x =int(
            (screen_width-row_width)/2
        )

        y1 =(
            keyboard_top+
            row_index*(
                row_height+
                row_gap
            )
        )

        y2 =(
            y1+
            row_height
        )

        x1 =start_x

        for label,key_width in zip(
            row,
            key_widths
        ):

            x2 =x1+key_width

            keys.append({
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
            x1 =x2+key_gap

    return keys

#find key containing current gaze
def get_gaze_key(
    keys,
    screen_gaze
):

    if screen_gaze is None:
        return None

    gaze_x,gaze_y =screen_gaze

    for key in keys:

        if (
            key["x1"]<=gaze_x<=key["x2"] and
            key["y1"]<=gaze_y<=key["y2"]
        ):

            return key["label"]

    return None

#distance in pixels from a point to a key rectangle (0 when the point is inside the key)
def distance_to_key(
    key,
    x,
    y
):

    dx =max(
        key["x1"]-x,
        0,
        x-key["x2"]
    )

    dy =max(
        key["y1"]-y,
        0,
        y-key["y2"]
    )

    return math.hypot(dx,dy)

#snap the gaze to the nearest key, but only when it is on or near the keys
#returns the snapped key label and the center of that key (both none when gaze is away from the keys)
def snap_to_key(
    keys,
    screen_gaze
):
    global current_snap_label

    if screen_gaze is None:

        current_snap_label =None

        return None,None

    gaze_x,gaze_y =screen_gaze

    #find nearest key to the gaze
    nearest_key =None
    nearest_distance =None

    for key in keys:
        distance =distance_to_key(
            key,
            gaze_x,
            gaze_y
        )

        if (
            nearest_distance is None or
            distance<nearest_distance
        ):

            nearest_key =key
            nearest_distance =distance

    #gaze is too far from every key so do not snap
    if (
        nearest_key is None or
        nearest_distance>SNAP_DISTANCE_PX
    ):
        current_snap_label =None

        return None,None

    chosen_key =nearest_key

    #hysteresis: stay on the current key until the gaze is clearly past its edge
    #stops the highlight flickering between two neighboring keys
    if current_snap_label is not None:

        for key in keys:
            if key["label"] ==current_snap_label:

                current_distance =distance_to_key(
                    key,
                    gaze_x,
                    gaze_y
                )

                if (
                    current_distance<=SNAP_DISTANCE_PX and
                    current_distance<=nearest_distance+SNAP_HYSTERESIS_PX
                ):

                    chosen_key =key

                break

    current_snap_label =chosen_key["label"]

    key_center =(
        (chosen_key["x1"]+chosen_key["x2"])//2,
        (chosen_key["y1"]+chosen_key["y2"])//2
    )

    return chosen_key["label"],key_center

#forget the current snapped key
def reset_snap():
    global current_snap_label
    current_snap_label =None