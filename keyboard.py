#this file manages the on-screen gaze keyboard
#it creates key positions and finds which key the user is looking at

#keyboard layout
KEYBOARD_ROWS =[
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L"],
    ["Z","X","C","V","B","N","M"]
]


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

    keyboard_top =int(
        screen_height*0.37
    )

    row_height =int(
        screen_height*0.10
    )

    key_gap =8
    row_gap =50

    #create letter rows
    for row_index,row in enumerate(
        KEYBOARD_ROWS
    ):

        key_width =int(
            (
                keyboard_width-
                key_gap*(len(row)-1)
            )/
            len(row)
        )

        row_width =(
            key_width*len(row)+
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

        for key_index,label in enumerate(
            row
        ):

            x1 =(
                start_x+
                key_index*
                (
                    key_width+
                    key_gap
                )
            )

            x2 =x1+key_width

            keys.append({
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

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