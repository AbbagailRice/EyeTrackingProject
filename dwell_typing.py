#this file handles dwell typing
#a key is typed when the gaze stays on it for the dwell time
#after a key is typed the gaze needs to leave it before it can be typed again (so a held look does not repeat)

class DwellTyper:
    def __init__(
        self,
        dwell_time =0.8,
        grace_time =0.15,
        release_time =0.25,
        flash_time =0.25
    ):

        #seconds the gaze must stay on a key to type it
        self.dwell_time =dwell_time

        #seconds the gaze can be off the keys before the dwell timer is cleared (covers noisy frames)
        self.grace_time =grace_time

        #seconds the gaze must be off a typed key before it can be typed again
        self.release_time =release_time

        #seconds the typed key stays flashed
        self.flash_time =flash_time

        #the text typed so far
        self.text =""

        self.last_time =None

        #key currently being dwelled on
        self.candidate_label =None
        self.held_time =0.0
        self.off_time =0.0

        #key that was just typed and is waiting for the gaze to leave
        self.locked_label =None
        self.locked_off_time =0.0

        #key to flash after typing
        self.flash_label =None
        self.flash_until =0.0

    #clear the dwell timer but keep the typed text
    def reset_dwell(self):
        self.last_time =None

        self.candidate_label =None
        self.held_time =0.0
        self.off_time =0.0

        self.locked_label =None
        self.locked_off_time =0.0

    #clear the typed text
    def clear_text(self):
        self.text =""

    #apply a typed key to the text
    def apply_key(
        self,
        label
    ):

        if label =="BACK":
            self.text =self.text[:-1]

        elif label =="SPACE":
            self.text +=" "

        else:
            self.text +=label

    #update the dwell timer with the key under the gaze at time now (seconds)
    #returns the label of the key that was typed this frame or none
    #paused freezes the timer (eyes closed or face lost)
    def update(
        self,
        gaze_key,
        now,
        paused =False
    ):

        #time since the last frame (capped so a stall does not count as dwell time)
        if self.last_time is None:
            dt =0.0

        else:
            dt =min(
                max(now-self.last_time,0.0),
                0.2
            )
        self.last_time =now

        if paused:
            return None

        #after typing a key the gaze needs to leave it before it can be typed again
        if self.locked_label is not None:
            if gaze_key ==self.locked_label:
                self.locked_off_time =0.0

            else:
                self.locked_off_time +=dt

                if self.locked_off_time>=self.release_time:
                    self.locked_label =None

        #still looking at the key that was just typed
        if (
            self.locked_label is not None and
            gaze_key ==self.locked_label
        ):

            self.candidate_label =None
            self.held_time =0.0
            self.off_time =0.0

            return None

        #gaze is not on any key
        if gaze_key is None:
            if self.candidate_label is not None:
                self.off_time +=dt

                if self.off_time>=self.grace_time:
                    self.candidate_label =None
                    self.held_time =0.0
                    self.off_time =0.0

            return None

        #gaze moved to a new key so the timer starts over
        if gaze_key !=self.candidate_label:

            self.candidate_label =gaze_key
            self.held_time =0.0
            self.off_time =0.0

            return None

        #gaze stayed on the same key
        self.held_time +=dt
        self.off_time =0.0

        if self.held_time<self.dwell_time:
            return None

        #dwell time reached so type the key
        label =self.candidate_label

        self.apply_key(label)

        self.locked_label =label
        self.locked_off_time =0.0

        self.candidate_label =None
        self.held_time =0.0

        self.flash_label =label
        self.flash_until =now+self.flash_time

        return label

    #key being dwelled on and how far along it is (0 to 1)
    def get_progress(self):
        if self.candidate_label is None:
            return None,0.0

        return (
            self.candidate_label,
            min(self.held_time/self.dwell_time,1.0)
        )

    #key to flash right now (the one just typed) or none
    def get_flash_label(
        self,
        now
    ):
        if now<self.flash_until:
            return self.flash_label

        return None