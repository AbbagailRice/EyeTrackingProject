#this file smooths the final screen gaze position with a one euro filter
#heavy smoothing when the eyes are still (fixation) and light smoothing while they move fast (saccade)

import numpy as np

class OneEuro:
    def __init__(
        self,
        min_cutoff =1.0,
        beta =0.02,
        d_cutoff =1.0
    ):

        #lower min_cutoff=smoother when still but more lag
        self.min_cutoff =min_cutoff

        #higher beta=less lag when moving fast but more jitter
        self.beta =beta

        #smoothing of the speed estimate itself
        self.d_cutoff =d_cutoff

        self.x_prev =None
        self.dx_prev =None
        self.t_prev =None

    #forget the past so the next value starts fresh
    def reset(self):
        self.x_prev =None
        self.dx_prev =None
        self.t_prev =None

    #convert a cutoff freq into a smoothing factor
    @staticmethod
    def _alpha(cutoff,dt):

        tau =1.0/(2*np.pi*cutoff)

        return 1.0/(1.0+tau/dt)

    #filter one value (number or array) taken at time t in seconds
    def __call__(self,x,t):

        x =np.asarray(
            x,
            dtype =np.float64
        )

        #first value passes straight through
        if self.t_prev is None:

            self.x_prev =x
            self.dx_prev =np.zeros_like(x)
            self.t_prev =t

            return x

        dt =max(
            t-self.t_prev,
            1e-3
        )

        #speed of the signal
        dx =(x-self.x_prev)/dt

        a_d =self._alpha(
            self.d_cutoff,
            dt
        )

        dx_hat =a_d*dx+(1-a_d)*self.dx_prev

        #faster movement will raise the cutoff so less lag
        cutoff =(
            self.min_cutoff+
            self.beta*np.linalg.norm(dx_hat)
        )

        a =self._alpha(
            cutoff,
            dt
        )

        x_hat =a*x+(1-a)*self.x_prev

        self.x_prev =x_hat
        self.dx_prev =dx_hat
        self.t_prev =t

        return x_hat
