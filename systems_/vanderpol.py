from functools import partial

import numpy as np

from base import simulate

def vanderpol_dynamics(y, t):
    a, b = y
    a = float(a)
    b = float(b)
    a_dot = b
    b_dot = (1 - a**2)*b - a
    dydt = [a_dot, b_dot]
    return dydt

if __name__ == "__main__":
    ts = 0.05
    max_t = 20.
    t = np.arange(0, max_t+ts, ts)
    vanderpol_simulator = partial(simulate, dynamic_func=vanderpol_dynamics, t=t)
    t, sol = vanderpol_simulator(x0=[1., 1.])
