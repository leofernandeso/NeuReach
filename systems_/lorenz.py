from functools import partial

import numpy as np

from base import simulate

def lorenz_dynamics(y, t):
    a, b, c = y
    a_dot = 10.0 * (b - a)
    b_dot = a * (28.0 - c) -b
    c_dot = a*b - 8.0/3*c
    dydt = [a_dot, b_dot, c_dot]
    return dydt

if __name__ == "__main__":
    ts = 0.05
    max_t = 20.
    t = np.arange(0, max_t+ts, ts)
    lorenz_simulator = partial(simulate, dynamic_func=lorenz_dynamics, t=t)
    t, sol = lorenz_simulator(x0=[15.0, 15.0, 36.])
