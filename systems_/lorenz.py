from functools import partial

from base import simulate

def lorenz_dynamics(y, t):
    a, b, c = y
    a_dot = 10.0 * (b - a)
    b_dot = a * (28.0 - c) -b
    c_dot = a*b - 8.0/3*c
    dydt = [a_dot, b_dot, c_dot]
    return dydt

if __name__ == "__main__":
    lorenz_simulator = partial(simulate, dynamic_func=lorenz_dynamics, ts=0.05, max_t=20.)
    t, sol = lorenz_simulator(x0=[15.0, 15.0, 36.])
