from functools import partial

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
    vanderpol_simulator = partial(simulate, dynamic_func=vanderpol_dynamics, ts=0.05, max_t=20.)
    t, sol = vanderpol_simulator(x0=[1., 1.])
