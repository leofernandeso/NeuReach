from typing import Callable
from functools import partial, cached_property

import numpy as np

from systems_.vanderpol import vanderpol_simulator

# An InitialStateSampler will be a callable that returns us with a sampled initial condition x0
InitialStateSampler = Callable[[], 'State']

class TraceSampler:
    def __init__(
        self, dt: float, tmax: float, n_states: int,
        simulate_func: 'Simulator', initial_state_sampler: InitialStateSampler
    ):
        self.dt = dt
        self.tmax = tmax
        self.n_states = n_states
        self.simulate_func = simulate_func
        self.initial_state_sampler = initial_state_sampler

        self.t = np.arange(0, self.tmax + self.dt, self.dt)

    @cached_property
    def n_timesteps(self) -> int:
        return len(self.t)

    def sample_t(self) -> np.array:
        return np.random.choice(self.t)

    def sample_trace(self, x0: 'State'=None) -> np.array:
        """
        Samples a trace. If x0 is not specified, it uses the provided initial_state_sampler to sample an initial condition
        """
        if not x0:
            x0 = self.initial_state_sampler()
        trajectory = self.simulate_func(x0=x0, t=self.t)
        return trajectory

    def sample_traces(self, n_traces: int) -> np.array:
        """
        Samples 'n_traces' trajectories using the attribute 'simulate_func' to generate initial conditions.
        Returns a np.array of shape (n_traces, number_of_timesteps, number_of_states).
        """
        traces = np.empty((n_traces, self.n_timesteps, self.n_states))
        for i in range(n_traces):
            trace = self.sample_trace()
            traces[i, :, :] = trace
        return traces

TMAX = 4.
dt = 0.05

# range of initial states
lower = np.array([1., 2.])
higher = np.array([2., 3.])
X0_center_range = np.array([lower, higher]).T
X0_r_max = 0.5

def sample_X0():
    center = X0_center_range[:,0] + np.random.rand(X0_center_range.shape[0]) * (X0_center_range[:,1]-X0_center_range[:,0])
    r = np.random.rand()*X0_r_max
    X0 = np.concatenate([center, np.array(r).reshape(-1)])
    return X0

def sample_x0(X0):
    center = X0[:-1]
    r = X0[-1]

    n = len(center)
    direction = np.random.randn(n)
    direction = direction / np.linalg.norm(direction)

    x0 = center + direction * r
    x0[x0>X0_center_range[:,1]] = X0_center_range[x0>X0_center_range[:,1],1]
    x0[x0<X0_center_range[:,0]] = X0_center_range[x0<X0_center_range[:,0],0]
    return x0

if __name__ == "__main__":
    X0 = sample_X0()
    n_states = X0.shape[0] - 1
    initial_state_sampler = partial(sample_x0, X0)
    trace_sampler = TraceSampler(
        dt=dt, tmax=TMAX, n_states=n_states,
        simulate_func=vanderpol_simulator, initial_state_sampler=initial_state_sampler
    )
