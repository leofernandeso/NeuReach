from typing import List, Union, Callable

import numpy as np
from scipy.integrate import odeint

State = Union[np.array, List]
DynamicFunc = Callable[[State, float], State]

def simulate(
    dynamic_func: DynamicFunc,
    x0: State,
    t: np.array,
    odeint_kwargs: dict=None
) -> np.array:
    
    if not odeint_kwargs:
        odeint_kwargs = {}

    sol = odeint(dynamic_func, x0, t, **odeint_kwargs)
    return t, sol
