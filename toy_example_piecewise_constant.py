import math
import matplotlib.pyplot as plt
import numpy as np

anchors = [0.0, 1.0, 3.0, 4.0, 8.0, 10.0]
values = [3.5, 4.5, 1.5, 6.5, 2.5]
#anchors = [0.5*(intervals[i]+intervals[i+1]) for i in range(len(intervals)-1)]
intervals = [0.5*(anchors[i]+anchors[i+1]) for i in range(len(anchors)-1)]
anchors = anchors[1:]

def hard_piecewise(x):
    val = None
    for idx, i in enumerate(intervals):
        if x<i:
            return val
        else:
            val = values[idx]

def soft_piecewise(x, T=1.0):
    diff = [abs(x-anchor) for anchor in anchors]
    weights = [math.exp(-d/T) for d in diff]
    weights = [weight/sum(weights) for weight in weights]
    return sum(weights[i]*values[i] for i in range(len(values)))

t = np.arange(0, 10.0, 0.1).tolist()
hard = [hard_piecewise(i) for i in t]
soft = [soft_piecewise(i, T=0.2) for i in t]
plt.plot(t, hard, label=r"$f(x)$")
plt.plot(t, soft, label=r"$g(x,T)$")
plt.legend()
plt.show()

