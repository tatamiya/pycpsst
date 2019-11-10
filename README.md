# pycpsst

`pycpsst` is a Python library for change point detection based on singular spectrum transformation.


# Basic Usage
```Python
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

from pycpsst import ChangePointSST

%matplotlib inline

# Sample data from ruptures
n = 2000  # number of samples, dimension
n_bkps, sigma = 2, 0.5  # number of change points, noise standart deviation
signal, bkps = rpt.pw_wavy(n, n_bkps, noise_std=sigma)

# Change Point Detection
Lm = 200
L_train, L_target = 150, 50
w = 30

sst = ChangePointSST(L_train=L_train, L_target=L_target, w=w)

Kw = signal.shape[0] - Lm + 1
sst_results = np.array([sst.score(signal[i:i+Lm]) for i in range(0, Kw)])


```

# References

- 井手剛「入門 機械学習による異常検知」（コロナ社，2015）
- 井手剛，杉山将「異常検知と変化検知」（講談社，2015）
