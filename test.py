import numpy as np, json
X = np.load('refined/X_test.npy')
y = np.load('refined/y_test.npy')
print('Range check  :', X.min(axis=0), X.max(axis=0))
print('Class balance:', np.bincount(y))