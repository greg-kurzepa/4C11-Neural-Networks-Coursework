import numpy as np

s = np.array([
    [4,2,1],
    [2,5,3],
    [1,3,6]
])

eigvals, eigvecs = np.linalg.eig(s)

print(eigvals)
print(eigvecs)

s_reconstructed = eigvecs @ np.diag(eigvals) @ np.linalg.inv(eigvecs)

print(s_reconstructed)