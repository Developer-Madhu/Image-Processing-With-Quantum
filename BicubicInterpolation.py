import numpy as np

v1 = int(input("Enter first value:- "))
v2 = int(input("Enter second value:- "))
v3 = int(input("Enter third value:- "))
v4 = int(input("Enter fourth value:- "))

# --- Cubic kernel (Catmull-Rom spline: a = -0.5) ---
def R3(u):
    u = abs(u)
    if u <= 1:
        return 1.5 * u**3 - 2.5 * u**2 + 1
    elif 1 < u < 2:
        return -0.5 * u**3 + 2.5 * u**2 - 4 * u + 2
    else:
        return 0

# --- 1D Cubic Interpolation ---
def cubic_interp(values, lam):
    return (
        R3(-1 - lam) * values[0] +
        R3(-lam)     * values[1] +
        R3(1 - lam)  * values[2] +
        R3(2 - lam)  * values[3]
    )

# --- Edge padding for a 1D array ---
def pad_1d(arr):
    return [arr[0], arr[0], arr[1], arr[1]]

# --- Apply 2D cubic interpolation on matrix ---
def interpolate_matrix(matrix, scale=4):
    orig_h, orig_w = matrix.shape
    out = np.zeros((scale, scale))

    # Step 1: Interpolate along x (rows), get intermediate values
    lambdas = np.linspace(0, 1, scale)
    intermediate = np.zeros((orig_h, scale))
    for row_idx in range(orig_h):
        row = pad_1d(matrix[row_idx])
        for j, lam in enumerate(lambdas):
            intermediate[row_idx][j] = cubic_interp(row, lam)

    # Step 2: Interpolate along y (columns), for each column in intermediate result
    for col_idx in range(scale):
        col = pad_1d(intermediate[:, col_idx])
        for i, lam in enumerate(lambdas):
            out[i][col_idx] = cubic_interp(col, lam)

    return np.round(out, 3)

# --- Input matrix ---
matrix = np.array([[v1, v2],
                   [v3, v4]])

# --- Perform interpolation ---
result = interpolate_matrix(matrix)

# --- Display output ---
print("Final 4×4 Interpolated Matrix (Cubic in both directions):")
print(result)
