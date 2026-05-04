import numpy as np


def lms(
    desired: np.ndarray,
    reference: np.ndarray,
    filter_order: int = 32,
    mu: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least Mean Squares (LMS) adaptive FIR filter.

    Attempts to make reference pass through the adaptive filter to produce
    output that matches desired.  Returns (output, error, final_weights).

    mu: step size. Convergence requires mu < 1 / (filter_order * E[x^2]).
    """
    n = len(desired)
    w = np.zeros(filter_order)
    output = np.zeros(n)
    error = np.zeros(n)
    x_buf = np.zeros(filter_order)
    for i in range(n):
        x_buf = np.roll(x_buf, 1)
        x_buf[0] = reference[i]
        y = np.dot(w, x_buf)
        e = desired[i] - y
        w += 2 * mu * e * x_buf
        output[i] = y
        error[i] = e
    return output, error, w
