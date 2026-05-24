import numpy as np

from map_builder.optimization.residual_math import finite_difference_jacobian


def test_finite_difference_jacobian_linear_function() -> None:
    A = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 3.0]])
    b = np.array([4.0, -2.0])
    x = np.array([0.2, -0.5, 1.0])

    J = finite_difference_jacobian(lambda v: A @ v + b, x)
    np.testing.assert_allclose(J, A, atol=1e-7)


def test_finite_difference_jacobian_quadratic_function() -> None:
    x = np.array([2.0, -3.0])
    J = finite_difference_jacobian(lambda v: np.array([v[0] ** 2, v[0] * v[1]]), x)
    np.testing.assert_allclose(J, [[4.0, 0.0], [-3.0, 2.0]], atol=1e-5)
