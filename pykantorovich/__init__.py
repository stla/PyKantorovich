__version__ = '0.1.0'

from math import isclose
import numpy as np
import cdd
from scipy.optimize import linprog
from scipy import sparse
import cvxpy as cp
import re
from fractions import Fraction
import numbers

def distance_matrix(numbers, distance="euclidean"):
    numbers = np.asarray(numbers)
    diffs = numbers[:, None, :] - numbers[None, :, :]
    if distance == "euclidean":
        return np.linalg.norm(diffs, axis=-1)
    if distance == "manhattan":
        return np.sum(np.abs(diffs), axis=-1)


def _makeH(mu, nu, number_type):
    n = len(mu)
    n2 = n*n
    b1 = np.zeros(n2, dtype=int)
    a1 = np.column_stack((b1, np.diag(np.ones(n2, dtype=int))))
    nones = np.ones(n, dtype=int)
    a2 = np.vstack(
            (
                np.kron(np.eye(n, dtype=int), nones),
                np.tile(np.diag(nones), n)
            )
        )
    b2 = np.concatenate((mu, nu))
    a2 = np.column_stack((b2, -a2))
    mat = cdd.Matrix(a1, linear=False, number_type=number_type)
    mat.extend(a2, linear=True)
    return (mat, n)

    
def _kantorovich_cdd(
    mu, nu, number_type="fraction", distance_matrix="0-1", prettyprint=True
):
    mat, n = _makeH(mu, nu, number_type)
    mat.obj_type = cdd.LPObjType.MIN
    if distance_matrix == "0-1":
        d = np.ones((n, n), dtype=int)
        np.fill_diagonal(d, 0)
        d = d.flatten()
    else:
        d = distance_matrix.flatten()
    d = np.concatenate(([0], d))
    mat.obj_func = d
    lp = cdd.LinProg(mat)
    lp.solve()
    sol = lp.primal_solution
    if prettyprint:
        joining_to_print = np.reshape(np.frompyfunc(format, 1, 1)(sol), (n,n))
        print("{\n",
            "distance:", format(lp.obj_value),
            "\n joining:\n", format(joining_to_print),
            "\n optimal:", "yes" if lp.status == cdd.LPStatusType.OPTIMAL else "no", 
        "\n}")
    return {
        "distance": lp.obj_value,
        "joining": np.reshape(sol, (n, n)),
        "optimal": "yes" if lp.status == cdd.LPStatusType.OPTIMAL else "no" 
    }

mu = ['1/7','2/7','4/7']
nu = ['1/4','1/4','1/2']

# mu = ['1/2','1/4','1/4']
# nu = ['1/4','1/4','1/2']

def extreme_joinings(mu, nu, number_type="fraction", prettyprint=True):
    """
    Extreme joinings of two probabiity measures.

    Parameters
    ----------
    mu : array-like
        A probability vector.
    nu : array-like
        A probability vector. Must have the same length as `mu`.
    number_type : str
        The type to use for the calculations, either `"fraction"` or `"float"`.
    prettyprint: bool
        Whether to pretty-print the results (especially when `number_type="fraction"`).

    Returns
    -------
    list
        The extreme joinings of `mu` and `nu`.
        
    Examples
    --------
    >>> mu = ['1/2','1/4','1/4']
    >>> nu = ['1/4','1/4','1/2']
    >>> joinings = extreme_joinings(mu, nu)
    [['1/4' '0' '1/4']
     ['0' '1/4' '0']
     ['0' '0' '1/4']]
    [['1/4' '0' '1/4']
     ['0' '0' '1/4']
     ['0' '1/4' '0']]
    [['1/4' '1/4' '0']
     ['0' '0' '1/4']
     ['0' '0' '1/4']]
    [['0' '1/4' '1/4']
     ['1/4' '0' '0']
     ['0' '0' '1/4']]
    [['0' '1/4' '1/4']
     ['0' '0' '1/4']
     ['1/4' '0' '0']]
    [['0' '0' '1/2']
     ['0' '1/4' '0']
     ['1/4' '0' '0']]
    [['0' '0' '1/2']
     ['1/4' '0' '0']
     ['0' '1/4' '0']]

    """
    if(len(mu) != len(nu)):
        raise ValueError("`mu` and `nu` must have the same length.")
    mu = np.asarray(mu)
    nu = np.asarray(nu)
    if (
            number_type == "fraction"
            and np.all(np.frompyfunc(_is_fraction, 1, 1)(mu))
            and np.all(np.frompyfunc(_is_fraction, 1, 1)(nu))
        ):
        mu_fr = np.frompyfunc(_to_fraction, 1, 1)(mu)
        if np.sum(mu_fr) != 1:
            raise ValueError("`mu` does not sum to one.")
        nu_fr = np.frompyfunc(_to_fraction, 1, 1)(nu)
        if np.sum(nu_fr) != 1:
            raise ValueError("`nu` does not sum to one.")
    else: 
        if not np.all(np.frompyfunc(_is_number, 1, 1)(mu)):
            raise ValueError("`mu` is not made of numbers.")
        if not np.all(np.frompyfunc(_is_number, 1, 1)(nu)):
            raise ValueError("`nu` is not made of numbers.")
        if not isclose(np.sum(mu), 1.0):
            raise ValueError("`mu` does not sum to one.")
        if not isclose(np.sum(nu), 1.0):
            raise ValueError("`nu` does not sum to one.")
    mat, n = _makeH(mu, nu, number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    V = cdd.Polyhedron(mat).get_generators()
    flatjoinings = np.asarray(V)[:, 1:].tolist()
    to_return = [np.reshape(j, (n, n)) for j in flatjoinings]
    if prettyprint:
        for i in range(len(flatjoinings)):
            flatjoinings[i] = np.frompyfunc(format, 1, 1)(flatjoinings[i])
        [print(format(np.reshape(j, (n, n)))) for j in flatjoinings]
    return to_return


def _kantorovich_sparse(mu, nu, distance_matrix = "0-1"):
    n = len(mu)
    n2 = n*n
    eyen = sparse.eye(n, dtype = int)    
    A = eyen
    B = sparse.csr_matrix(np.ones(n, dtype = int))
    M1 = sparse.kron(A, B)
    M2 = sparse.hstack([eyen]*n, dtype=int)
    M = sparse.vstack([M1, M2])
    a1 = sparse.eye(n2, dtype = int)
    b1 = np.zeros(n2, dtype=int)
    b2 = np.concatenate((mu, nu))
    if distance_matrix == "0-1":
        d = np.ones((n, n), dtype=int)
        np.fill_diagonal(d, 0)
        d = d.flatten()
    else:
        d = distance_matrix.flatten()
    res = linprog(
        d, A_ub = -a1, b_ub = b1, A_eq = M, b_eq = b2, 
        method="highs-ipm"
    )
    return {
        "distance": res.fun,
        "joining": np.reshape(res.x, (n,n)),
        "message": res.message
    }

# mu = [1/7,2/7,4/7]
# nu = [1/4,1/4,1/2]


def _kantorovich_cvx(mu, nu, distance_matrix = "0-1"):
    n = len(mu)
    n2 = n*n
    eyen = np.eye(n, dtype = int)    
    A = eyen
    nones = np.ones(n, dtype = int)
    M1 = np.kron(A, nones)
    M2 = np.tile(np.diag(nones), n)
    M = np.vstack((M1, M2))
    a1 = np.eye(n2, dtype = int)
    b1 = np.zeros(n2, dtype=int)
    b2 = np.concatenate((mu, nu))
    if distance_matrix == "0-1":
        d = np.ones((n, n), dtype=int)
        np.fill_diagonal(d, 0)
        d = d.flatten()
    else:
        d = distance_matrix.flatten()
    p = cp.Variable(n*n)
    constraints = [M @ p == b2, p >= 0]
    obj = cp.Minimize(sum(cp.multiply(p, d)))
    prob = cp.Problem(obj, constraints)
    distance = prob.solve(solver=cp.OSQP)
    joinings = prob.solution.primal_vars
    for k in joinings.keys():
        joinings[k] = np.reshape(joinings[k], (n, n))
    return {
        "distance": distance, 
        "joining": joinings
    }


def _is_fraction(a):
    if not isinstance(a, str):
        return False
    r = "^\\d+\/\\d+$"
    m = re.fullmatch(r, a)
    if m is None:
        return False
    return m.string == a

def _to_fraction(a):
    num, den = re.split("\/", a)
    return Fraction(int(num), int(den))

def _is_number(x):
    return isinstance(x, numbers.Number) and (not isinstance(x, complex))


def kantorovich(
    mu, nu, distance_matrix = "0-1", method = "cdd", number_type="fraction",
    prettyprint = True
):
    """
    Kantorovich distance between two probabiity measures on a finite set.

    Parameters
    ----------
    mu : array-like
        Array-like representing a probability.
    nu : array-like
        Array-like representing a probability. Must have the same length as `mu`.
    distance_matrix : n x n matrix
        The cost matrix. The default is "0-1", the zero-or-one 
        distance. Note that the implemented calculation of the Kantorovich 
        distance is totally useless in this case, because it equals 
        `1.0 - np.sum(np.minimum(mu, nu))`.
    method : string
        The method to use. Can be "cdd", "sparse", or "cvx".
    number_type : string
        For method="cdd" only. Can be "float" or "fraction". 
        The default is "fraction".
    prettyprint : bool
        This is only for `method="cdd"` and `number_type="fraction"`. This 
        prints a more readable result.

    Returns
    -------
    The Kantorovich distance between `mu` and `nu` and a joining of `mu` and 
    `nu` for which the Kantorovich distance is the probability that the two 
    margins differ.
    
    Examples
    --------
    >>> mu = ['1/7','2/7','4/7']
    >>> nu = ['1/4','1/4','1/2']
    >>> kantorovich(mu, nu)
    {
     distance: 3/28 
     joining:
     [['1/7' '0' '0']
     ['1/28' '1/4' '0']
     ['1/14' '0' '1/2']] 
     optimal: yes 
    }

    """
    n = len(mu)
    if(len(nu) != n):
        raise ValueError("`mu` and `nu` must have the same length.")
    mu = np.asarray(mu)
    nu = np.asarray(nu)
    if (
            number_type == "fraction"
            and np.all(np.frompyfunc(_is_fraction, 1, 1)(mu))
            and np.all(np.frompyfunc(_is_fraction, 1, 1)(nu))
        ):
        mu_fr = np.frompyfunc(_to_fraction, 1, 1)(mu)
        if np.sum(mu_fr) != 1:
            raise ValueError("`mu` does not sum to one.")
        nu_fr = np.frompyfunc(_to_fraction, 1, 1)(nu)
        if np.sum(nu_fr) != 1:
            raise ValueError("`nu` does not sum to one.")
    else: 
        if not np.all(np.frompyfunc(_is_number, 1, 1)(mu)):
            raise ValueError("`mu` is not made of numbers.")
        if not np.all(np.frompyfunc(_is_number, 1, 1)(nu)):
            raise ValueError("`nu` is not made of numbers.")
        if not isclose(np.sum(mu), 1.0):
            raise ValueError("`mu` does not sum to one.")
        if not isclose(np.sum(nu), 1.0):
            raise ValueError("`nu` does not sum to one.")
    if distance_matrix != "0-1":
        distance_matrix = np.asarray(distance_matrix)
        if distance_matrix.shape != (n, n):
            raise ValueError(f"The distance matrix must be square, with {n} rows and {n} columns.") 
        type_d = distance_matrix.dtype.kind
        if type_d != "f" and type_d != "i":
            raise ValueError("The distance matrix is not a matrix of numbers.")
    if method != "cdd" and method != "sparse" and method != "cvx":
        raise ValueError("Invalid `method` argument.")
    if method == "cdd":
        return _kantorovich_cdd(mu, nu, number_type, distance_matrix, prettyprint)
    if method == "sparse":
        return _kantorovich_sparse(mu, nu, distance_matrix)
    if method == "cvx":
        return _kantorovich_cvx(mu, nu, distance_matrix)