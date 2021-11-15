__version__ = '0.1.0'

from math import isclose
import numpy as np
import cdd
from scipy.optimize import linprog
from scipy import sparse


def distance_matrix(numbers, distance="euclidean"):
    numbers = np.asarray(numbers)
    diffs = numbers[:, None, :] - numbers[None, :, :]
    if distance == "euclidean":
        return np.linalg.norm(diffs, axis=-1)
    if distance == "manhattan":
        return np.sum(np.abs(diffs), axis=-1)


def makeH(mu, nu, number_type):
    mu = np.asarray(mu)
    nu = np.asarray(nu)
    n = len(mu)
    if n != len(nu):
        raise ValueError("")
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

    
def kantorovich_cdd(mu, nu, number_type="fraction", distance_matrix = "0-1"):
    mat, n = makeH(mu, nu, number_type)
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
    return {
        "distance": lp.obj_value,
        "joining": np.reshape(lp.primal_solution, (n, n)),
        "optimal": "yes" if lp.status == cdd.LPStatusType.OPTIMAL else "no" 
    }

mu = ['1/7','2/7','4/7']
nu = ['1/4','1/4','1/2']

# kantorovich_cdd(mu, nu)  

def extreme_joinings(mu, nu, number_type="fraction"):
    mat, n = makeH(mu, nu, number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    V = cdd.Polyhedron(mat).get_generators()
    flatjoinings= np.asarray(V)[:, 1:].tolist()
    return [np.reshape(j, (n, n)) for j in flatjoinings]


def kantorovich_sparse(mu, nu, distance_matrix = "0-1"):
    mu = np.asarray(mu)
    nu = np.asarray(nu)
    n = len(mu)
    if n != len(nu):
        raise ValueError("")
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

mu = [1/7,2/7,4/7]
nu = [1/4,1/4,1/2]

import cvxpy as cp

def kantorovich_cvx(mu, nu, distance_matrix = "0-1"):
    mu = np.asarray(mu)
    nu = np.asarray(nu)
    n = len(mu)
    if n != len(nu):
        raise ValueError("")
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
    return (distance, prob.solution.primal_vars)
