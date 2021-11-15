__version__ = '0.1.0'

from math import isclose
import numpy as np
import cdd

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

# mu = ['1/7','2/7','4/7']
# nu = ['1/4','1/4','1/2']

# kantorovich_cdd(mu, nu)  

def extreme_joinings(mu, nu, number_type="fraction"):
    mat, n = makeH(mu, nu, number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    V = cdd.Polyhedron(mat).get_generators()
    flatjoinings= np.asarray(V)[:, 1:].tolist()
    return [np.reshape(j, (n, n)) for j in flatjoinings]

