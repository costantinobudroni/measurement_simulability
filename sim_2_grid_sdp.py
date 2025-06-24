"""
This code computes the minimum distance between a measurement assemblage M^eta, consisting of noisy versions of Pauli mesurements,
and the set of assemblage that can be simulated by two measurements with a set of preprocessing distribution on a finite grid, i.e.,
a grid approximation of SIM_2.

The grid consist of Ngrid^3 points and for each the minimal distance is computed via an SDP.

"""
import numpy as np
import cvxpy as cp
import mosek
from datetime import datetime
from joblib import Parallel, delayed

## Transform list of indices (tensor) into single index (vector)
def ntoone(v, d):
    ind=0
    n=len(v)
    for i in range(n):
        ind += v[i]*d**(n-i-1)

    return ind

## Transform single index (vector) into list of indices (tensor)
def oneton(ind, d, n):
    v = np.zeros(n, dtype=int)
    for i in range(n):
        v[-i-1] = ind % d
        ind = ind // d

    return list(v)

## Compute spectral projectors of a Pauli matrix
def obs_to_proj(X):
    I = np.eye(2)
    p0 = (X+ I)/2
    return p0, I-p0


## Define deterministic strategy for classical postprocessing of measurements
def q(a,x,ap):
    apv = oneton(ap,2,3)
    if apv[x] == a:
        return 1

    return 0

## Solve the SDP with parameters, provide the first probability for the grid search in order to parallelize
def SDP_grid(M, Ngrid, pgrid0):
# Initialize the SDP 
    nA = 2
    nX = 3
    nB = 8
    nY = 2
    dim = M[0].shape[0]

    ## representation of M'_{a'|x'} as N_{b|y}
    N = [cp.Variable((dim, dim), hermitian=True) for i in range(nB*nY)]
    constr = []
    constr += [N[i] >> 0 for i in range(nB*nY)]
    constr += [sum([N[y*nB+b] for b in range(nB)]) == np.eye(2) for y in range(nY)]
    nu = cp.Variable(nA*nX)
    ppar = cp.Parameter(6)

    for x in range(nX):
        for a in range(nA):
            #construct the simulated POVM for M_{a|x}
            Mp = sum([ppar[y*nY+x]*sum([N[y*nB+b]*q(a, x, b) for b in range(nB)]) for y in range(nY)])
            constr += [nu[x*nA+a]*I - M[x*nA+a] + Mp >> 0]
            constr += [M[x*nA+a] - Mp + nu[x*nA+a]*I >> 0]

    objexp = sum(nu)
    obj = cp.Minimize(objexp)
    probl = cp.Problem(obj, constr)

    # Construct the remaining grid points
    pgrid = np.arange(Ngrid)/(Ngrid-1)
    val = []
    for i in range(Ngrid):
        for j in range(Ngrid):
            # definition of the probability parameters p(y|x), only p(0|x) is relevant
            pp = [pgrid0, pgrid[i], pgrid[j]]
            p = np.array([pp[0], 1-pp[0], pp[1], 1-pp[1], pp[2], 1-pp[2]])
            # set the SDP parameter to the probability distribution p
            ppar.value = p
            probl.solve(solver='MOSEK',verbose=False)
            val += [objexp.value]

    return val

##########################################################################
#                                                                        #
################################## MAIN ##################################
#                                                                        #
##########################################################################

start_time = datetime.now()

## Define the measurement assemblage M^eta
X = np.array([[0,1],[1,0]])
Y = np.array([[0, -1j],[1j, 0]])
Z = np.array([[1,0],[0,-1]])
I = np.eye(2)
Pauli = [X,Y,Z]

eta = (np.sqrt(2)+1)/3
M = []
for i in range(3):
    p0, p1 = obs_to_proj(Pauli[i])
    M += [eta*p0 + (1-eta)*I/2]
    M += [eta*p1 + (1-eta)*I/2]



# Define the grid size
Ngrid = 10

# construct the vector of grid points in one direction, this is used to parellelize the computation
pgrid0 = np.arange(Ngrid)/(Ngrid-1)

value_v = Parallel(n_jobs=-1)(delayed(SDP_grid)(M, Ngrid, pgrid0[i])
                                          for i in range(Ngrid))
val = []
for i in range(len(value_v)):
    val += value_v[i]

time_elapsed = datetime.now() - start_time

print("Time elapsed in hh:mm:ss =", time_elapsed)
print("Best approximation of nu* with a grid-point distribution:", min(val))
print("Error estimate:", 6/Ngrid)
