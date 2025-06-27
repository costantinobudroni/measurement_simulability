# measurement_simulability
This code computes the minimum distance between a measurement assemblage M^eta, consisting of noisy versions of Pauli mesurements, and the set of assemblage that can be simulated by two measurements with a set of preprocessing distribution on a finite grid, i.e., a grid approximation of SIM_2. See the connected paper [arXiv:2506.21223](https://arxiv.org/abs/2506.21223).



## Usage Guide for Boxworld Process Optimization Package

To use this package, follow these steps:

### Installation
First, make sure you have the following Python packages installed:
- `numpy`: for numerical operations
- `cvxpy`: for convex optimization
- `mosek`: for SDP solver
- `joblib`: for parallelization

### Optimization
To run the optimization, first choose the parameter Ngrid in sim_2_grid_sdp.py to fix the number of grid points in every direction.
