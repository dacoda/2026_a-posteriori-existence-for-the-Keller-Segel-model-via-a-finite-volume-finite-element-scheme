# Numerical Experiments

This directory contains the code that was used to produce the results from subsection 7.2 of the article mentioned in the `README.md` file outside this folder.
In the header of each file you can find additional information on what this piece of code does.
As the name of the folder suggests, we are working in three space dimensions. 

The code was developed with Python 3.8.10. To reproduce the results you can follow the guide in the sequel.

1) Clone this git repository into your favorite directory and choose a path, where to store the numerical results and auxiliary files that will be produced on the way. Replace all strings `'put_some_path_here'` occurring in the code by a path that suits you. We recommend using the hotkey `ctrl + f`.
2) Generate well-centered tetrahedral meshes for different mesh sizes using the file `generate_meshes.py`.
3) Use Algorithm 4.3, implemented in `FVFEscheme.py`, to obtain numerical approximations and the respective interpolation of Morley-type to the solution to the Keller-Segel system.
4) Compute the 'exact error' measured in the L∞(0,T;L²(Ω))- and L²(0,T;H¹(Ω))-norm and corresponding EOCs using `manuf_error.py` and compute the a posteriori residual estimator (27), see Theorem 6.8, and corresponding EOCs using `error_estimator.py`.

Processing this code will take a while. To speed up the computations, we recommend replacing at least the outermost for-loop in each file by parallel computing.
Very fine meshes can result in very large files, that will be stored on your hard drive.