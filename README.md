# A posteriori existence for the Keller-Segel model via a Finite Volume scheme

This repository contains the code used to produce the numerical results presented in Section 7 of the article

```bibtex
@online{giesselmann2026aposteriori,
  title={{A} posteriori existence for the {K}eller-{S}egel model via a finite volume - finite element scheme},
  author={Giesselmann, Jan and Hoffmann, Marc},
  year={2026},
  month={06},
  eprint={2509.17710v1},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

If you find these results useful, please cite either the article mentioned above or the corresponding reproduction code, as appropriate.

```bibtex
@software{giesselmann2026code,
  title        = {{R}eproduction code for "{A} posteriori existence for the {K}eller-{S}egel model via a finite volume - finite element scheme"},
  author={Giesselmann, Jan and Hoffmann, Marc},
  year         = {2026},
  publisher    = {Zenodo},
  url          = {https://doi.org/10.5281/zenodo.20425488}
}
```

## Abstract

We derive two forms of conditional a posteriori error estimates for a finite volume scheme approximating the parabolic-elliptic Keller-Segel system. 
The estimates control the error in the L∞(0,T;L²(Ω))- and L²(0,T;H¹(Ω))-norm and exhibit linear convergence in the mesh size, as observed in numerical experiments. 
Crucially, we show that as long as the condition of the error estimate is satisfied a weak solution exists. 
This means, as long as the numerical solution has good properties, we can rigorously infer existence of an exact solution.


## Numerical experiments

The numerical experiments where implemented using Python 3.8.10.

In each folder, `2D` and `3D`, there is a respective `README.md` file guiding you through the code and explaining how to reproduce the results from the article.


## Authors

- Jan Giesselmann (TU Darmstadt, Germany)
- Marc Hoffmann (TU Darmstadt, Germany)


## License

The code in this repository is published under the MIT license, see the
`LICENSE` file.


## Disclaimer

Everything is provided as is and without warranty. Use at your own risk!
