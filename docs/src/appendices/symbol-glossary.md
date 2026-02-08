# Appendix: Symbol Glossary

## Core EM Symbols

- ``c_0``: speed of light in vacuum
- ``\mu_0,\epsilon_0``: free-space permeability/permittivity
- ``\eta_0=\sqrt{\mu_0/\epsilon_0}``: free-space impedance
- ``f``: frequency (Hz)
- ``\omega=2\pi f``: angular frequency
- ``k=\omega\sqrt{\mu_0\epsilon_0}``: wavenumber

## Geometry Symbols

- ``\Gamma``: scattering surface
- ``\hat{\mathbf n}``: unit surface normal
- ``\mathbf r,\mathbf r'``: observation/source points
- ``R=|\mathbf r-\mathbf r'|``

## Discretization Symbols

- ``N_v``: number of vertices
- ``N_t``: number of triangles
- ``N``: number of RWG unknowns
- ``\mathbf f_n``: RWG basis function
- ``\mathbf I\in\mathbb C^N``: current coefficient vector
- ``A_n^\pm,\ell_n``: RWG triangle areas and shared-edge length

## System and Objective Symbols

- ``\mathbf Z``: MoM matrix
- ``\mathbf v``: excitation vector
- ``\mathbf Q``: quadratic objective matrix
- ``J``: scalar objective value
- ``\boldsymbol\lambda``: adjoint vector
- ``\boldsymbol\theta\in\mathbb R^P``: design parameters
- ``\mathbf M_p``: patch mass matrix for parameter ``p``

## Far-Field and RCS Symbols

- ``\hat{\mathbf r}``: observation direction
- ``\mathbf E^\infty(\hat{\mathbf r})``: far-field vector
- ``\theta,\phi``: spherical angles
- ``\sigma``: radar cross section (m²)

## Validation Terms

- FD: finite difference
- CS: complex step
- RMSE: root mean square error
- SLL: sidelobe level (or sidelobe-level suppression, context dependent)
