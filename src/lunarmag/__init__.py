"""lunarmag: rover magnetometry of magnetized lunar bedrock.

Code for C. A. Zareck, *SM thesis, MIT Department of Aeronautics and Astronautics* (2026), built on the
reproduction code of the companion paper (Narrett, Zareck, Weiss & Hoffman, in preparation).

Subpackages and modules
-----------------------
core, inversion_helpers, isolated_outcrop_1cm, correlated_noise_sensitivity
    The simulation engine of release v1.0.0 (relocated from ``engine/``; numerics unchanged).
physics
    Closed-form field, magnetization and error-metric formulas (thesis Eqs. 3.2, 3.3, 3.6-3.8, 3.27-3.29).
estimators
    Least-squares estimator, covariance and nuisance-parameter designs (thesis Eqs. 3.11-3.26).
spectral
    Wavenumber band of the estimator weights (thesis Section 5.1).
plotting, figures
    The thesis figure style and one module per thesis figure (3.1-5.2).
pdfa
    PDF/A-1b helpers for matplotlib figures (no transparency, CIDSet for embedded fonts).
manifest, cli
    The figure manifest (figure -> script -> result file -> sha256) and the ``lunarmag`` command.

Importing the package itself is cheap; SimPEG is imported only by the engine modules.
"""
__version__ = "1.1.0.dev0"
__all__ = ["__version__"]
