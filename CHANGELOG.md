# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## 0.1.0

Initial release.

### Added

- `LightningBoostRegressor`, `LightningBoostClassifier`, and `LightningBoostSurvival` estimators with full scikit-learn compatibility
- 12 distributions: Normal, LogNormal, Exponential, Gamma, Poisson, Laplace, StudentT, Weibull, HalfNormal, Cauchy, Bernoulli, and Categorical
- LogScore (NLL), CRPScore (CRPS), and CensoredLogScore scoring rules
- Right-censored survival analysis via `LightningBoostSurvival`
- Evaluation utilities: PIT values, calibration curves, concordance index, and plotting helpers
- Early stopping with validation data
- Feature importances per distribution parameter
- Sample weights, column subsampling, staged predictions, custom loss monitors
- UCI regression benchmark suite reproducing NGBoost paper Table 1
