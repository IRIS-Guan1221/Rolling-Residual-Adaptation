# Third-party methods

This distribution contains the RRA implementation. It does not redistribute
pretrained weights or third-party library sources. Dependencies are installed
from their respective distributions and remain subject to their licenses.

- Chronos-2: Ansari et al., *Chronos-2: From Univariate to Universal Forecasting*,
  2025. https://arxiv.org/abs/2510.15821
  Software: https://github.com/amazon-science/chronos-forecasting
  Weights: https://huggingface.co/amazon/chronos-2
  Revision: `29ec3766d36d6f73f0696f85560a422f50e8498c`.
- LightGBM: Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision
  Tree*, Advances in Neural Information Processing Systems 30, 2017.
  Software: https://github.com/microsoft/LightGBM

Additional dependencies are NumPy, pandas, SciPy, scikit-learn, threadpoolctl,
and, for Chronos inference, PyTorch and the Chronos package dependencies.
