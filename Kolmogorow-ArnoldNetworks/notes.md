# [Kolmogorov-Arnold Networks](https://arxiv.org/pdf/2404.19756)
Alternative to MLP where the activation function is a part of the learning.

The code in this folder is entirely based on [code from the paper itself](https://github.com/KindXiaoming/pykan). All the credit for code should go to them, yet they of course should not be held accountable for any mistakes, problems, etc. with code presented here.

## Keypoints:
- The parameters of each layers becomes far greater than in MLPs. In contrast, if used on a suitable problem, KANs is usually much smaller in both width and depth compared to MLPs, making them not only able to properly model the input function, but do so in shorter time.
- Residual activation functions are used to parameterize $\phi$. This is used together with a base function: $b(x) = x/(1+e^(-x))$ 
- 