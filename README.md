# PyTorch Curvature
A PyTorch extension for second-order optimization in training neural networks.
- provides `torch-curv.SecondOrderOptimizer` for updating the parameters with the gradients pre-conditioned by the curvature of the loss function (`torch.nn.functional.cross_entropy`) *for each layer*
- provides tools for estimating the curvature *for each layer*
 
## Curvature
You can specify a type of matrix to be used as curvature from the following.
- Hessian
- Gauss-Newton matrix
- Fisher information matrix

Refer Section 6 of [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838) by L´eon Bottou et al. (2018) for a clear explanation of the second-order optimzation using these matrices as curvature.

## Estimators
You can specify the estimator(s) of curvature for each layer from the follwing.
1. No approximation
2. Diagonal approximation
3. [K-FAC (Kronecker-Factored Approximate Curvature)](https://arxiv.org/abs/1503.05671)
4. Block-diagonal approximation based on K-FAC
5. Automatic reduction of curvature update frequency 

(5 can be combined with 1, 2, 3 or 4)

## Applications
- Image classification
  - MNIST
  - CIFAR-10
  - ImageNet (1,000class)
