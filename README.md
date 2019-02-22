# PyTorch Curvature
A PyTorch extension for second-order optimization & variational inference in training neural networks.
 
## Optimizers
### `torchcurv.optim.SecondOrderOptimizer` [[source](https://github.com/rioyokotalab/pytorch-curv/blob/master/torchcurv/optim/secondorder.py)]
updates the parameters with the gradients pre-conditioned by the curvature of the loss function (`torch.nn.functional.cross_entropy`) for each `param_group`.
### `torchcurv.optim.VIOptimizer` [[source](https://github.com/rioyokotalab/pytorch-curv/blob/master/torchcurv/optim/vi.py)]
updates the distribution of the parameters by using the curvature as the covariance matrix for each `param_group`.
 
## Curvature Types
You can specify a type of matrix to be used as curvature from the following.
- Hessian [[source](https://github.com/rioyokotalab/pytorch-curv/blob/master/torchcurv/curv/hessian/hessian.py)]
- Gauss-Newton matrix [[source](https://github.com/rioyokotalab/pytorch-curv/blob/master/torchcurv/curv/gn/gn.py)] 
- Fisher information matrix (Empirical Fisher) [[source](https://github.com/rioyokotalab/pytorch-curv/blob/master/torchcurv/curv/fisher/fisher.py)] 

Refer Section 6 of [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838) by L´eon Bottou et al. (2018) for a clear explanation of the second-order optimzation using these matrices as curvature.

## Approximation Methods
You can specify the approximation method(s) of curvature for each layer from the follwing.
1. No approximation
2. Diagonal approximation
3. [K-FAC (Kronecker-Factored Approximate Curvature)](https://arxiv.org/abs/1503.05671)
4. Block-diagonal approximation based on K-FAC
5. Automatic reduction of curvature update frequency 

(5 can be combined with 1, 2, 3 or 4)

## Quick Start
To build the extension run
```
python setup.py install
```
in the root directory of the cloned repository.

To use the extension
```
import torchcurv
```

## Applications
- Image classification
  - MNIST
  - CIFAR-10
  - ImageNet (1,000class)
