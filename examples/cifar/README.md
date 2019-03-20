To run training LeNet-5 for CIFAR-10 classification
```bash
cd pytorch-curv/examples/cifar
python main.py --config={config file path}
```
| optimizer | config file path |
| --- | --- |
| [Adam](https://arxiv.org/abs/1412.6980) | [configs/lenet_adam.json](https://github.com/rioyokotalab/pytorch-curv/blob/master/examples/cifar/configs/lenet_adam.json) |
| [K-FAC](https://arxiv.org/abs/1503.05671) | [configs/lenet_kfac.json](https://github.com/rioyokotalab/pytorch-curv/blob/master/examples/cifar/configs/lenet_kfac.json) |
| [Noisy K-FAC](https://arxiv.org/abs/1712.02390) | [configs/lenet_noisykfac.json](https://github.com/rioyokotalab/pytorch-curv/blob/master/examples/cifar/configs/lenet_noisykfac.json) |
| [VOGN](https://arxiv.org/abs/1806.04854) | [configs/lenet_vogn.json](https://github.com/rioyokotalab/pytorch-curv/blob/master/examples/cifar/configs/lenet_vogn.json) |
