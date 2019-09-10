"""
Library for extracting interesting quantites from autograd.

Not thread-safe because of module-level variables affecting state of autograd

Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
s: spatial dimension (Oh*Ow)
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)

Hi: per-example Hessian
    Linear layer: shape [do*di, do*di]
    Conv2d layer: shape [do*di*Kh*Kw, do*di*Kh*Kw]
Hi_bias: per-example Hessian of bias
H: mean Hessian of matmul
H_bias: mean Hessian of bias


Jo: batch output Jacobian of matmul, gradient of output for each output,example pair, [o, n, ....]
Jo_bias: output Jacobian of bias

A, activations: inputs into matmul
    Linear: [n, di]
    Conv2d: [n, di, Ih, Iw] -> (unfold) -> [n, di, Oh, Ow]
B, backprops: backprop values (aka Lop aka Jacobian-vector product) for current layer
    Linear: [n, do]
    Conv2d: [n, do, Oh, Ow]

weight: matmul part of layer, Linear [di, do], Conv [do, di, Kh, Kw]

"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import util as u
import globals as gl

_supported_layers = ['Linear', 'Conv2d']  # Supported layer class types  TODO(y): make non-private
_supported_methods = ['exact', 'kron', 'mean_kron', 'experimental_kfac']  # supported approximation methods
_supported_losses = ['LeastSquares', 'CrossEntropy']

# module-level variables affecting state of autograd
_global_hooks_disabled: bool = False  # work-around for https://github.com/pytorch/pytorch/issues/25723
_global_enforce_fresh_backprop: bool = False  # global switch to catch double backprop errors on Hessian computation
_global_backprops_prefix = ''    # hooks save backprops to, ie param.{_backprops_prefix}backprops_list


def add_hooks(model: nn.Module) -> None:
    """
    Adds hooks to model to save activations and backprop values.

    The hooks will
    1. assign activations to layer.activations during forward pass
    2. assign layer output to layer.output during forward pass
    2. append backprops to layer.backprops_list during backward pass

    Call "clear_backprops" to clear backprops_list values for all parameters in the model
    Call "remove_hooks(model)" to undo this operation.

    Args:
        model:
    """

    global _global_hooks_disabled
    _global_hooks_disabled = False

    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            handles.append(layer.register_forward_hook(_capture_activations))
            handles.append(layer.register_forward_hook(_capture_output))
            handles.append(layer.register_backward_hook(_capture_backprops))

    model.__dict__.setdefault('autograd_hacks_hooks', []).extend(handles)


def remove_hooks(model: nn.Module) -> None:
    """
    Remove hooks added by add_hooks.
    """

    assert model == 0, "not working, remove this after fix to https://github.com/pytorch/pytorch/issues/25723"

    if not hasattr(model, 'autograd_hacks_hooks'):
        print("Warning, asked to remove hooks, but no hooks found")
    else:
        for handle in model.autograd_hacks_hooks:
            handle.remove()
        del model.autograd_hacks_hooks


def disable_hooks() -> None:
    """
    Globally disable all hooks installed by this library.
    """

    global _global_hooks_disabled
    _global_hooks_disabled = True


def enable_hooks() -> None:
    """The opposite of disable_hooks()."""

    global _global_hooks_disabled
    _global_hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported."""

    return _layer_type(layer) in _supported_layers


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _global_hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "activations", input[0].detach())


def _capture_output(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _global_hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "output", output.detach())


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _global_enforce_fresh_backprop

    if _global_hooks_disabled:
        return

    backprops_list_attr = _global_backprops_prefix + 'backprops_list'
    if _global_enforce_fresh_backprop:
        assert not hasattr(layer, backprops_list_attr), f"Seeing result of previous backprop in {backprops_list_attr}, use {_global_backprops_prefix}clear_backprops(model) to clear"
        _global_enforce_fresh_backprop = False

    if not hasattr(layer, backprops_list_attr):
        setattr(layer, backprops_list_attr, [])
    getattr(layer, backprops_list_attr).append(output[0].detach())


def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list


def clear_hess_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'hess_backprops_list'):
            del layer.hess_backprops_list


def compute_grad1(model: nn.Module, loss_type: str = 'mean') -> None:
    """
    Compute per-example gradients and save them under 'param.grad1'. Must be called after loss.backprop()

    Args:
        model:
        loss_type: either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
    """

    assert loss_type in ('sum', 'mean')
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"
        assert len(layer.backprops_list) == 1, "Multiple backprops detected, make sure to call clear_backprops(model)"

        A = layer.activations
        n = A.shape[0]
        if loss_type == 'mean':
            B = layer.backprops_list[0] * n
        else:  # loss_type == 'sum':
            B = layer.backprops_list[0]

        if layer_type == 'Linear':
            setattr(layer.weight, 'grad1', torch.einsum('ni,nj->nij', B, A))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', B)

        elif layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels
            Oh, Ow = layer.backprops_list[0].shape[2:]
            weight_shape = [n] + list(layer.weight.shape)  # n, do, di, Kh, Kw
            assert weight_shape == [n, do, di, Kh, Kw]
            A = torch.nn.functional.unfold(A, layer.kernel_size)  # n, di * Kh * Kw, Oh * Ow

            assert A.shape == (n, di * Kh * Kw, Oh * Ow)
            assert layer.backprops_list[0].shape == (n, do, Oh, Ow)

            # B = B.reshape(n, -1, A.shape[-1])
            B = B.reshape(n, do, Oh * Ow)
            grad1 = torch.einsum('ijk,ilk->ijl', B, A)  # n, do, di * Kh * Kw
            assert grad1.shape == (n, do, di * Kh * Kw)

            setattr(layer.weight, 'grad1', grad1.reshape(weight_shape))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', torch.sum(B, dim=2))


def compute_hess(model: nn.Module, method='exact', attr_name=None) -> None:
    """Compute Hessian (torch.Tensor) for each parameter and save it under 'param.hess or param.hess_factored'.

    If method is exact, saves Tensor hessian under param.hess. Otherwise save u.FactoredMatrix instance under param.hess_factored

    method: which method to use for factoring
      kron: kronecker product
      mean_kron: mean of kronecker products, one kronecker product per datapoint
      experimental_kfac: experimental method for Conv2d
    field_name: If None, will save hessian to "hess" for exact computation and to "hess_factored" for factored, otherwise use this attr name

    Must be called after backprop_hess().
    """

    assert method in _supported_methods

    # TODO: get rid of factored flag

    if attr_name is None:
        hess_attr = 'hess' if (method == 'exact' or method == 'autograd') else 'hess_factored'
    else:
        hess_attr = attr_name

    li = 0
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'hess_backprops_list'), "No backprops detected, run hess_backprop"

        if layer_type == 'Linear':
            A = layer.activations
            B = torch.stack(layer.hess_backprops_list)

            n = A.shape[0]
            di = A.shape[1]
            do = layer.hess_backprops_list[0].shape[1]
            o = B.shape[0]

            A = torch.stack([A] * o)

            if method == 'exact':
                Jo = torch.einsum("oni,onj->onij", B, A).reshape(n * o, -1)
                H = torch.einsum('ni,nj->ij', Jo, Jo) / n

                # Alternative way
                # Jo = torch.einsum("oni,onj->onij", B, A)
                # H = torch.einsum('onij,onkl->ijkl', Jo, Jo) / n
                # H = H.reshape(do*di, do*di)

                H_bias = torch.einsum('oni,onj->ij', B, B) / n
            else:  # TODO(y): can optimize this case by not stacking A
                assert method == 'kron'
                AA = torch.einsum("oni,onj->ij", A, A) / (o * n)  # remove factor of o because A is repeated o times
                BB = torch.einsum("oni,onj->ij", B, B) / n
                H = u.KronFactored(AA, BB)
                H_bias = u.KronFactored(torch.eye(1), torch.einsum("oni,onj->ij", B, B) / n)  # TODO: reuse BB
                setattr(layer.bias, 'hess_factored', H_bias)

        elif layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels
            n, do, Oh, Ow = layer.hess_backprops_list[0].shape
            o = len(layer.hess_backprops_list)

            A = layer.activations
            A = torch.nn.functional.unfold(A, kernel_size=layer.kernel_size,
                                           stride=layer.stride,
                                           padding=layer.padding,
                                           dilation=layer.dilation)  # n, di * Kh * Kw, Oh * Ow
            assert A.shape == (n, di * Kh * Kw, Oh * Ow)
            B = torch.stack([Bh.reshape(n, do, -1) for Bh in layer.hess_backprops_list])  # o, n, do, Oh*Ow

            A = torch.stack([A] * o)  # o, n, di * Kh * Kw, Oh*Ow
            if gl.debug_dump_stats:
                print(f'layerA {li}', A)
                print(f'layerB {li}', B)

            if method == 'exact':
                Jo = torch.einsum('onis,onks->onik', B, A)  # o, n, do, di * Kh * Kw
                Jo_bias = torch.einsum('onis->oni', B)

                Hi = torch.einsum('onij,onkl->nijkl', Jo, Jo)  # n, do, di*Kh*Kw, do, di*Kh*Kw
                Hi = Hi.reshape(n, do * di * Kh * Kw, do * di * Kh * Kw)  # n, do*di*Kh*Kw, do*di*Kh*Kw
                Hi_bias = torch.einsum('oni,onj->nij', Jo_bias, Jo_bias)  # n, do, do
                H = Hi.mean(dim=0)
                H_bias = Hi_bias.mean(dim=0)
            elif method == 'kron':
                AA = torch.einsum("onis->oni", A) / (Oh * Ow)  # group input channels
                AA = torch.einsum("oni,onj->onij", AA, AA) / (o * n)  # remove factor of o because A is repeated o times

                AA = torch.einsum("onij->ij", AA)  # sum out outputs/classes

                BB = torch.einsum("onip->oni", B)  # group output channels
                BB = torch.einsum("oni,onj->ij", BB, BB) / n
            elif method == 'mean_kron':
                AA = torch.einsum("onis->oni", A) / (Oh * Ow)  # group input channels
                AA = torch.einsum("oni,onj->onij", AA, AA) / (o)  # remove factor of o because A is repeated o times

                AA = torch.einsum("onij->nij", AA)  # sum out outputs/classes

                BB = torch.einsum("onip->oni", B)  # group output channels
                BB = torch.einsum("oni,onj->nij", BB, BB)

            elif method == 'experimental_kfac':
                AA = torch.einsum("onis,onjs->onijs", A, A)
                AA = torch.einsum("onijs->onij", AA) / (Oh * Oh)
                AA = torch.einsum("onij->oij", AA) / n
                AA = torch.einsum("oij->ij", AA) / o

                BB = torch.einsum("onip,onjp->onijp", B, B) / n
                BB = torch.einsum("onijp->onij", BB)
                BB = torch.einsum("onij->nij", BB)
                BB = torch.einsum("nij->ij", BB)

            if method != 'exact':
                if method == 'mean_kron':
                    H = u.MeanKronFactored(AA, BB)
                    # H = u.KronFactored(AA[0,...], BB[0,...])
                else:
                    H = u.KronFactored(AA, BB)

                BB_bias = torch.einsum("onip->oni", B)  # group output channels
                BB_bias = torch.einsum("oni,onj->onij", BB_bias, BB_bias) / n  # covariance
                BB_bias = torch.einsum("onij->ij", BB_bias)  # sum out outputs + examples
                H_bias = u.KronFactored(torch.eye(1), BB_bias)

        setattr(layer.weight, hess_attr, H)
        if layer.bias is not None:
            setattr(layer.bias, hess_attr, H_bias)
        li+=1


def backprop_hess(output: torch.Tensor, hess_type: str) -> None:
    """
    Call backprop 1 or more times to accumulate values needed for Hessian computation.

    Values are accumulated under .backprops_list attr of each layer and used by downstream functions like compute_hess

    Args:
        output: prediction of neural network (ie, input of nn.CrossEntropyLoss())
        hess_type: type of Hessian propagation, "CrossEntropy" results in exact Hessian for CrossEntropy

    """

    global _global_enforce_fresh_backprop, _global_hooks_disabled, _global_backprops_prefix

    assert not _global_hooks_disabled
    _global_enforce_fresh_backprop = True  # enforce empty backprops_list on first backprop

    old_backprops_prefix = _global_backprops_prefix
    _global_backprops_prefix = 'hess_'    # backprops go into hess_backprops_list

    valid_hess_types = ('LeastSquares', 'CrossEntropy', 'DebugLeastSquares')
    assert hess_type in valid_hess_types, f"Unexpected hessian type: {hess_type}, valid types are {valid_hess_types}"
    n, o = output.shape

    if hess_type == 'CrossEntropy':
        batch = F.softmax(output, dim=1)

        mask = torch.eye(o).expand(n, o, o)
        diag_part = batch.unsqueeze(2).expand(n, o, o) * mask
        outer_prod_part = torch.einsum('ij,ik->ijk', batch, batch)
        hess = diag_part - outer_prod_part
        assert hess.shape == (n, o, o)

        for i in range(n):
            if torch.get_default_dtype() == torch.float64:
                hess[i, :, :] = u.symsqrt_svd(hess[i, :, :])  # more stable method since we don't care about speed with float64
            else:
                hess[i, :, :] = u.symsqrt(hess[i, :, :])
        hess = hess.transpose(0, 1)

    elif hess_type == 'LeastSquares':
        hess = []
        assert len(output.shape) == 2
        batch_size, output_size = output.shape

        id_mat = torch.eye(output_size)
        for out_idx in range(output_size):
            hess.append(torch.stack([id_mat[out_idx]] * batch_size))

    elif hess_type == 'DebugLeastSquares':
        hess = []
        assert len(output.shape) == 2
        batch_size, output_size = output.shape

        id_mat = torch.eye(output_size)
        id_mat[0, 0] = 10
        for out_idx in range(output_size):
            hess.append(torch.stack([id_mat[out_idx]] * batch_size))

    for o in range(o):
        output.backward(hess[o], retain_graph=True)

    _global_backprops_prefix = old_backprops_prefix

