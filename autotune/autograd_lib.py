"""
Library for extracting interesting quantites from autograd.

Not thread-safe because of module-level variables

Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)

Hi: per-example Hessian
    Linear layer: shape [do*di, do*di]
    Conv2d layer: shape [do*di*Kh*Kw, do*di*Kh*Kw]
Hi_bias: per-example Hessian of bias
H: mean Hessian of matmul
H_bias: mean Hessian of bias


Jb: batch output Jacobian of matmul, output sensitivity for each output,example pair, [o, n, ....]
Jb_bias: output Jacobian of bias

A, activations: inputs into matmul
    Linear: [n, di]
    Conv2d: [n, di, Ih, Iw] -> (unfold) -> [n, di, Oh, Ow]
B, backprops: backprop values (aka Lop aka Jacobian-vector product) for current layer
    Linear: [n, do]
    Conv2d: [n, do, Oh, Ow]

"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import util as u

_supported_layers = ['Linear', 'Conv2d']  # Supported layer class types
_hooks_disabled: bool = False           # work-around for https://github.com/pytorch/pytorch/issues/25723
_enforce_fresh_backprop: bool = False   # global switch to catch double backprop errors on Hessian computation


def add_hooks(model: nn.Module) -> None:
    """
    Adds hooks to model to save activations and backprop values.

    The hooks will
    1. assign activations to param.activations during forward pass
    2. append backprops to param.backprops_list during backward pass

    Call "clear_backprops" to clear backprops_list values for all parameters in the model
    Call "remove_hooks(model)" to undo this operation.

    Args:
        model:
    """

    global _hooks_disabled
    _hooks_disabled = False

    handles = []
    for layer in model.modules():
        if _layer_type(layer) in _supported_layers:
            handles.append(layer.register_forward_hook(_capture_activations))
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

    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks() -> None:
    """The opposite of disable_hooks()."""

    global _hooks_disabled
    _hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported."""

    return _layer_type(layer) in _supported_layers


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__


def _capture_activations(layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    setattr(layer, "activations", input[0].detach())


def _capture_backprops(layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""
    global _enforce_fresh_backprop

    if _hooks_disabled:
        return

    if _enforce_fresh_backprop:
        assert not hasattr(layer, 'backprops_list'), "Seeing result of previous backprop, use clear_backprops(model) to clear"
        _enforce_fresh_backprop = False

    if not hasattr(layer, 'backprops_list'):
        setattr(layer, 'backprops_list', [])
    layer.backprops_list.append(output[0].detach())


def clear_backprops(model: nn.Module) -> None:
    """Delete layer.backprops_list in every layer."""
    for layer in model.modules():
        if hasattr(layer, 'backprops_list'):
            del layer.backprops_list


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
            A = torch.nn.functional.unfold(A, layer.kernel_size)
            B = B.reshape(n, -1, A.shape[-1])
            grad1 = torch.einsum('ijk,ilk->ijl', B, A)
            shape = [n] + list(layer.weight.shape)
            setattr(layer.weight, 'grad1', grad1.reshape(shape))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', torch.sum(B, dim=2))


def compute_hess(model: nn.Module, kron=False) -> None:
    """Compute Hessian (torch.Tensor) for each parameter and save it under 'param.hess'.

    If kron is True, instead compute Kronecker-factored Hessian (KronFactored), save under 'param.hess_kron'

    Must be called after backprop_hess().
    """

    hess_attr = 'hess' if not kron else 'hess_kron'
    for layer in model.modules():
        layer_type = _layer_type(layer)
        if layer_type not in _supported_layers:
            continue
        assert hasattr(layer, 'activations'), "No activations detected, run forward after add_hooks(model)"
        assert hasattr(layer, 'backprops_list'), "No backprops detected, run backward after add_hooks(model)"

        if layer_type == 'Linear':
            A = layer.activations
            B = torch.stack(layer.backprops_list)

            n = A.shape[0]
            di = A.shape[1]
            do = layer.backprops_list[0].shape[1]
            o = B.shape[0]

            A = torch.stack([A] * o)

            if not kron:
                Jb = torch.einsum("oni,onj->onij", B, A).reshape(n*o,  -1)
                H = torch.einsum('ni,nj->ij', Jb, Jb) / n

                # Alternative way
                # Jb = torch.einsum("oni,onj->onij", B, A)
                # H = torch.einsum('onij,onkl->ijkl', Jb, Jb) / n
                # H = H.reshape(do*di, do*di)

                H_bias = torch.einsum('oni,onj->ij', B, B)/n
            else:
                # TODO(y): can optimize this case by not stacking A
                AA = torch.einsum("oni,onj->ij", A, A) / (o * n)  # remove factor of o because A is repeated o times
                BB = torch.einsum("oni,onj->ij", B, B) / n
                H = u.KronFactored(AA, BB)
                H_bias = u.KronFactored(torch.eye(1), torch.einsum("oni,onj->ij", B, B) / n) # TODO: reuse BB
                setattr(layer.bias, 'hess_kron', H_bias)

        elif layer_type == 'Conv2d':
            Kh, Kw = layer.kernel_size
            di, do = layer.in_channels, layer.out_channels

            A = layer.activations.detach()
            A = torch.nn.functional.unfold(A, (Kh, Kw))       # n, di * Kh * Kw, Oh * Ow
            n = A.shape[0]
            B = torch.stack([Bt.reshape(n, do, -1) for Bt in layer.backprops_list])  # o, n, do, Oh*Ow
            o = B.shape[0]

            A = torch.stack([A] * o)                          # o, n, di * Kh * Kw, Oh*Ow

            if not kron:
                Jb = torch.einsum('onij,onkj->onik', B, A)        # o, n, do, di * Kh * Kw
                Jb_bias = torch.einsum('onij->oni', B)

                Hi = torch.einsum('onij,onkl->nijkl', Jb, Jb)     # n, do, di*Kh*Kw, do, di*Kh*Kw
                Hi_bias = torch.einsum('oni,onj->nij', Jb_bias, Jb_bias)
                H = Hi.mean(dim=0)
                H_bias = Hi_bias.mean(dim=0)

            else:
                AA = torch.einsum("onic,onjc->ij", A, A) / (o * n)  # remove factor of o because A is repeated o times
                BB = torch.einsum("onic,onjc->ij", B, B) / n
                H = u.KronFactored(AA, BB)
                H_bias = u.KronFactored(torch.eye(1), BB)

        setattr(layer.weight, hess_attr, H)
        if layer.bias is not None:
            setattr(layer.bias, hess_attr, H_bias)


def backprop_hess(output: torch.Tensor, hess_type: str) -> None:
    """
    Call backprop 1 or more times to accumulate values needed for Hessian computation.

    Values are accumulated under .backprops_list attr of each layer and used by downstream functions (ie, compute_hess, compute_hess_kron)

    Args:
        output: prediction of neural network (ie, input of nn.CrossEntropyLoss())
        hess_type: type of Hessian propagation, "CrossEntropy" results in exact Hessian for CrossEntropy

    """

    global _enforce_fresh_backprop, _hooks_disabled

    assert not _hooks_disabled
    _enforce_fresh_backprop = True   # enforce empty backprops_list on first backprop

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
                hess[i, :, :] = u.symsqrt_svd(hess[i, :, :])  # use more stable method for numerical debugging
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

