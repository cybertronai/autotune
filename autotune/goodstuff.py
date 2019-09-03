
def test_subsampled_hessian():
    """Test backward propagation for subsampled Hessian. Check that quality improves as we use more samples."""
    batch_size = 500

    data_width = 4
    targets_width = 4
    train_steps = 3

    d1 = data_width ** 2
    d2 = 10
    d3 = targets_width ** 2
    o = d3
    n = batch_size
    d = [d1, d2, d3]

    dataset = u.TinyMNIST(data_width=data_width, targets_width=targets_width,
                          dataset_size=batch_size * train_steps)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_iter = iter(trainloader)
    data, targets = next(train_iter)

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == batch_size
        return torch.sum(err * err) / 2 / len(data)

    u.seed_random(1)
    model: u.SimpleModel = u.SimpleFullyConnected(d, nonlin=False)
    
    loss_hessian = u.HessianExactSqrLoss()
    output = model(data)

    for bval in loss_hessian(output):
        output.backward(bval, retain_graph=True)
    model.skip_backward_hooks = True
    i, layer = next(enumerate(model.layers))

    A_t = layer.activations
    Bh_t = layer.backprops_list
    H = u.hessian_from_backprops(A_t, Bh_t)

    # sanity check autograd
    u.clear_backprops(model)
    output = model(data)
    loss = loss_fn(output, targets)
    model.skip_backward_hooks = True

    H_autograd = u.hessian(loss, layer.weight)
    u.check_close(H, H_autograd.reshape(d[i] * d[i + 1], d[i] * d[i + 1]), rtol=1e-4, atol=1e-7)

    # get subsampled Hessian
    u.seed_random(1)
    model = u.SimpleFullyConnected(d, nonlin=False)
    
    loss_hessian = u.HessianSampledSqrLoss(samples=1)
    output = model(data)

    for bval in loss_hessian(output):
        output.backward(bval, retain_graph=True)
    model.skip_backward_hooks = True
    i, layer = next(enumerate(model.layers))

    A_t = layer.activations
    Bh_t = layer.backprops_list
    H_approx1 = u.hessian_from_backprops(A_t, Bh_t)

    # use more samples
    u.seed_random(1)
    model = u.SimpleFullyConnected(d, nonlin=False)
    
    loss_hessian = u.HessianSampledSqrLoss(samples=o)
    output = model(data)

    for bval in loss_hessian(output):
        output.backward(bval, retain_graph=True)
    model.skip_backward_hooks = True
    i, layer = next(enumerate(model.layers))

    A_t = layer.activations
    Bh_t = layer.backprops_list
    H_approx2 = u.hessian_from_backprops(A_t, Bh_t)

    print(abs(u.l2_norm(H)/u.l2_norm(H_approx1)-1))
    print(abs(u.l2_norm(H)/u.l2_norm(H_approx2)-1))

    assert(abs(u.l2_norm(H)/u.l2_norm(H_approx1)-1) < 0.04)
    assert(abs(u.l2_norm(H)/u.l2_norm(H_approx2)-1) < 0.02)
    assert u.kl_div_cov(H_approx1, H) < 0.11   # 0.0673
    assert u.kl_div_cov(H_approx2, H) < 0.03   # 0.0020
