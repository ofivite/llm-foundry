import os
import torch

__all__ = [
    'Muon',
]

def zeropower_via_svd(G, steps=None, dual_norm_scaling=False, **kwargs):
    U, S, V = G.svd()
    X = U @ V.T
    if dual_norm_scaling:
        # https://x.com/leloykun/status/1874358290093924849
        X = torch.einsum('ij,ij,ab->ab', G.type_as(X), X, X)  # Adaptive scaling,`(G * X).sum() * X` == (G.T @ X).trace() * X

    return X

def l1_to_rms_norm(W):
    norm = torch.max(torch.norm(W.data.to(torch.float32), p=2, dim=0, dtype=torch.float32))
    scale = torch.sqrt(torch.tensor(W.data.shape[0], dtype=W.dtype, device=W.device))
    norm /= scale
    return norm

def rms_to_l1_norm(W):
    norm = torch.max(torch.norm(W.data.to(torch.float32), p=2, dim=1, dtype=torch.float32))
    scale = torch.sqrt(torch.tensor(W.data.shape[1], dtype=W.dtype, device=W.device))
    norm *= scale
    return norm
    
def duality_map_L1_to_RMS_embed(G, steps=None, **kwargs):
    """
    Computes the duality map for the L1 -> RMS induced operator norm for a matrix G.
    
    Given G of shape (m, n), the function returns a matrix T of the same shape,
    where T is zero in every column except for the column j* that maximizes the L2 norm.
    That column is set to sqrt(m) * (G[:, j*] / ||G[:, j*]||_2).
    
    Args:
        G (torch.Tensor): Input matrix of shape (m, n).
        
    Returns:
        torch.Tensor: The dual matrix T of shape (m, n) with ||T||_{L1->RMS} = 1.
    """

    G = G.T # SharedEmbedding has shape=(vocab, d_model), need to transpose
    m, n = G.shape

    # Compute the L2 norm of each column
    col_norms = torch.norm(G, p=2, dim=0)  # shape (n,)
    
    # Find the column index with maximum L2 norm
    j_star = torch.argmax(col_norms)
    
    # Get the column vector corresponding to j_star
    col = G[:, j_star]
    
    # Normalize the selected column (avoid division by zero)
    col_norm = torch.norm(col, p=2)
    if col_norm.item() == 0:
        # If the column norm is zero, return the zero matrix.
        return torch.zeros_like(G)
    
    col_normalized = col / col_norm

    # Create the output matrix T: all zeros except in column j_star.
    T = torch.zeros_like(G)
    # Multiply by sqrt(m) so that the RMS norm of the nonzero column becomes:
    # RMS_norm = (1/sqrt(m)) * || sqrt(m)*(col/||col||) ||_2 = 1.
    T[:, j_star] = torch.sqrt(torch.tensor(m, dtype=G.dtype, device=G.device)) * col_normalized
    
    T = T.T # Transpose back to original shape
    return T


def duality_map_RMS_to_L1(G, steps=None, **kwargs):
    """
    Computes the duality map for the RMS -> L1 induced operator norm for a matrix G.
    
    For a matrix G of shape (m, n), the duality map returns a matrix T of the same shape,
    where T is zero in every row except for the row i* that maximizes the L2 norm of the row.
    That row is set to (1/sqrt(n)) * (G[i*, :] / ||G[i*, :]||_2) so that ||T||_{RMS->L1} = 1.
    
    Args:
        G (torch.Tensor): Input matrix of shape (m, n).
        
    Returns:
        torch.Tensor: The dual matrix T of shape (m, n) for the RMS -> L1 norm.
    """
    m, n = G.shape

    # Compute the L2 norm of each row (over the columns)
    row_norms = torch.norm(G, p=2, dim=1)  # shape: (m,)
    
    # Find the index of the row with the maximum L2 norm
    i_star = torch.argmax(row_norms)
    
    # Extract the selected row
    row = G[i_star, :]
    
    # Avoid division by zero: if the row is all zeros, return a zero matrix.
    row_norm = torch.norm(row, p=2)
    if row_norm.item() == 0:
        return torch.zeros_like(G)
    
    # Normalize the selected row
    row_normalized = row / row_norm
    
    # Create an output matrix T of the same shape as G (initialized with zeros)
    T = torch.zeros_like(G)
    
    # Set the selected row to (1/sqrt(n)) * normalized row.
    # This scaling ensures that when the RMS norm on the input is defined as (1/sqrt(n)) * ||.||_2,
    # the resulting operator has norm 1.
    scale = 1.0 / torch.sqrt(torch.tensor(n, dtype=G.dtype, device=G.device))
    T[i_star, :] = scale * row_normalized

    return T

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, dual_norm_scaling=False, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    # print('\n\n\n')
    # print(G.shape)
    # print('\n\n\n')
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    
    if dual_norm_scaling:
        # https://x.com/leloykun/status/1874358290093924849
        X = torch.einsum('ij,ij,ab->ab', G.type_as(X), X, X)  # Adaptive scaling,`(G * X).sum() * X` == (G.T @ X).trace() * X

    return X

zeropower_backends = dict(svd=zeropower_via_svd, 
                            newtonschulz5=zeropower_via_newtonschulz5, 
                            identity=lambda x, **kwargs: x,
                            dual_embed=duality_map_L1_to_RMS_embed,
                            dual_unembed=duality_map_RMS_to_L1)

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """

    metric_functions = {
        # 'l2_norm/moment': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg']),
        'l2_norm/param': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.data),
        'l2_norm/update': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(step_tensor),
        # 'l2_norm/grad': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.grad),
        'l1_norm/param': lambda param, optim_state, step_tensor: torch.abs(param.data).mean(),
        'l1_norm/update': lambda param, optim_state, step_tensor: torch.abs(step_tensor).mean(),
        'spectral_norm/param': lambda param, optim_state, step_tensor: torch.linalg.norm(param.data.to(torch.float32), ord=2, dtype=torch.float32),
        'spectral_norm/update': lambda param, optim_state, step_tensor: torch.linalg.norm(step_tensor.to(torch.float32), ord=2, dtype=torch.float32),
        'l1_to_rms/param': lambda param, optim_state, step_tensor: l1_to_rms_norm(param),
        'l1_to_rms/update': lambda param, optim_state, step_tensor: l1_to_rms_norm(step_tensor),
        'rms_to_l1/param': lambda param, optim_state, step_tensor: rms_to_l1_norm(param),
        'rms_to_l1/update': lambda param, optim_state, step_tensor: rms_to_l1_norm(step_tensor),
    }

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, dual_norm_scaling=False, eps=1e-7, norm_factor='none',
                 backend='newtonschulz5', backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, 
                        dual_norm_scaling=dual_norm_scaling, eps=eps, norm_factor=norm_factor, 
                        backend=backend, backend_steps=backend_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            eps = group['eps']
            zeropower_backend = zeropower_backends[group['backend']]
            backend_steps = group['backend_steps']
            norm_factor = group['norm_factor']
            nesterov = group['nesterov']
            dual_norm_scaling = group['dual_norm_scaling']
            
            for _, p in enumerate(group['params']):
                g = p.grad
                if g is None or not p.requires_grad:
                    continue

                # State initialization
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = torch.zeros((), dtype=torch.float, device=p.device)
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                if 'update' not in state:
                    state['update'] = torch.zeros_like(g)

                # Compute updated gradient
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_backend(g, steps=backend_steps, dual_norm_scaling=dual_norm_scaling, eps=eps)
                if norm_factor == 'linear':
                    # print('\n\n\n')
                    # print('LINEAR, shape: ', g.shape)
                    # print('\n\n\n')
                    g *= (g.size(0)/g.size(1))**0.5
                elif norm_factor == 'embed':
                    # print('\n\n\n')
                    # print('EMBED, shape: ', g.shape)
                    # print('\n\n\n')
                    g *= torch.rsqrt(g.pow(2).mean(1, keepdim=True) + eps) # + eps maybe
                    g *= g.size(1)**0.5
                elif norm_factor == 'unembed':
                    # print('\n\n\n')
                    # print('UNEMBED, shape: ', g.shape)
                    # print('\n\n\n')
                    g *= torch.rsqrt(g.pow(2).mean(1, keepdim=True) + eps) # + eps maybe
                    g /= g.size(1)**0.5
                elif norm_factor == 'none':
                    pass
                else:
                    raise ValueError(f"Unknown norm_factor: {norm_factor}")

                # Update the parameter
                p.data.add_(g, alpha=-lr)
                state['update'] = g*lr

                # Update the steps for each param group update
                state['step'] += 1

        return loss
    
    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        if param in self.state:
            param_optim_state = self.state[param]
            step_tensor = self.state[param]['update']
            for metric in self.metric_functions:
                if 'wte' in name: 
                    # transpose to align with shape convention [out, in]
                    param.data = param.data.T
                    step_tensor = step_tensor.T
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[metric](
                    param,
                    param_optim_state,
                    step_tensor,
                )

        return optimizer_metrics