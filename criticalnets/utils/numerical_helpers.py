import torch
import torch.nn.functional as F

def get_activation_derivatives(activation_func, z):
    """
    Compute first and second derivatives of activation function with respect to preactivation.
    
    Args:
        activation_func: The activation function
        z: Preactivation values (tensor)
        
    Returns:
        Tuple of (first_derivative, second_derivative)
    """
    # Ensure z requires gradient
    z = z.detach().clone().requires_grad_(True)
    
    # First forward pass
    y = activation_func(z)
    
    # First derivative: σ'(z)
    first_derivative = torch.autograd.grad(y.sum(), z, create_graph=True)[0]
    
    # Second derivative: σ''(z)
    second_derivative = torch.autograd.grad(first_derivative.sum(), z, create_graph=True)[0]
    
    return first_derivative, second_derivative

def compute_full_jacobian(model, x, activation_func):
    """
    Compute the full Jacobian of σ(Wx+b) with respect to x directly
    
    Args:
        model: The layer (nn.Linear or nn.Conv2d)
        x: Input tensor
        activation_func: The activation function
    
    Returns:
        Jacobian matrix/tensor
    """
    batch_size = x.shape[0]
    x = x.detach().clone().requires_grad_(True)
    
    # Forward pass through the layer
    z = model(x)
    a = activation_func(z)
    
    # Compute gradients for each output dimension
    jacobian = []
    for i in range(a.numel() // batch_size):
        # Create a one-hot vector to select each output element
        mask = torch.zeros_like(a)
        mask.view(batch_size, -1)[:, i] = 1
        
        # Compute gradient
        grad = torch.autograd.grad(a, x, grad_outputs=mask, create_graph=True)[0]
        jacobian.append(grad.view(batch_size, -1))
    
    # Stack gradients to form Jacobian matrix
    jacobian = torch.stack(jacobian, dim=1)
    return jacobian

def compute_full_laplacian(model, x, activation_func):
    """
    Compute the full Laplacian of σ(Wx+b) with respect to x directly
    
    Args:
        model: The layer (nn.Linear or nn.Conv2d)
        x: Input tensor
        activation_func: The activation function
    
    Returns:
        Laplacian (sum of second derivatives)
    """
    # Get Jacobian first
    jacobian = compute_full_jacobian(model, x, activation_func)
    
    # Compute trace of Hessian (Laplacian)
    laplacian = 0
    for i in range(jacobian.shape[1]):
        # Compute second derivative for each element
        grad_i = jacobian[:, i, :]
        laplacian += torch.autograd.grad(grad_i.sum(), x, create_graph=True)[0]
    
    return laplacian

def criticality_regularization(model, x, activation_func, layer_type='conv'):
    """
    Calculate the Edge of Chaos regularization term:
    R(layer) = (2σ′(z)∇²ₓσ(z)/√N) * (1/N - 1/||∇ₓσ(z)||)
    
    Args:
        model: The layer (nn.Linear or nn.Conv2d)
        x: Input tensor
        activation_func: The activation function
        layer_type: Type of layer ('conv' or 'fc')
    
    Returns:
        Regularization loss
    """
    # Forward pass to get preactivation
    z = model(x)
    
    # Calculate derivatives with respect to preactivation
    sigma_prime, _ = get_activation_derivatives(activation_func, z)
    
    # Calculate full Jacobian and Laplacian w.r.t. input x
    jacobian = compute_full_jacobian(model, x, activation_func)
    laplacian = compute_full_laplacian(model, x, activation_func)
    
    # Number of neurons
    if layer_type == 'conv':
        N = z.size(1) * z.size(2) * z.size(3)
    else:  # 'fc'
        N = z.size(1)
    
    # Compute Jacobian norm
    jacobian_norm = torch.sqrt(torch.sum(jacobian**2))
    
    # Compute Laplacian approximation
    laplacian_sum = torch.sum(laplacian)
    
    # Compute the regularization term
    reg_term = (2 * sigma_prime.mean() * laplacian_sum / torch.sqrt(torch.tensor(N, device=z.device))) * \
              (1.0 / N - 1.0 / jacobian_norm)
    
    return torch.abs(reg_term)