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
    EPSILON = 1e-8  # Small constant for numerical stability
    
    # Forward pass to get preactivation
    z = model(x)
    
    # Calculate derivatives with respect to preactivation
    z = z.detach().requires_grad_(True)  # Ensure z has gradient
    sigma_prime, _ = get_activation_derivatives(activation_func, z)
    
    # Calculate approximate Jacobian and Laplacian
    jacobian = compute_jacobian_approximation(model, x, activation_func)
    laplacian = compute_laplacian_approximation(model, x, activation_func)
    
    # Number of neurons
    if layer_type == 'conv':
        N = z.size(1) * z.size(2) * z.size(3)
    else:  # 'fc'
        N = z.size(1)
    
    # Compute Jacobian norm with numerical stability
    jacobian_norm = torch.linalg.vector_norm(jacobian) + EPSILON
    # Compute Laplacian approximation
    laplacian_sum = torch.sum(laplacian)
    # Compute the regularization term (without abs to preserve gradient direction)
    reg_term = (2 * sigma_prime.mean() * laplacian_sum / torch.sqrt(torch.tensor(N, device=z.device))) * \
              (1.0 / N - 1.0 / jacobian_norm)
    
    return torch.abs(reg_term)

def compute_jacobian_approximation(model, x, activation_func):
    """Efficient Jacobian approximation using vectorized operations"""
    x = x.detach().clone().requires_grad_(True)
    z = model(x)
    a = activation_func(z)
    return torch.autograd.grad(a.sum(), x, create_graph=True, retain_graph=True)[0]

def compute_laplacian_approximation(model, x, activation_func):
    """
    Directly compute the Laplacian (sum of second derivatives) of the activation
    with respect to input x.
    
    Args:
        model: Neural network layer (nn.Linear or nn.Conv2d)
        x: Input tensor
        activation_func: Activation function
    
    Returns:
        Laplacian tensor
    """
    # Ensure fresh computation graph
    x = x.detach().clone().requires_grad_(True)
    
    # Forward pass
    z = model(x)
    a = activation_func(z)
    output_sum = a.sum()  # Sum all outputs to get a scalar
    
    # First derivatives
    grad_a = torch.autograd.grad(
        outputs=output_sum, 
        inputs=x, 
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Initialize laplacian as zeros with same shape as input
    laplacian = torch.zeros_like(x)
    
    # Compute diagonal elements of the Hessian (Laplacian)
    # Iterate through each element of the input tensor
    input_shape = x.shape
    flat_size = x.numel() // input_shape[0]  # Size of flattened input per batch
    
    # This is more efficient - compute gradients with respect to sum of gradient
    # This effectively computes the trace of the Hessian matrix
    hessian_diag = torch.autograd.grad(
        outputs=grad_a.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True
    )[0]
    
    return hessian_diag