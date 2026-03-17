import torch
from scipy.stats import qmc


def generate_grid_points(
    x_min: float, x_max: float, num_points: int, requires_grad: bool = True
) -> torch.Tensor:
    """
    Genera una malla de puntos equidistantes.

    Args:
        x_min (float): Límite inferior del dominio.
        x_max (float): Límite superior del dominio.
        num_points (int): Número de puntos a generar.
        requires_grad (bool): Si es True, PyTorch rastreará las derivadas en estos puntos.

    Returns:
        torch.Tensor: Tensor columna de forma (num_points, 1).
    """
    x = torch.linspace(x_min, x_max, num_points).unsqueeze(1)
    if requires_grad:
        x.requires_grad_(True)
    return x


def generate_random_points(
    x_min: float, x_max: float, num_points: int, requires_grad: bool = True
) -> torch.Tensor:
    """
    Genera puntos aleatorios con distribución uniforme continua.

    Args:
        x_min (float): Límite inferior.
        x_max (float): Límite superior.
        num_points (int): Número de puntos a generar.
        requires_grad (bool): Si es True, PyTorch rastreará las derivadas.

    Returns:
        torch.Tensor: Tensor columna de forma (num_points, 1).
    """

    x = x_min + (x_max - x_min) * torch.rand((num_points, 1))
    if requires_grad:
        x.requires_grad_(True)
    return x


def generate_boundary_points(
    x_min: float, x_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Genera los tensores correspondientes a las fronteras del dominio espacial.

    Args:
        x_min (float): Límite inferior.
        x_max (float): Límite superior.

    Returns:
        tuple: (tensor_frontera_izquierda, tensor_frontera_derecha)
    """

    x_left = torch.tensor([[x_min]], dtype=torch.float32)
    x_right = torch.tensor([[x_max]], dtype=torch.float32)

    return x_left, x_right


def generate_lhs_points(
    x_min: float, x_max: float, num_points: int, requires_grad: bool = True
) -> torch.Tensor:
    """
    Genera puntos utilizando Latin Hypercube Sampling (LHS).
    Garantiza una cobertura uniforme del dominio manteniendo la aleatoriedad.

    Args:
        x_min (float): Límite inferior del dominio.
        x_max (float): Límite superior del dominio.
        num_points (int): Número de puntos a generar.
        requires_grad (bool): Si es True, PyTorch rastreará las derivadas.

    Returns:
        torch.Tensor: Tensor columna de forma (num_points, 1).
    """
    # 1. Instancia del motor LHS para 1 dimensión (d=1)
    sampler = qmc.LatinHypercube(d=1)

    # 2. Generación de las muestras.
    sample = sampler.random(n=num_points)

    # 3. Escala matemáticamente al dominio físico.
    scaled_sample = x_min + sample * (x_max - x_min)

    # 4. Conversión del array de NumPy a un tensor de PyTorch.
    x = torch.tensor(scaled_sample, dtype=torch.float32)

    if requires_grad:
        x.requires_grad_(True)

    return x
