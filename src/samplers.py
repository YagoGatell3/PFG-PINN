import torch
from scipy.stats import qmc


def generate_grid_points(
    x_min: float, x_max: float, num_points: int, requires_grad: bool = True
) -> torch.Tensor:
    """
    Genera una malla espacial unidimensional con puntos uniformemente espaciados.

    Args:
        x_min (float): Límite inferior del dominio espacial.
        x_max (float): Límite superior del dominio espacial.
        num_points (int): Número de puntos a generar en la malla.
        requires_grad (bool, opcional): Si es True, habilita el cálculo de gradientes 
                                        para diferenciación automática. Por defecto es True.

    Returns:
        torch.Tensor: Tensor columna de dimensión (num_points, 1) con las coordenadas.
    """
    x = torch.linspace(x_min, x_max, num_points).unsqueeze(1)
    if requires_grad:
        x.requires_grad_(True)
    return x


def generate_random_points(
    x_min: float, x_max: float, num_points: int, requires_grad: bool = True
) -> torch.Tensor:
    """
    Genera un conjunto de puntos espaciales muestreados desde una distribución uniforme continua.

    Args:
        x_min (float): Límite inferior del dominio espacial.
        x_max (float): Límite superior del dominio espacial.
        num_points (int): Número de puntos aleatorios a generar.
        requires_grad (bool, opcional): Si es True, habilita el cálculo de gradientes. Por defecto es True.

    Returns:
        torch.Tensor: Tensor columna de dimensión (num_points, 1) con las coordenadas.
    """
    x = x_min + (x_max - x_min) * torch.rand((num_points, 1))
    if requires_grad:
        x.requires_grad_(True)
    return x


def generate_boundary_points(
    x_min: float, x_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Genera los tensores correspondientes a las fronteras estrictas del dominio espacial.

    Args:
        x_min (float): Límite inferior del dominio.
        x_max (float): Límite superior del dominio.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tupla conteniendo:
            - Tensor con la coordenada de la frontera izquierda.
            - Tensor con la coordenada de la frontera derecha.
    """
    x_left = torch.tensor([[x_min]], dtype=torch.float32)
    x_right = torch.tensor([[x_max]], dtype=torch.float32)

    return x_left, x_right


def generate_lhs_points(
    x_min: float, x_max: float, num_points: int, requires_grad: bool = True
) -> torch.Tensor:
    """
    Genera puntos de colocación utilizando Latin Hypercube Sampling (LHS).
    Este método garantiza una cobertura pseudoaleatoria más uniforme y representativa 
    del dominio en comparación con el muestreo puramente aleatorio.

    Args:
        x_min (float): Límite inferior del dominio espacial.
        x_max (float): Límite superior del dominio espacial.
        num_points (int): Número de puntos de muestra a generar.
        requires_grad (bool, opcional): Si es True, habilita el cálculo de gradientes. Por defecto es True.

    Returns:
        torch.Tensor: Tensor columna de dimensión (num_points, 1) con las coordenadas LHS.
    """
    # Instancia del motor LHS para un espacio unidimensional (d=1)
    sampler = qmc.LatinHypercube(d=1)

    # Generación de muestras en el intervalo normalizado [0, 1)
    sample = sampler.random(n=num_points)

    # Mapeo lineal de las muestras al dominio físico [x_min, x_max]
    scaled_sample = x_min + sample * (x_max - x_min)

    # Conversión estructural a tensor de PyTorch
    x = torch.tensor(scaled_sample, dtype=torch.float32)

    if requires_grad:
        x.requires_grad_(True)

    return x