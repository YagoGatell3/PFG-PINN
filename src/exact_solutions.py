import math

import torch


def hermite(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Calcula los polinomios de Hermite de forma recursiva usando tensores de PyTorch.

    Args:
        n (int): Grado del polinomio (estado de energía cuántico).
        x (torch.Tensor): Tensor con las coordenadas espaciales evaluadas.

    Returns:
        torch.Tensor: Evaluaciones del polinomio de Hermite de grado n en x.
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 2 * x
    return 2 * x * hermite(n - 1, x) - 2 * (n - 1) * hermite(n - 2, x)


def psi(
    x: torch.Tensor, n: int, mass: float = 1.0, omega: float = 1.0, hbar: float = 1.0
) -> torch.Tensor:
    """
    Solución analítica exacta de la función de onda para el Oscilador Armónico Cuántico 1D.

    Args:
        x (torch.Tensor): Coordenadas espaciales.
        n (int): Número cuántico principal (nivel de energía).
        mass (float): Masa de la partícula.
        omega (float): Frecuencia angular del oscilador.
        hbar (float): Constante reducida de Planck.

    Returns:
        torch.Tensor: Valores exactos de la función de onda psi_n(x).
    """
    # Constantes físicas a tensores
    mass_t = torch.tensor(mass, dtype=torch.float32)
    omega_t = torch.tensor(omega, dtype=torch.float32)
    hbar_t = torch.tensor(hbar, dtype=torch.float32)

    # 1. Constante de normalización
    factor_term = 1.0 / torch.sqrt(
        torch.tensor(2.0**n * math.factorial(n), dtype=torch.float32)
    )

    # 2. Factor de escala alpha
    alpha = torch.sqrt(mass_t * omega_t / hbar_t)

    # 3. Decaimiento gaussiano (envoltura)
    exp_term = torch.exp(-alpha * x**2 / 2.0)

    # 4. Polinomios de Hermite responsables de los nodos/oscilaciones
    hermite_term = hermite(n, alpha * x)

    # Composición final de la función de onda
    return factor_term * (alpha / torch.pi) ** 0.25 * exp_term * hermite_term
