import torch
import torch.nn as nn


def physics_loss(
    model: nn.Module,
    x: torch.Tensor,
    mass: float = 1.0,
    omega: float = 1.0,
    hbar: float = 1.0,
) -> torch.Tensor:
    """
    Calcula el residuo de la Ecuación de Schrödinger Independiente del Tiempo (PINN).

    Args:
        model (nn.Module): La red neuronal que aproxima la función de onda.
        x (torch.Tensor): Puntos de colocación en el dominio espacial.
        mass (float): Masa de la partícula.
        omega (float): Frecuencia angular.
        hbar (float): Constante reducida de Planck.

    Returns:
        torch.Tensor: Error Cuadrático Medio (MSE) del residuo físico.
    """
    # Nos aseguramos de que PyTorch rastree las operaciones para calcular derivadas
    if not x.requires_grad:
        x.requires_grad_(True)

    u = model(x)  # Predicción de la red

    # 1. Diferenciación Automática (Derivadas exactas)
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    # Constantes físicas a tensores
    mass_t = torch.tensor(mass, dtype=torch.float32)
    omega_t = torch.tensor(omega, dtype=torch.float32)
    hbar_t = torch.tensor(hbar, dtype=torch.float32)

    # 2. Definición del sistema físico
    potential = 0.5 * mass_t * omega_t**2 * x**2
    energy = (
        hbar_t * omega_t * model.epsilon
    )  # El autovalor que la red está aprendiendo

    # 3. Residuo de Schrödinger
    ph_loss = -0.5 * (hbar_t**2 / mass_t) * d2u_dx2 + potential * u - energy * u

    return torch.mean(ph_loss**2)


def boundary_loss(
    model: nn.Module, x_min: torch.Tensor, x_max: torch.Tensor
) -> torch.Tensor:
    """
    Impone condiciones de contorno de Dirichlet homogéneas (psi = 0 en los extremos).

    Args:
        model (nn.Module): La red neuronal.
        x_min (torch.Tensor): Límite inferior del dominio espacial.
        x_max (torch.Tensor): Límite superior del dominio espacial.

    Returns:
        torch.Tensor: MSE de las predicciones en las fronteras.
    """
    u_min = model(x_min)
    u_max = model(x_max)

    # Penalización de cualquier desviación de 0 en los extremos
    return torch.mean(u_min**2) + torch.mean(u_max**2)


def orthogonality_loss(
    model: nn.Module, x: torch.Tensor, psi_n: torch.Tensor, domain_length: float = 20.0
) -> torch.Tensor:
    """
    Fuerza a la red a generar un estado que sea ortogonal a un estado previo conocido.
    Crucial para descubrir el estado excitado n=n+1 sin usar datos empíricos.

    Args:
        model (nn.Module): La red neuronal prediciendo el estado actual.
        x (torch.Tensor): Puntos de colocación en el dominio espacial.
        psi_n (torch.Tensor): Los valores del estado excitado (n=n) evaluados en x.
                              Pueden venir de la solución exacta o de otra PINN pre-entrenada.

    Returns:
        torch.Tensor: Penalización por falta de ortogonalidad.
    """
    u_pred = model(x)

    # Aproximación de la integral
    integral = domain_length * torch.mean(u_pred * psi_n)

    return integral**2


def normalization_loss(
    model: nn.Module, x: torch.Tensor, domain_length: float = 20.0
) -> torch.Tensor:
    """
    Fuerza a que la probabilidad total de la función de onda sea 1.
    """
    u_pred = model(x)

    # Aproximación de la integral
    integral = domain_length * torch.mean(u_pred**2)

    # Penalización de la desviación respecto a 1
    return (integral - 1.0) ** 2
