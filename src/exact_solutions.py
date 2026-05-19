import math

import torch


def hermite(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Calcula los polinomios de Hermite de forma recursiva.

    Args:
        n (int): Grado del polinomio.
        x (torch.Tensor): Coordenadas espaciales evaluadas.

    Returns:
        torch.Tensor: Evaluaciones del polinomio de Hermite de grado n en x.
    """
    if n == 0:
        return torch.ones_like(x)
    if n == 1:
        return 2 * x
    return 2 * x * hermite(n - 1, x) - 2 * (n - 1) * hermite(n - 2, x)


def psi_QHO(
    x: torch.Tensor, n: int, mass: float = 1.0, omega: float = 1.0, hbar: float = 1.0
) -> torch.Tensor:
    """
    Solución analítica exacta de la función de onda para el Oscilador Armónico Cuántico 1D.

    Args:
        x (torch.Tensor): Coordenadas espaciales.
        n (int): Número cuántico principal (nivel de energía).
        mass (float, opcional): Masa de la partícula. Por defecto es 1.0.
        omega (float, opcional): Frecuencia angular del oscilador. Por defecto es 1.0.
        hbar (float, opcional): Constante reducida de Planck. Por defecto es 1.0.

    Returns:
        torch.Tensor: Valores de la función de onda correspondiente al estado n.
    """
    # Conversión de constantes físicas a tensores
    mass_t = torch.tensor(mass, dtype=torch.float32)
    omega_t = torch.tensor(omega, dtype=torch.float32)
    hbar_t = torch.tensor(hbar, dtype=torch.float32)

    # Cálculo de la constante de normalización
    factor_term = 1.0 / torch.sqrt(
        torch.tensor(2.0**n * math.factorial(n), dtype=torch.float32)
    )

    # Factor de escala alpha
    alpha = torch.sqrt(mass_t * omega_t / hbar_t)

    # Decaimiento gaussiano (envoltura)
    exp_term = torch.exp(-alpha * x**2 / 2.0)

    # Polinomios de Hermite (nodos y oscilaciones)
    hermite_term = hermite(n, alpha * x)

    return factor_term * (alpha / torch.pi) ** 0.25 * exp_term * hermite_term


def psi_infinite_well(x: torch.Tensor, n: int, L: float = 1.0) -> torch.Tensor:
    """
    Solución analítica exacta para la partícula en un pozo de potencial infinito 1D.

    Args:
        x (torch.Tensor): Coordenadas espaciales.
        n (int): Estado cuántico (1 = fundamental, 2 = primer excitado, etc.).
        L (float, opcional): Anchura del pozo (rango de x=0 a x=L). Por defecto es 1.0.

    Returns:
        torch.Tensor: Valores de la función de onda en el estado n.
    """
    L_t = torch.tensor(L, dtype=torch.float32)

    # Constante de normalización
    norm_factor = torch.sqrt(2.0 / L_t)

    # Componente oscilatoria
    sin_term = torch.sin(n * torch.pi * x / L_t)

    psi = norm_factor * sin_term

    # Anulación estricta de la probabilidad fuera de los límites del pozo (x < 0 o x > L)
    psi = torch.where((x < 0.0) | (x > L), torch.zeros_like(psi), psi)

    return psi


def classical_oscillator(
    t: torch.Tensor,
    mass: float = 1.0,
    k: float = 1.0,
    u_0: float = 1.0,
    v_0: float = 0.0,
) -> torch.Tensor:
    """
    Solución analítica exacta para el Oscilador Armónico Clásico (sistema masa-muelle).

    Args:
        t (torch.Tensor): Coordenadas temporales.
        mass (float, opcional): Masa del objeto. Por defecto es 1.0.
        k (float, opcional): Constante elástica del muelle. Por defecto es 1.0.
        u_0 (float, opcional): Posición inicial en t=0. Por defecto es 1.0.
        v_0 (float, opcional): Velocidad inicial en t=0. Por defecto es 0.0.

    Returns:
        torch.Tensor: Posición del oscilador u(t) en cada instante de tiempo.
    """
    mass_t = torch.tensor(mass, dtype=torch.float32)
    k_t = torch.tensor(k, dtype=torch.float32)

    # Frecuencia angular del sistema
    omega = torch.sqrt(k_t / mass_t)

    # Componentes de la ecuación de movimiento
    term_cos = u_0 * torch.cos(omega * t)
    term_sin = (v_0 / omega) * torch.sin(omega * t)

    return term_cos + term_sin


def psi_tunnel_initial(
    x: torch.Tensor,
    x0: float = -3.0,
    sigma: float = 0.5,
    k0: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generación de un paquete de ondas gaussiano inicial (t=0) para simular el efecto túnel.
    La función implementa: Psi(x,0) = A * exp(-(x-x0)^2 / 2sigma^2) * exp(i*k0*x)

    Args:
        x (torch.Tensor): Coordenadas espaciales.
        x0 (float, opcional): Posición central inicial del paquete. Por defecto es -3.0.
        sigma (float, opcional): Desviación estándar (anchura) del paquete. Por defecto es 0.5.
        k0 (float, opcional): Número de onda inicial (momento). Por defecto es 3.0.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tupla que contiene:
            - Parte real de la función de onda Psi.
            - Parte imaginaria de la función de onda Psi.
    """
    norm = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
    gauss = torch.exp(-((x - x0) ** 2) / (2.0 * sigma**2))

    u = norm * gauss * torch.cos(k0 * x)
    v = norm * gauss * torch.sin(k0 * x)
    return u, v


def heat_exact(
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float = 0.1,
    L: float = 1.0,
    n_terms: int = 50,
) -> torch.Tensor:
    """
    Solución analítica exacta de la ecuación del calor 1D mediante Series de Fourier.
    Condiciones aplicadas:
        - Inicial: Distribución gaussiana centrada en L/2.
        - Frontera: Condiciones de Neumann (du/dx = 0 en x=0 y x=L, extremos aislados).

    La serie implementada es:
        u(x,t) = A0/2 + sum_{n=1}^{N} A_n * cos(n*pi*x/L) * exp(-alpha*(n*pi/L)^2*t)

    Args:
        x (torch.Tensor): Coordenadas espaciales.
        t (torch.Tensor): Coordenadas temporales.
        alpha (float, opcional): Difusividad térmica. Por defecto es 0.1.
        L (float, opcional): Longitud del dominio espacial. Por defecto es 1.0.
        n_terms (int, opcional): Número de términos de la serie de Fourier a computar. Por defecto es 50.

    Returns:
        torch.Tensor: Perfil de temperatura u(x,t).
    """
    L_t     = torch.tensor(L,     dtype=torch.float32)
    alpha_t = torch.tensor(alpha, dtype=torch.float32)

    x0    = L / 2.0
    sigma = L / 8.0

    # Cuadratura base para el cálculo de coeficientes
    x_int = torch.linspace(0.0, L, 1000).unsqueeze(1)
    f_int = torch.exp(-((x_int - x0) ** 2) / (2.0 * sigma**2))

    # Cálculo del término constante A0/2 (n=0)
    A0     = (2.0 / L) * torch.trapezoid(f_int.squeeze(), x_int.squeeze())
    result = (A0 / 2.0) * torch.ones_like(x)

    # Sumatoria de términos de la serie de Fourier
    for n in range(1, n_terms + 1):
        n_t = torch.tensor(float(n), dtype=torch.float32)

        cos_int = torch.cos(n_t * torch.pi * x_int / L_t)
        A_n     = (2.0 / L_t) * torch.trapezoid(
            (f_int * cos_int).squeeze(), x_int.squeeze()
        )

        decay  = torch.exp(-alpha_t * (n_t * torch.pi / L_t) ** 2 * t)
        result = result + A_n * torch.cos(n_t * torch.pi * x / L_t) * decay

    return result