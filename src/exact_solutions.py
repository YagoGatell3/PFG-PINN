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


def psi_QHO(
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


def psi_infinite_well(x: torch.Tensor, n: int, L: float = 1.0) -> torch.Tensor:
    """
    Solución analítica exacta para la partícula en un pozo de potencial infinito 1D.

    Args:
        x (torch.Tensor): Coordenadas espaciales.
        n (int): Estado cuántico (0 = fundamental, 1 = primer excitado...).
        L (float): Anchura del pozo (va desde x=0 hasta x=L).

    Returns:
        torch.Tensor: Valores exactos de la función de onda psi_n(x).
    """
    L_t = torch.tensor(L, dtype=torch.float32)

    # Constante de normalización
    norm_factor = torch.sqrt(2.0 / L_t)

    # Onda senoidal
    sin_term = torch.sin(n * torch.pi * x / L_t)

    psi = norm_factor * sin_term

    # Fuera del pozo (x < 0 o x > L) la probabilidad es estrictamente 0
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
    Solución analítica exacta para el Oscilador Armónico Clásico (Masa-Muelle).

    Args:
        t (torch.Tensor): Coordenadas temporales.
        mass (float): Masa del objeto.
        k (float): Constante elástica del muelle.
        u_0 (float): Posición inicial en t=0.
        v_0 (float): Velocidad inicial en t=0.

    Returns:
        torch.Tensor: Valores exactos de la posición u(t).
    """
    mass_t = torch.tensor(mass, dtype=torch.float32)
    k_t = torch.tensor(k, dtype=torch.float32)

    # Frecuencia angular clásica
    omega = torch.sqrt(k_t / mass_t)

    # Ecuación de movimiento
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
    Paquete gaussiano inicial (t=0) para el efecto túnel.
    Psi(x,0) = A * exp(-(x-x0)^2 / 2sigma^2) * exp(i*k0*x)

    Returns:
        u: Parte real de Psi
        v: Parte imaginaria de Psi
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
    Solución analítica exacta de la ecuación del calor 1D con:
      - Condición inicial: gaussiana centrada en L/2
      - Condiciones de frontera: u(0,t) = u(L,t) = 0

    Calcula los coeficientes B_n por proyección de la gaussiana
    sobre la base de senos y acumula la serie de Fourier.

    Args:
        x: Coordenadas espaciales (N, 1)
        t: Coordenadas temporales (N, 1)
        alpha: Difusividad térmica
        L: Longitud del dominio
        n_terms: Número de términos de la serie de Fourier

    Returns:
        torch.Tensor: Temperatura u(x, t) de forma (N, 1)
    """
    L_t     = torch.tensor(L,     dtype=torch.float32)
    alpha_t = torch.tensor(alpha, dtype=torch.float32)

    # Condición inicial: gaussiana centrada en L/2
    x0    = L / 2.0
    sigma = L / 8.0

    result = torch.zeros_like(x)

    for n in range(1, n_terms + 1):
        n_t = torch.tensor(float(n), dtype=torch.float32)

        # Coeficiente B_n = (2/L) * integral_0^L f(x)*sin(n*pi*x/L) dx
        # Para gaussiana: aproximación numérica de la integral
        x_int = torch.linspace(0.0, L, 1000).unsqueeze(1)
        f_int = torch.exp(-((x_int - x0) ** 2) / (2.0 * sigma**2))
        sin_int = torch.sin(n_t * torch.pi * x_int / L_t)
        B_n = (2.0 / L_t) * torch.trapezoid(
            (f_int * sin_int).squeeze(), x_int.squeeze()
        )

        # Decaimiento exponencial temporal
        decay = torch.exp(-alpha_t * (n_t * torch.pi / L_t) ** 2 * t)

        # Término n de la serie
        result = result + B_n * torch.sin(n_t * torch.pi * x / L_t) * decay

    return result