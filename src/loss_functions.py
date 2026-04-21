import torch
import torch.nn as nn


def physics_loss_QHO(
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


def physics_loss_infinite_well(
    model: torch.nn.Module,
    x: torch.Tensor,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> torch.Tensor:
    """
    Calcula el residuo para la partícula en un pozo de potencial infinito 1D.
    """
    if not x.requires_grad:
        x.requires_grad_(True)

    u = model(x)

    # Diferenciación automática
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    mass_t = torch.tensor(mass, dtype=torch.float32)
    hbar_t = torch.tensor(hbar, dtype=torch.float32)

    # Dentro del pozo, V(x) = 0.
    energy = model.epsilon

    # Ecuación de Schrödinger
    ph_loss = -0.5 * (hbar_t**2 / mass_t) * d2u_dx2 - energy * u

    return torch.mean(ph_loss**2)


def physics_loss_classical_oscillator(
    model: torch.nn.Module,
    t: torch.Tensor,
    mass: float = 1.0,
    k: float = 1.0,  # Constante elástica del muelle
) -> torch.Tensor:
    """
    Calcula el residuo para el Oscilador Armónico Clásico (Muelle-Masa).
    """
    if not t.requires_grad:
        t.requires_grad_(True)

    u = model(t)  # u es la posición, t es el tiempo

    # Diferenciación automática respecto al tiempo
    du_dt = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dt2 = torch.autograd.grad(
        du_dt, t, grad_outputs=torch.ones_like(du_dt), create_graph=True
    )[0]

    mass_t = torch.tensor(mass, dtype=torch.float32)
    k_t = torch.tensor(k, dtype=torch.float32)

    # Segunda ley de Newton
    ph_loss = mass_t * d2u_dt2 + k_t * u

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


def initial_condition_loss(
    model: nn.Module, t_0: torch.Tensor, u_0: float = 1.0, v_0: float = 0.0
) -> torch.Tensor:
    """
    Calcula la pérdida de las condiciones iniciales para el Oscilador Clásico (IVP).
    Fuerza a la red a respetar la posición inicial u(0) y la velocidad inicial u'(0).

    Args:
        model (nn.Module): La red neuronal prediciendo la posición en el tiempo.
        t_0 (torch.Tensor): El tensor que representa el instante t=0.
        u_0 (float): Posición inicial deseada.
        v_0 (float): Velocidad inicial deseada.
    """
    if not t_0.requires_grad:
        t_0.requires_grad_(True)

    # 1. Predicción de la posición en t=0
    u_pred = model(t_0)

    # 2. Predicción de la velocidad en t=0
    v_pred = torch.autograd.grad(
        u_pred, t_0, grad_outputs=torch.ones_like(u_pred), create_graph=True
    )[0]

    u_target = torch.tensor(u_0, dtype=torch.float32, device=t_0.device)
    v_target = torch.tensor(v_0, dtype=torch.float32, device=t_0.device)

    # 3. Calculamos el Error Cuadrático Medio para posición y velocidad
    loss_position = torch.mean((u_pred - u_target) ** 2)
    loss_velocity = torch.mean((v_pred - v_target) ** 2)

    # La pérdida total de condición inicial es la suma de ambas
    return loss_position + loss_velocity


def physics_loss_damped_pendulum(
    model: torch.nn.Module, 
    t: torch.Tensor, 
    L: float = 1.0
) -> torch.Tensor:
    """
    Residuo para el péndulo amortiguado: d2theta/dt2 + mu*dtheta/dt + (g/L)*sin(theta) = 0
    """
    if not t.requires_grad:
        t.requires_grad_(True)

    theta = model(t)

    # Derivadas respecto al tiempo
    dtheta_dt = torch.autograd.grad(
        theta, t, grad_outputs=torch.ones_like(theta), create_graph=True
    )[0]
    
    d2theta_dt2 = torch.autograd.grad(
        dtheta_dt, t, grad_outputs=torch.ones_like(dtheta_dt), create_graph=True
    )[0]

    L_t = torch.tensor(L, dtype=torch.float32, device=t.device)
    
    # Extraemos los parámetros entrenables del modelo
    g_pred = model.g
    mu_pred = model.mu

    # Ecuación del movimiento
    ph_loss = d2theta_dt2 + mu_pred * dtheta_dt + (g_pred / L_t) * torch.sin(theta)

    return torch.mean(ph_loss**2)

def physics_loss_tunnel(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    V0: float = 1.5,
    x_barrier_left: float = 0.5,
    x_barrier_right: float = 1.5,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> torch.Tensor:
    """
    Residuo de la TDSE separada en parte real e imaginaria.

    du/dt = -(hbar/2m) * d2v/dx2 + (V/hbar) * v
    dv/dt = +(hbar/2m) * d2u/dx2 - (V/hbar) * u
    """
    if not x.requires_grad:
        x.requires_grad_(True)
    if not t.requires_grad:
        t.requires_grad_(True)

    u, v = model(x, t)

    # --- Derivadas espaciales de u ---
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    # --- Derivadas espaciales de v ---
    dv_dx = torch.autograd.grad(
        v, x, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]
    d2v_dx2 = torch.autograd.grad(
        dv_dx, x, grad_outputs=torch.ones_like(dv_dx), create_graph=True
    )[0]

    # --- Derivadas temporales ---
    du_dt = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    dv_dt = torch.autograd.grad(
        v, t, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]

    hbar_t = torch.tensor(hbar, dtype=torch.float32)
    mass_t = torch.tensor(mass, dtype=torch.float32)

    # Potencial de barrera rectangular
    V = torch.where(
        (x >= x_barrier_left) & (x <= x_barrier_right),
        torch.tensor(V0, dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    )

    coeff = (hbar_t**2) / (2.0 * mass_t)

    # Residuos Re/Im de la TDSE
    res_u = du_dt + coeff * d2v_dx2 - (V / hbar_t) * v
    res_v = dv_dt - coeff * d2u_dx2 + (V / hbar_t) * u

    return torch.mean(res_u**2) + torch.mean(res_v**2)


def initial_condition_loss_tunnel(
    model,
    x: torch.Tensor,
    t0: torch.Tensor,
    x0: float = -3.0,
    sigma: float = 0.5,
    k0: float = 3.0,
    hbar: float = 1.0,
) -> torch.Tensor:
    """
    Fuerza la condición inicial: paquete gaussiano complejo en t=0.
    Psi(x,0) = A * exp(-(x-x0)^2 / 2sigma^2) * exp(i*k0*x)
    """
    # Amplitud gaussiana
    gauss = torch.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
    norm = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
    
    # Parte real e imaginaria de exp(i*k0*x)
    u_true = norm * gauss * torch.cos(k0 * x)
    v_true = norm * gauss * torch.sin(k0 * x)

    u_pred, v_pred = model(x, t0)

    return torch.mean((u_pred - u_true) ** 2) + torch.mean((v_pred - v_true) ** 2)


def boundary_loss_tunnel(
    model,
    t: torch.Tensor,
    x_min: float = -10.0,
    x_max: float = 10.0,
) -> torch.Tensor:
    """
    Condiciones de frontera de Dirichlet: Psi = 0 en x = x_min y x = x_max.
    """
    x_left  = torch.full_like(t, x_min)
    x_right = torch.full_like(t, x_max)

    u_left,  v_left  = model(x_left,  t)
    u_right, v_right = model(x_right, t)

    return (
        torch.mean(u_left**2)  + torch.mean(v_left**2) +
        torch.mean(u_right**2) + torch.mean(v_right**2)
    )


def normalization_loss_tunnel(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    domain_length: float = 20.0,
) -> torch.Tensor:
    """
    Fuerza la conservación de la norma: integral |Psi|^2 dx = 1 para cada t.
    """
    u, v = model(x, t)
    prob_density = u**2 + v**2
    integral = domain_length * torch.mean(prob_density)
    return (integral - 1.0) ** 2

def physics_loss_heat_inverse(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Residuo de la ecuación del calor usando alpha entrenable del modelo.
    du/dt - alpha * d2u/dx2 = 0
    """
    if not x.requires_grad:
        x.requires_grad_(True)
    if not t.requires_grad:
        t.requires_grad_(True)

    u = model(x, t)

    du_dt = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    # alpha viene del modelo, no de un argumento fijo
    residuo = du_dt - model.alpha * d2u_dx2

    return torch.mean(residuo**2)


def initial_condition_loss_heat(
    model: nn.Module,
    x: torch.Tensor,
    t0: torch.Tensor,
    L: float = 1.0,
) -> torch.Tensor:
    """
    Fuerza la condición inicial: gaussiana centrada en L/2.
    u(x, 0) = exp(-(x - L/2)^2 / (2*(L/8)^2))

    Args:
        model: Red neuronal
        x: Coordenadas espaciales (N, 1)
        t0: Tensor de ceros representando t=0 (N, 1)
        L: Longitud del dominio

    Returns:
        torch.Tensor: MSE respecto a la condición inicial
    """
    x0    = L / 2.0
    sigma = L / 8.0

    u_true = torch.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
    u_pred = model(x, t0)

    return torch.mean((u_pred - u_true) ** 2)


def boundary_loss_heat(
    model: nn.Module,
    t: torch.Tensor,
    x_min: float = 0.0,
    x_max: float = 1.0,
) -> torch.Tensor:
    """
    Condiciones de frontera de Dirichlet: u = 0 en x=0 y x=L.
    Simula extremos del dominio fijados a temperatura cero.

    Args:
        model: Red neuronal
        t: Instantes temporales muestreados (N, 1)
        x_min: Frontera izquierda
        x_max: Frontera derecha

    Returns:
        torch.Tensor: MSE en las fronteras
    """
    x_left  = torch.full_like(t, x_min)
    x_right = torch.full_like(t, x_max)

    u_left  = model(x_left,  t)
    u_right = model(x_right, t)

    return torch.mean(u_left**2) + torch.mean(u_right**2)