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
        model (nn.Module): Red neuronal que aproxima la función de onda.
        x (torch.Tensor): Puntos de colocación en el dominio espacial.
        mass (float, opcional): Masa de la partícula. Por defecto es 1.0.
        omega (float, opcional): Frecuencia angular. Por defecto es 1.0.
        hbar (float, opcional): Constante reducida de Planck. Por defecto es 1.0.

    Returns:
        torch.Tensor: Error Cuadrático Medio (MSE) del residuo físico.
    """
    # Habilitar el cálculo de gradientes respecto a la entrada espacial
    if not x.requires_grad:
        x.requires_grad_(True)

    u = model(x)

    # Diferenciación automática para obtener las derivadas exactas
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    # Conversión de constantes físicas a tensores
    mass_t = torch.tensor(mass, dtype=torch.float32)
    omega_t = torch.tensor(omega, dtype=torch.float32)
    hbar_t = torch.tensor(hbar, dtype=torch.float32)

    # Definición del potencial armónico y la energía (autovalor entrenable)
    potential = 0.5 * mass_t * omega_t**2 * x**2
    energy = hbar_t * omega_t * model.epsilon

    # Cálculo del residuo de la ecuación de Schrödinger
    ph_loss = -0.5 * (hbar_t**2 / mass_t) * d2u_dx2 + potential * u - energy * u

    return torch.mean(ph_loss**2)


def physics_loss_infinite_well(
    model: torch.nn.Module,
    x: torch.Tensor,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> torch.Tensor:
    """
    Calcula el residuo de la ecuación de Schrödinger para la partícula en un 
    pozo de potencial infinito 1D.

    Args:
        model (nn.Module): Red neuronal que aproxima la función de onda.
        x (torch.Tensor): Puntos de colocación en el dominio espacial.
        mass (float, opcional): Masa de la partícula. Por defecto es 1.0.
        hbar (float, opcional): Constante reducida de Planck. Por defecto es 1.0.

    Returns:
        torch.Tensor: Error Cuadrático Medio (MSE) del residuo físico.
    """
    if not x.requires_grad:
        x.requires_grad_(True)

    u = model(x)

    # Diferenciación automática espacial
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    mass_t = torch.tensor(mass, dtype=torch.float32)
    hbar_t = torch.tensor(hbar, dtype=torch.float32)

    # Dentro del pozo el potencial es nulo (V(x) = 0)
    energy = model.epsilon

    # Residuo de la ecuación de Schrödinger
    ph_loss = (-0.5 * (hbar_t**2 / mass_t) * d2u_dx2) / energy - u
    return torch.mean(ph_loss**2)


def physics_loss_classical_oscillator(
    model: torch.nn.Module,
    t: torch.Tensor,
    mass: float = 1.0,
    k: float = 1.0,
) -> torch.Tensor:
    """
    Calcula el residuo de la ecuación de movimiento para el Oscilador Armónico Clásico.

    Args:
        model (nn.Module): Red neuronal que aproxima la posición en función del tiempo.
        t (torch.Tensor): Puntos de colocación en el dominio temporal.
        mass (float, opcional): Masa del oscilador. Por defecto es 1.0.
        k (float, opcional): Constante elástica del muelle. Por defecto es 1.0.

    Returns:
        torch.Tensor: Error Cuadrático Medio (MSE) del residuo físico.
    """
    if not t.requires_grad:
        t.requires_grad_(True)

    u = model(t)

    # Diferenciación automática temporal
    du_dt = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dt2 = torch.autograd.grad(
        du_dt, t, grad_outputs=torch.ones_like(du_dt), create_graph=True
    )[0]

    mass_t = torch.tensor(mass, dtype=torch.float32)
    k_t = torch.tensor(k, dtype=torch.float32)

    # Aplicación de la Segunda Ley de Newton
    ph_loss = mass_t * d2u_dt2 + k_t * u

    return torch.mean(ph_loss**2)


def boundary_loss(
    model: nn.Module, x_min: torch.Tensor, x_max: torch.Tensor
) -> torch.Tensor:
    """
    Impone condiciones de contorno de Dirichlet homogéneas (función nula en los extremos).

    Args:
        model (nn.Module): Red neuronal.
        x_min (torch.Tensor): Límite inferior del dominio espacial.
        x_max (torch.Tensor): Límite superior del dominio espacial.

    Returns:
        torch.Tensor: MSE de las predicciones en las fronteras.
    """
    u_min = model(x_min)
    u_max = model(x_max)

    return torch.mean(u_min**2) + torch.mean(u_max**2)


def orthogonality_loss(
    model: nn.Module, x: torch.Tensor, psi_n: torch.Tensor, domain_length: float = 20.0
) -> torch.Tensor:
    """
    Fuerza a la red a generar un estado que sea ortogonal a un estado previo conocido.
    Esencial para descubrir estados excitados (n+1) sin requerir datos empíricos.

    Args:
        model (nn.Module): Red neuronal prediciendo el estado actual.
        x (torch.Tensor): Puntos de colocación en el dominio espacial.
        psi_n (torch.Tensor): Valores del estado previo evaluados en x (solución exacta o pre-entrenada).
        domain_length (float, opcional): Longitud del dominio de integración. Por defecto es 20.0.

    Returns:
        torch.Tensor: Penalización por falta de ortogonalidad.
    """
    u_pred = model(x)

    # Aproximación discreta de la integral del producto interno
    integral = domain_length * torch.mean(u_pred * psi_n)

    return integral**2


def normalization_loss(
    model: nn.Module, x: torch.Tensor, domain_length: float = 20.0
) -> torch.Tensor:
    """
    Garantiza que la probabilidad total de la función de onda integre a 1.

    Args:
        model (nn.Module): Red neuronal que aproxima la función de onda.
        x (torch.Tensor): Puntos de colocación en el dominio espacial.
        domain_length (float, opcional): Longitud del dominio de integración. Por defecto es 20.0.

    Returns:
        torch.Tensor: Penalización por la desviación de la norma respecto a la unidad.
    """
    u_pred = model(x)

    # Aproximación discreta de la integral de densidad de probabilidad
    integral = domain_length * torch.mean(u_pred**2)

    return (integral - 1.0) ** 2


def initial_condition_loss(
    model: nn.Module, t_0: torch.Tensor, u_0: float = 1.0, v_0: float = 0.0
) -> torch.Tensor:
    """
    Calcula la pérdida asociada a las condiciones iniciales para problemas de valor inicial (IVP).
    Restringe la posición u(0) y la velocidad u'(0) del sistema.

    Args:
        model (nn.Module): Red neuronal prediciendo la variable de estado en el tiempo.
        t_0 (torch.Tensor): Tensor que representa el instante inicial (t=0).
        u_0 (float, opcional): Posición inicial objetivo. Por defecto es 1.0.
        v_0 (float, opcional): Velocidad inicial objetivo. Por defecto es 0.0.

    Returns:
        torch.Tensor: MSE combinado para la posición y velocidad iniciales.
    """
    if not t_0.requires_grad:
        t_0.requires_grad_(True)

    # Predicción del estado en t=0
    u_pred = model(t_0)

    # Predicción de la derivada temporal en t=0
    v_pred = torch.autograd.grad(
        u_pred, t_0, grad_outputs=torch.ones_like(u_pred), create_graph=True
    )[0]

    u_target = torch.tensor(u_0, dtype=torch.float32, device=t_0.device)
    v_target = torch.tensor(v_0, dtype=torch.float32, device=t_0.device)

    # Cálculo del error cuadrático respecto a los objetivos
    loss_position = torch.mean((u_pred - u_target) ** 2)
    loss_velocity = torch.mean((v_pred - v_target) ** 2)

    return loss_position + loss_velocity


def physics_loss_damped_pendulum(
    model: torch.nn.Module, 
    t: torch.Tensor, 
    L: float = 1.0
) -> torch.Tensor:
    """
    Calcula el residuo de la ecuación diferencial del péndulo amortiguado:
    d^2(theta)/dt^2 + mu * d(theta)/dt + (g/L) * sin(theta) = 0

    Args:
        model (torch.nn.Module): Red neuronal que predice el ángulo theta en función del tiempo.
        t (torch.Tensor): Puntos de colocación temporal.
        L (float, opcional): Longitud del péndulo. Por defecto es 1.0.

    Returns:
        torch.Tensor: Error Cuadrático Medio (MSE) del residuo físico.
    """
    if not t.requires_grad:
        t.requires_grad_(True)

    theta = model(t)

    # Derivadas temporales de primer y segundo orden
    dtheta_dt = torch.autograd.grad(
        theta, t, grad_outputs=torch.ones_like(theta), create_graph=True
    )[0]
    
    d2theta_dt2 = torch.autograd.grad(
        dtheta_dt, t, grad_outputs=torch.ones_like(dtheta_dt), create_graph=True
    )[0]

    L_t = torch.tensor(L, dtype=torch.float32, device=t.device)
    
    # Extracción de parámetros físicos entrenables del modelo
    g_pred = model.g
    mu_pred = model.mu

    # Ecuación de movimiento del péndulo
    ph_loss = d2theta_dt2 + mu_pred * dtheta_dt + (g_pred / L_t) * torch.sin(theta)

    return torch.mean(ph_loss**2)


def physics_loss_tunnel(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    V0: float = 1.5,
    x_barrier_left: float = 0.5,
    x_barrier_right: float = 1.5,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> torch.Tensor:
    """
    Calcula el residuo de la Ecuación de Schrödinger Dependiente del Tiempo (TDSE),
    desacoplada en sus componentes real (u) e imaginaria (v):
        du/dt = -(hbar/2m) * d^2v/dx^2 + (V/hbar) * v
        dv/dt = +(hbar/2m) * d^2u/dx^2 - (V/hbar) * u

    Args:
        model (nn.Module): Red neuronal que retorna (u, v).
        x (torch.Tensor): Puntos de colocación espacial.
        t (torch.Tensor): Puntos de colocación temporal.
        V0 (float, opcional): Altura de la barrera de potencial. Por defecto es 1.5.
        x_barrier_left (float, opcional): Inicio de la barrera. Por defecto es 0.5.
        x_barrier_right (float, opcional): Fin de la barrera. Por defecto es 1.5.
        mass (float, opcional): Masa de la partícula. Por defecto es 1.0.
        hbar (float, opcional): Constante reducida de Planck. Por defecto es 1.0.

    Returns:
        torch.Tensor: MSE combinado de los residuos real e imaginario.
    """
    if not x.requires_grad:
        x.requires_grad_(True)
    if not t.requires_grad:
        t.requires_grad_(True)

    u, v = model(x, t)

    # Derivadas espaciales para la parte real (u)
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    # Derivadas espaciales para la parte imaginaria (v)
    dv_dx = torch.autograd.grad(
        v, x, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]
    d2v_dx2 = torch.autograd.grad(
        dv_dx, x, grad_outputs=torch.ones_like(dv_dx), create_graph=True
    )[0]

    # Derivadas temporales
    du_dt = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    dv_dt = torch.autograd.grad(
        v, t, grad_outputs=torch.ones_like(v), create_graph=True
    )[0]

    hbar_t = torch.tensor(hbar, dtype=torch.float32, device=x.device)
    mass_t = torch.tensor(mass, dtype=torch.float32, device=x.device)

    # Definición de la barrera de potencial rectangular
    V = torch.where(
        (x >= x_barrier_left) & (x <= x_barrier_right),
        torch.full_like(x, V0),
        torch.zeros_like(x),
    )

    coeff = (hbar_t**2) / (2.0 * mass_t)

    # Cálculo de los residuos para el sistema acoplado
    res_u = du_dt + coeff * d2v_dx2 - (V / hbar_t) * v
    res_v = dv_dt - coeff * d2u_dx2 + (V / hbar_t) * u

    return torch.mean(res_u**2) + torch.mean(res_v**2)


def initial_condition_loss_tunnel(
    model: nn.Module,
    x: torch.Tensor,
    t0: torch.Tensor,
    x0: float = -3.0,
    sigma: float = 0.5,
    k0: float = 3.0,
    hbar: float = 1.0,
) -> torch.Tensor:
    """
    Impone un paquete de ondas gaussiano complejo como condición inicial en t=0:
    Psi(x,0) = A * exp(-(x-x0)^2 / 2*sigma^2) * exp(i*k0*x)

    Args:
        model (nn.Module): Red neuronal.
        x (torch.Tensor): Puntos de colocación espacial.
        t0 (torch.Tensor): Tensor correspondiente a t=0.
        x0 (float, opcional): Posición central del paquete. Por defecto es -3.0.
        sigma (float, opcional): Anchura del paquete. Por defecto es 0.5.
        k0 (float, opcional): Número de onda inicial. Por defecto es 3.0.
        hbar (float, opcional): Constante reducida de Planck. Por defecto es 1.0.

    Returns:
        torch.Tensor: MSE combinado para las partes real e imaginaria del estado inicial.
    """
    # Amplitud de la envoltura gaussiana y constante de normalización
    gauss = torch.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
    norm = 1.0 / (torch.pi * sigma**2)**0.25
    
    # Separación de Euler para la fase compleja exp(i*k0*x)
    u_true = norm * gauss * torch.cos(k0 * x)
    v_true = norm * gauss * torch.sin(k0 * x)

    u_pred, v_pred = model(x, t0)

    return torch.mean((u_pred - u_true) ** 2) + torch.mean((v_pred - v_true) ** 2)


def boundary_loss_tunnel(
    model: nn.Module,
    t: torch.Tensor,
    x_min: float = -10.0,
    x_max: float = 10.0,
) -> torch.Tensor:
    """
    Impone condiciones de frontera de Dirichlet homogéneas (Psi = 0) en los bordes del dominio.

    Args:
        model (nn.Module): Red neuronal.
        t (torch.Tensor): Puntos de colocación temporal.
        x_min (float, opcional): Frontera espacial izquierda. Por defecto es -10.0.
        x_max (float, opcional): Frontera espacial derecha. Por defecto es 10.0.

    Returns:
        torch.Tensor: MSE de las predicciones en ambas fronteras.
    """
    x_left  = torch.full_like(t, x_min)
    x_right = torch.full_like(t, x_max)

    u_left,  v_left  = model(x_left,  t)
    u_right, v_right = model(x_right, t)

    return (
        torch.mean(u_left**2)  + torch.mean(v_left**2) +
        torch.mean(u_right**2) + torch.mean(v_right**2)
    )


def data_loss_tunnel(
    model: nn.Module,
    x_train: torch.Tensor,
    t_train: torch.Tensor,
    prob_train: torch.Tensor,
) -> torch.Tensor:
    """
    Evalúa el error basado en datos para el modelo de efecto túnel.
    Compara la densidad de probabilidad inferida |Psi_pred|^2 con los datos empíricos o numéricos.

    Args:
        model (nn.Module): Red neuronal.
        x_train (torch.Tensor): Puntos espaciales de entrenamiento.
        t_train (torch.Tensor): Puntos temporales de entrenamiento.
        prob_train (torch.Tensor): Densidad de probabilidad objetivo en (x, t).

    Returns:
        torch.Tensor: MSE respecto a las probabilidades objetivo.
    """
    u, v = model(x_train, t_train)
    prob_pred = u ** 2 + v ** 2
    return torch.mean((prob_pred - prob_train) ** 2)


def normalization_loss_tunnel(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    domain_length: float = 20.0,
) -> torch.Tensor:
    """
    Garantiza la conservación de la probabilidad en el tiempo, forzando a que la integral 
    espacial de |Psi|^2 sea igual a 1 para cada instante t.

    Args:
        model (nn.Module): Red neuronal.
        x (torch.Tensor): Puntos de colocación espacial.
        t (torch.Tensor): Puntos de colocación temporal.
        domain_length (float, opcional): Longitud del dominio de integración. Por defecto es 20.0.

    Returns:
        torch.Tensor: Penalización por la desviación de la norma respecto a la unidad.
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
    Calcula el residuo de la Ecuación del Calor 1D resolviendo un problema inverso,
    donde la difusividad térmica (alpha) es un parámetro a descubrir por la red.
    Ecuación: du/dt - alpha * d^2u/dx^2 = 0

    Args:
        model (nn.Module): Red neuronal que contiene 'alpha' como parámetro entrenable.
        x (torch.Tensor): Puntos de colocación espacial.
        t (torch.Tensor): Puntos de colocación temporal.

    Returns:
        torch.Tensor: Error Cuadrático Medio (MSE) del residuo de la ecuación del calor.
    """
    if not x.requires_grad:
        x.requires_grad_(True)
    if not t.requires_grad:
        t.requires_grad_(True)

    u = model(x, t)

    # Derivadas espaciales y temporales
    du_dt = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True
    )[0]

    # El parámetro alpha se extrae directamente del modelo
    residuo = du_dt - model.alpha * d2u_dx2

    return torch.mean(residuo**2)


def initial_condition_loss_heat(
    model: nn.Module,
    x: torch.Tensor,
    t0: torch.Tensor,
    L: float = 1.0,
) -> torch.Tensor:
    """
    Impone la distribución inicial de temperatura (t=0) como una curva de Gauss centrada en el dominio.

    Args:
        model (nn.Module): Red neuronal.
        x (torch.Tensor): Puntos de colocación espacial.
        t0 (torch.Tensor): Tensor correspondiente a t=0.
        L (float, opcional): Longitud total del dominio espacial. Por defecto es 1.0.

    Returns:
        torch.Tensor: MSE de la predicción respecto a la condición inicial térmica.
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
    Impone condiciones de frontera de Neumann homogéneas (du/dx = 0 en los extremos).
    Físicamente representa extremos térmicamente aislados (flujo de calor nulo).

    Args:
        model (nn.Module): Red neuronal.
        t (torch.Tensor): Puntos de colocación temporal.
        x_min (float, opcional): Frontera espacial izquierda. Por defecto es 0.0.
        x_max (float, opcional): Frontera espacial derecha. Por defecto es 1.0.

    Returns:
        torch.Tensor: MSE de las derivadas espaciales evaluadas en ambas fronteras.
    """
    x_left  = torch.full_like(t, x_min, requires_grad=True)
    x_right = torch.full_like(t, x_max, requires_grad=True)

    u_left  = model(x_left,  t)
    u_right = model(x_right, t)

    # Evaluación del gradiente térmico en los bordes
    du_dx_left = torch.autograd.grad(
        u_left, x_left,
        grad_outputs=torch.ones_like(u_left),
        create_graph=True,
    )[0]

    du_dx_right = torch.autograd.grad(
        u_right, x_right,
        grad_outputs=torch.ones_like(u_right),
        create_graph=True,
    )[0]

    return torch.mean(du_dx_left**2) + torch.mean(du_dx_right**2)