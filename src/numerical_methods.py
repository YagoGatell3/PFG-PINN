import numpy as np
from scipy.linalg import eigh


def solve_schrodinger_fdm(
    x: np.ndarray, V: np.ndarray, mass: float = 1.0, hbar: float = 1.0, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve la Ecuación de Schrödinger 1D independiente del tiempo usando
    el Método de Diferencias Finitas (FDM).

    Convierte el operador Hamiltoniano en una matriz tridiagonal y calcula
    sus autovalores y autovectores.

    Args:
        x (np.ndarray): Array del dominio espacial (discretizado).
        V (np.ndarray): Array con los valores del potencial en cada punto x.
        mass (float): Masa de la partícula.
        hbar (float): Constante reducida de Planck.
        k (int): Número de autovalores/autovectores (estados de energía) a devolver.

    Returns:
        tuple:
            - eigenvalues (np.ndarray): Los primeros 'k' niveles de energía.
            - eigenvectors (np.ndarray): Las 'k' funciones de onda correspondientes (columnas).
    """
    N = len(x)
    dx = x[1] - x[0]

    # 1. Matriz de Energía Cinética (T) usando diferencias finitas centrales
    const_T = -(hbar**2) / (2.0 * mass * dx**2)

    # Diagonal principal (-2) y diagonales superior/inferior (1)
    main_diag = -2.0 * np.ones(N)
    off_diag = 1.0 * np.ones(N - 1)

    T = const_T * (
        np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    )

    # 2. Matriz de Energía Potencial (V)
    V_matrix = np.diag(V)

    # 3. El Hamiltoniano (H = T + V)
    H = T + V_matrix

    # 4. Resolvemos el problema de autovalores (H * psi = E * psi)
    eigenvalues, eigenvectors = eigh(H)

    # 5. Normalizamos las funciones de onda (probabilidad total = 1)
    for i in range(k):
        norm = np.sqrt(np.sum(eigenvectors[:, i] ** 2) * dx)
        eigenvectors[:, i] = eigenvectors[:, i] / norm

        # Por convención, forzamos a que el primer pico sea positivo
        if eigenvectors[np.argmax(np.abs(eigenvectors[:, i])), i] < 0:
            eigenvectors[:, i] *= -1

    return eigenvalues[:k], eigenvectors[:, :k]


def solve_classical_oscillator_rk4(
    t: np.ndarray, mass: float = 1.0, k: float = 1.0, u0: float = 1.0, v0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve el Oscilador Armónico Clásico usando el método Runge-Kutta de 4º Orden (RK4).

    El problema m*u'' + k*u = 0 se reduce a un sistema de primer orden:
        du/dt = v
        dv/dt = -(k/m)*u

    Args:
        t (np.ndarray): Array temporal.
        mass (float): Masa del oscilador.
        k (float): Constante elástica.
        u0 (float): Posición inicial.
        v0 (float): Velocidad inicial.

    Returns:
        tuple: Arrays de posición (u) y velocidad (v).
    """
    dt = t[1] - t[0]
    N = len(t)

    u = np.zeros(N)
    v = np.zeros(N)

    u[0] = u0
    v[0] = v0

    omega_sq = k / mass

    def f(u_val, v_val):
        """Derivadas de estado [du/dt, dv/dt]"""
        return v_val, -omega_sq * u_val

    # Bucle principal de RK4
    for i in range(N - 1):
        u_i = u[i]
        v_i = v[i]

        k1_u, k1_v = f(u_i, v_i)

        k2_u, k2_v = f(u_i + 0.5 * dt * k1_u, v_i + 0.5 * dt * k1_v)
        k3_u, k3_v = f(u_i + 0.5 * dt * k2_u, v_i + 0.5 * dt * k2_v)
        k4_u, k4_v = f(u_i + dt * k3_u, v_i + dt * k3_v)

        u[i + 1] = u_i + (dt / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        v[i + 1] = v_i + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    return u, v

def solve_damped_pendulum_rk4(
    t: np.ndarray, g: float = 9.81, mu: float = 0.5, L: float = 1.0, 
    theta0: float = np.pi/4, omega0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera los datos empíricos del péndulo amortiguado usando RK4.
    """
    dt = t[1] - t[0]
    N = len(t)
    theta = np.zeros(N)
    omega = np.zeros(N) # Velocidad angular
    
    theta[0] = theta0
    omega[0] = omega0

    def f(th, om):
        # d(theta)/dt = omega
        # d(omega)/dt = -mu*omega - (g/L)*sin(theta)
        return om, -mu * om - (g / L) * np.sin(th)

    for i in range(N - 1):
        th_i, om_i = theta[i], omega[i]
        
        k1_th, k1_om = f(th_i, om_i)
        k2_th, k2_om = f(th_i + 0.5 * dt * k1_th, om_i + 0.5 * dt * k1_om)
        k3_th, k3_om = f(th_i + 0.5 * dt * k2_th, om_i + 0.5 * dt * k2_om)
        k4_th, k4_om = f(th_i + dt * k3_th, om_i + dt * k3_om)

        theta[i + 1] = th_i + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[i + 1] = om_i + (dt / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    return theta, omega

def solve_tunnel_crank_nicolson(
    x: np.ndarray,
    t: np.ndarray,
    x0: float = -3.0,
    sigma: float = 0.5,
    k0: float = 3.0,
    V0: float = 1.5,
    x_barrier_left: float = 0.5,
    x_barrier_right: float = 1.5,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    """
    Resuelve la TDSE con el método de Crank-Nicolson (implícito, estable).
    Devuelve |Psi(x,t)|^2 para todos los instantes temporales.

    Args:
        x: Array espacial
        t: Array temporal
        ...parámetros físicos...

    Returns:
        prob: Array (len(t), len(x)) con la densidad de probabilidad en cada instante.
    """
    from scipy.linalg import solve_banded

    Nx = len(x)
    Nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Condición inicial: paquete gaussiano complejo
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    psi = norm * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)

    # Potencial
    V = np.where((x >= x_barrier_left) & (x <= x_barrier_right), V0, 0.0)

    # Coeficiente de Crank-Nicolson
    r = 1j * hbar * dt / (4.0 * mass * dx**2)

    # Diagonales de las matrices tridiagonales A (izquierda) y B (derecha)
    diag_A  =  (1.0 + 2.0*r + 1j * dt * V / (2.0 * hbar)) * np.ones(Nx, dtype=complex)
    off_A   = -r * np.ones(Nx - 1, dtype=complex)
    diag_B  =  (1.0 - 2.0*r - 1j * dt * V / (2.0 * hbar)) * np.ones(Nx, dtype=complex)
    off_B   =  r * np.ones(Nx - 1, dtype=complex)

    # Formato banded para scipy (3 filas: sup, diag, inf)
    A_banded = np.zeros((3, Nx), dtype=complex)
    A_banded[0, 1:]  = off_A    # superdiagonal
    A_banded[1, :]   = diag_A   # diagonal
    A_banded[2, :-1] = off_A    # subdiagonal

    prob = np.zeros((Nt, Nx))
    prob[0] = np.abs(psi) ** 2

    for n in range(Nt - 1):
        # Lado derecho: B * psi_n
        rhs = diag_B * psi
        rhs[1:]  += off_B * psi[:-1]
        rhs[:-1] += off_B * psi[1:]

        # Condiciones de contorno de Dirichlet
        rhs[0]  = 0.0
        rhs[-1] = 0.0
        A_banded[1, 0]  = 1.0
        A_banded[1, -1] = 1.0
        A_banded[0, 1]  = 0.0
        A_banded[2, -2] = 0.0

        psi = solve_banded((1, 1), A_banded, rhs)
        prob[n + 1] = np.abs(psi) ** 2

    return prob