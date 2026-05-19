import numpy as np
from scipy.linalg import eigh


def solve_schrodinger_fdm(
    x: np.ndarray, V: np.ndarray, mass: float = 1.0, hbar: float = 1.0, k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve la Ecuación de Schrödinger 1D independiente del tiempo utilizando 
    el Método de Diferencias Finitas (FDM). Convierte el operador Hamiltoniano 
    en una matriz tridiagonal y calcula sus autovalores y autovectores.

    Args:
        x (np.ndarray): Array del dominio espacial discretizado.
        V (np.ndarray): Array con los valores del potencial evaluados en cada punto de x.
        mass (float, opcional): Masa de la partícula. Por defecto es 1.0.
        hbar (float, opcional): Constante reducida de Planck. Por defecto es 1.0.
        k (int, opcional): Número de estados de energía (autovalores/autovectores) a devolver. Por defecto es 5.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tupla que contiene:
            - eigenvalues (np.ndarray): Los primeros 'k' niveles de energía.
            - eigenvectors (np.ndarray): Las 'k' funciones de onda correspondientes (dispuestas por columnas).
    """
    N = len(x)
    dx = x[1] - x[0]

    # Construcción de la Matriz de Energía Cinética (T) usando diferencias finitas centrales
    const_T = -(hbar**2) / (2.0 * mass * dx**2)

    # Definición de la diagonal principal (-2) y las diagonales superior/inferior (1)
    main_diag = -2.0 * np.ones(N)
    off_diag = 1.0 * np.ones(N - 1)

    T = const_T * (
        np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    )

    # Matriz de Energía Potencial (V)
    V_matrix = np.diag(V)

    # Construcción del Hamiltoniano total (H = T + V)
    H = T + V_matrix

    # Resolución del problema de autovalores (H * psi = E * psi)
    eigenvalues, eigenvectors = eigh(H)

    # Normalización de las funciones de onda para garantizar probabilidad total = 1
    for i in range(k):
        norm = np.sqrt(np.sum(eigenvectors[:, i] ** 2) * dx)
        eigenvectors[:, i] = eigenvectors[:, i] / norm

        # Convención de fase: forzar a que el primer pico de amplitud sea positivo
        if eigenvectors[np.argmax(np.abs(eigenvectors[:, i])), i] < 0:
            eigenvectors[:, i] *= -1

    return eigenvalues[:k], eigenvectors[:, :k]


def solve_classical_oscillator_rk4(
    t: np.ndarray, mass: float = 1.0, k: float = 1.0, u0: float = 1.0, v0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve la ecuación de movimiento del Oscilador Armónico Clásico utilizando 
    el método numérico de Runge-Kutta de 4º Orden (RK4).
    Reduce el problema de segundo orden (m*u'' + k*u = 0) a un sistema de primer orden:
        du/dt = v
        dv/dt = -(k/m)*u

    Args:
        t (np.ndarray): Array temporal discretizado.
        mass (float, opcional): Masa del oscilador. Por defecto es 1.0.
        k (float, opcional): Constante elástica del sistema. Por defecto es 1.0.
        u0 (float, opcional): Posición inicial en t=0. Por defecto es 1.0.
        v0 (float, opcional): Velocidad inicial en t=0. Por defecto es 0.0.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays de posición (u) y velocidad (v) a lo largo del tiempo.
    """
    dt = t[1] - t[0]
    N = len(t)

    u = np.zeros(N)
    v = np.zeros(N)

    u[0] = u0
    v[0] = v0

    omega_sq = k / mass

    def f(u_val, v_val):
        """Función que define el sistema de derivadas de estado [du/dt, dv/dt]."""
        return v_val, -omega_sq * u_val

    # Bucle principal de integración RK4
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
    Resuelve la dinámica de un péndulo amortiguado utilizando el método de Runge-Kutta 
    de 4º Orden (RK4) para generar datos empíricos.

    Args:
        t (np.ndarray): Array temporal discretizado.
        g (float, opcional): Aceleración gravitacional. Por defecto es 9.81.
        mu (float, opcional): Coeficiente de amortiguamiento. Por defecto es 0.5.
        L (float, opcional): Longitud del péndulo. Por defecto es 1.0.
        theta0 (float, opcional): Ángulo inicial en radianes. Por defecto es pi/4.
        omega0 (float, opcional): Velocidad angular inicial. Por defecto es 0.0.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays que representan la posición angular (theta) 
        y la velocidad angular (omega) a lo largo del tiempo.
    """
    dt = t[1] - t[0]
    N = len(t)
    
    theta = np.zeros(N)
    omega = np.zeros(N)
    
    theta[0] = theta0
    omega[0] = omega0

    def f(th, om):
        """Sistema de ecuaciones diferenciales: d(theta)/dt y d(omega)/dt."""
        return om, -mu * om - (g / L) * np.sin(th)

    # Bucle principal de integración RK4
    for i in range(N - 1):
        th_i, om_i = theta[i], omega[i]
        
        k1_th, k1_om = f(th_i, om_i)
        k2_th, k2_om = f(th_i + 0.5 * dt * k1_th, om_i + 0.5 * dt * k1_om)
        k3_th, k3_om = f(th_i + 0.5 * dt * k2_th, om_i + 0.5 * dt * k2_om)
        k4_th, k4_om = f(th_i + dt * k3_th, om_i + dt * k3_om)

        theta[i + 1] = th_i + (dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
        omega[i + 1] = om_i + (dt / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)

    return theta, omega


def solve_heat_crank_nicolson(
    x: np.ndarray,
    t: np.ndarray,
    alpha: float = 0.1,
    L: float = 1.0,
) -> np.ndarray:
    """
    Resuelve numéricamente la Ecuación del Calor 1D utilizando el esquema implícito 
    incondicionalmente estable de Crank-Nicolson, con condiciones de contorno de Neumann.

    Args:
        x (np.ndarray): Array del dominio espacial discretizado.
        t (np.ndarray): Array temporal discretizado.
        alpha (float, opcional): Difusividad térmica. Por defecto es 0.1.
        L (float, opcional): Longitud del dominio espacial. Por defecto es 1.0.

    Returns:
        np.ndarray: Matriz solución 2D (len(t), len(x)) con el perfil térmico u(x,t).
    """
    from scipy.linalg import solve_banded

    Nx = len(x)
    Nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Condición inicial: Distribución gaussiana centrada
    x0    = L / 2.0
    sigma = L / 8.0
    u     = np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))

    # Constante característica del esquema numérico
    r = alpha * dt / (2.0 * dx ** 2)

    # --- Construcción de la Matriz Tridiagonal A en formato banded ---
    A_banded = np.zeros((3, Nx))
    
    # Diagonal principal (fila 1 del formato banded)
    A_banded[1, :] = 1.0 + 2.0 * r
    
    # Superdiagonal (fila 0 del formato banded, corresponde a A[i, i+1])
    A_banded[0, 1:] = -r
    A_banded[0, 1]  = -2.0 * r  # Modificación para frontera izquierda de Neumann
    
    # Subdiagonal (fila 2 del formato banded, corresponde a A[i, i-1])
    A_banded[2, :-1] = -r
    A_banded[2, -2]  = -2.0 * r # Modificación para frontera derecha de Neumann

    sol    = np.zeros((Nt, Nx))
    sol[0] = u.copy()

    # Bucle de avance temporal
    for n in range(Nt - 1):
        # Lado derecho de la ecuación (RHS = B * u^n)
        rhs = (1.0 - 2.0 * r) * u.copy()

        # Nodos interiores
        rhs[1:-1] += r * (u[:-2] + u[2:])

        # Frontera izquierda (Neumann): u_{-1} = u_1 -> contribución 2r*u_1
        rhs[0] += 2.0 * r * u[1]

        # Frontera derecha (Neumann): u_N = u_{N-2} -> contribución 2r*u_{N-2}
        rhs[-1] += 2.0 * r * u[-2]

        # Resolución del sistema tridiagonal
        u = solve_banded((1, 1), A_banded, rhs)
        sol[n + 1] = u.copy()

    return sol


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
    Resuelve la Ecuación de Schrödinger Dependiente del Tiempo (TDSE) para el problema 
    del efecto túnel cuántico utilizando el método de Crank-Nicolson.

    Args:
        x (np.ndarray): Array del dominio espacial discretizado.
        t (np.ndarray): Array temporal discretizado.
        x0 (float, opcional): Posición inicial del paquete de ondas. Por defecto es -3.0.
        sigma (float, opcional): Anchura inicial del paquete. Por defecto es 0.5.
        k0 (float, opcional): Número de onda inicial. Por defecto es 3.0.
        V0 (float, opcional): Altura de la barrera de potencial. Por defecto es 1.5.
        x_barrier_left (float, opcional): Posición inicial de la barrera. Por defecto es 0.5.
        x_barrier_right (float, opcional): Posición final de la barrera. Por defecto es 1.5.
        mass (float, opcional): Masa de la partícula. Por defecto es 1.0.
        hbar (float, opcional): Constante reducida de Planck. Por defecto es 1.0.

    Returns:
        np.ndarray: Array 2D (len(t), len(x)) con la evolución espacio-temporal de la 
        densidad de probabilidad |Psi(x,t)|^2.
    """
    from scipy.linalg import solve_banded

    Nx = len(x)
    Nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Generación de la condición inicial (paquete gaussiano complejo)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    psi = norm * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2)) * np.exp(1j * k0 * x)

    # Definición espacial de la barrera de potencial
    V = np.where((x >= x_barrier_left) & (x <= x_barrier_right), V0, 0.0)

    # Parámetro complejo de estabilidad
    r = 1j * hbar * dt / (4.0 * mass * dx**2)

    # Componentes de las matrices tridiagonales A (implícita) y B (explícita)
    diag_A  =  (1.0 + 2.0*r + 1j * dt * V / (2.0 * hbar)) * np.ones(Nx, dtype=complex)
    off_A   = -r * np.ones(Nx - 1, dtype=complex)
    
    diag_B  =  (1.0 - 2.0*r - 1j * dt * V / (2.0 * hbar)) * np.ones(Nx, dtype=complex)
    off_B   =  r * np.ones(Nx - 1, dtype=complex)

    # Construcción de la matriz A en formato banded para scipy_linalg (superior, diagonal, inferior)
    A_banded = np.zeros((3, Nx), dtype=complex)
    A_banded[0, 1:]  = off_A    # Superdiagonal
    A_banded[1, :]   = diag_A   # Diagonal principal
    A_banded[2, :-1] = off_A    # Subdiagonal

    prob = np.zeros((Nt, Nx))
    prob[0] = np.abs(psi) ** 2

    # Bucle de avance temporal
    for n in range(Nt - 1):
        # Evaluación del lado derecho (RHS = B * psi_n)
        rhs = diag_B * psi
        rhs[1:]  += off_B * psi[:-1]
        rhs[:-1] += off_B * psi[1:]

        # Imposición de condiciones de contorno de Dirichlet (nulas en los extremos)
        rhs[0]  = 0.0
        rhs[-1] = 0.0
        A_banded[1, 0]  = 1.0
        A_banded[1, -1] = 1.0
        A_banded[0, 1]  = 0.0
        A_banded[2, -2] = 0.0

        # Resolución del sistema lineal y almacenamiento de la probabilidad
        psi = solve_banded((1, 1), A_banded, rhs)
        prob[n + 1] = np.abs(psi) ** 2

    return prob