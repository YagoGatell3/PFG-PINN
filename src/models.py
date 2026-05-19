import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Arquitectura base de Perceptrón Multicapa (MLP) para una Red Neuronal Informada por la Física (PINN).
    Diseñada para aproximar la función de onda cuántica y descubrir simultáneamente el autovalor de energía.
    """

    def __init__(self, hidden_layers: list[int]):
        """
        Inicializa la estructura de la red neuronal y los parámetros físicos entrenables.

        Args:
            hidden_layers (list[int]): Lista que define el número de neuronas en cada capa oculta.
                                       Ejemplo: [32, 32, 32] para tres capas de 32 neuronas.
        """
        super(PINN, self).__init__()

        layers = []
        # Capa de entrada (coordenada espacial unidimensional)
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.Tanh())

        # Construcción de capas ocultas dinámicas
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())

        # Capa de salida (predicción del valor escalar de la función de onda psi)
        layers.append(nn.Linear(hidden_layers[-1], 1))

        self.net = nn.Sequential(*layers)

        # Autovalor de energía (Epsilon) a optimizar mediante la función de pérdida física
        self.epsilon = nn.Parameter(
            torch.tensor(0.0),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el paso hacia adelante (forward pass) de la red.

        Args:
            x (torch.Tensor): Tensor con las coordenadas espaciales.

        Returns:
            torch.Tensor: Predicción de la red para la función de onda evaluada en x.
        """
        return self.net(x)


class PINNWell(nn.Module):
    """
    Red Neuronal Informada por la Física para el problema del Pozo de Potencial Infinito 1D.
    Aproxima la función de onda para un estado energético específico, asumiendo el autovalor 
    de energía como un parámetro fijo y conocido (no entrenable).
    """

    def __init__(self, hidden_layers: list[int], epsilon_init: float = 0.0):
        """
        Inicializa la red y establece el nivel de energía fijo.

        Args:
            hidden_layers (list[int]): Topología de las capas ocultas.
            epsilon_init (float, opcional): Nivel de energía exacto conocido. Por defecto es 0.0.
        """
        super(PINNWell, self).__init__()

        layers = []
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.Tanh())

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)

        # Autovalor fijo para el estado (no registrado como parámetro optimizable)
        self.epsilon = nn.Parameter(
            torch.tensor(epsilon_init),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el paso hacia adelante (forward pass).

        Args:
            x (torch.Tensor): Tensor de entrada espacial.

        Returns:
            torch.Tensor: Predicción de la función de onda.
        """
        return self.net(x)


class PINNDampedPendulum(nn.Module):
    """
    Red Neuronal Informada por la Física orientada al Péndulo Amortiguado Clásico (Problema Inverso).
    Aproxima la evolución temporal de la posición angular theta(t) y, simultáneamente, 
    descubre los parámetros físicos del sistema: gravedad (g) y factor de amortiguamiento (mu).
    """

    def __init__(self, hidden_layers: list[int]):
        """
        Inicializa la arquitectura de la red y los parámetros físicos de búsqueda.

        Args:
            hidden_layers (list[int]): Topología de las capas ocultas.
        """
        super(PINNDampedPendulum, self).__init__()

        layers = []
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.Tanh())

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)

        # Parámetros físicos dinámicos a descubrir por retropropagación
        self.g = nn.Parameter(torch.tensor(6.0), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor(0.25), requires_grad=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Predice el ángulo del péndulo en un instante dado.

        Args:
            t (torch.Tensor): Tensor temporal.

        Returns:
            torch.Tensor: Ángulo theta(t) predicho.
        """
        return self.net(t)


class PINNTunnel(nn.Module):
    """
    Red Neuronal Informada por la Física para la Ecuación de Schrödinger Dependiente del Tiempo (TDSE).
    Implementa 'Fourier Feature Encoding' para mapear las entradas espacio-temporales a un espacio de 
    alta frecuencia, facilitando el aprendizaje de los componentes oscilatorios de la función de onda.
    """

    def __init__(
        self,
        hidden_layers: list[int],
        n_fourier: int = 64,
        sigma_fourier: float = 2.0,
    ):
        """
        Inicializa la red y la matriz de proyección de Fourier.

        Args:
            hidden_layers (list[int]): Topología de las capas ocultas del MLP.
            n_fourier (int, opcional): Número de características de Fourier a generar. Por defecto es 64.
            sigma_fourier (float, opcional): Varianza para la distribución gaussiana de la matriz de codificación. Por defecto es 2.0.
        """
        super(PINNTunnel, self).__init__()

        # --- Codificación de Fourier (Fourier Encoding) ---
        # Matriz B fija (no entrenable) para proyectar el vector (x, t) en el dominio de frecuencias
        B = torch.randn(2, n_fourier) * sigma_fourier
        self.register_buffer("B", B)  # Almacenamiento seguro; permite migración a GPU vía .to(device)

        # La dimensión de entrada se duplica al aplicar funciones seno y coseno
        input_dim = 2 * n_fourier

        # --- Construcción del Perceptrón Multicapa (MLP) ---
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.Tanh())
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())
        # Capa de salida bidimensional: parte real (u) e imaginaria (v)
        layers.append(nn.Linear(hidden_layers[-1], 2))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mapea las entradas al dominio de Fourier y predice la función de onda compleja.

        Args:
            x (torch.Tensor): Tensor de coordenadas espaciales (N, 1).
            t (torch.Tensor): Tensor de coordenadas temporales (N, 1).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tupla conteniendo las partes real (u) e imaginaria (v).
        """
        inp = torch.cat([x, t], dim=1)                 # Concatenación espacio-temporal (N, 2)
        z = torch.matmul(inp, self.B)                  # Proyección lineal (N, n_fourier)
        z = torch.cat([torch.cos(z), torch.sin(z)], dim=1)  # Activación trigonométrica (N, 2*n_fourier)
        
        out = self.net(z)                              # Inferencia a través del MLP (N, 2)
        
        return out[:, 0:1], out[:, 1:2]


class PINNHeatInverse(nn.Module):
    """
    Red Neuronal Informada por la Física orientada a la Ecuación del Calor 1D (Problema Inverso).
    Diseñada para inferir la distribución espacio-temporal de temperatura u(x,t) mientras 
    descubre iterativamente el coeficiente de difusividad térmica (alpha).
    """

    def __init__(self, hidden_layers: list[int], alpha_init: float = 0.5):
        """
        Inicializa la red y el parámetro de difusividad entrenable.

        Args:
            hidden_layers (list[int]): Topología de las capas ocultas.
            alpha_init (float, opcional): Valor de inicialización para la difusividad térmica. Por defecto es 0.5.
        """
        super(PINNHeatInverse, self).__init__()
        
        layers = []
        # La entrada comprende coordenadas espaciales y temporales
        layers.append(nn.Linear(2, hidden_layers[0]))
        layers.append(nn.Tanh())
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)

        # Coeficiente físico a descubrir mediante la minimización del residuo
        self.alpha = nn.Parameter(
            torch.tensor(alpha_init), requires_grad=True
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evalúa el perfil térmico en el dominio espacio-temporal.

        Args:
            x (torch.Tensor): Puntos de evaluación espaciales.
            t (torch.Tensor): Puntos de evaluación temporales.

        Returns:
            torch.Tensor: Predicción de temperatura u(x,t).
        """
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


class PINNDynamic(nn.Module):
    """
    Red Neuronal Informada por la Física simplificada para problemas dinámicos puramente temporales,
    tales como Ecuaciones Diferenciales Ordinarias (EDOs) de osciladores clásicos.
    A diferencia de los modelos cuánticos, carece del parámetro de autovalor (epsilon).
    """

    def __init__(self, hidden_layers: list[int]):
        """
        Inicializa la red neuronal.

        Args:
            hidden_layers (list[int]): Topología de las capas ocultas.
        """
        super(PINNDynamic, self).__init__()
        
        layers = []
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.Tanh())
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Predice la trayectoria dinámica del sistema.

        Args:
            t (torch.Tensor): Tensor de entrada temporal.

        Returns:
            torch.Tensor: Predicción del estado del sistema u(t).
        """
        return self.net(t)