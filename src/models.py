import torch
import torch.nn as nn


class PINN(nn.Module):
    """
    Perceptrón Multicapa (MLP) base para la Red Neuronal Informada por la Física (PINN).
    Aproxima la función de onda cuántica y descubre el autovalor de energía.
    """

    def __init__(self, hidden_layers: list[int]):
        """
        Inicializa la arquitectura de la red neuronal.

        Args:
            hidden_layers (List[int]): Lista con el número de neuronas por cada capa oculta.
            Ejemplo: [32, 32, 32]
        """
        super(PINN, self).__init__()

        layers = []
        # Capa de entrada (x unidimensional)
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.Tanh())

        # Capas ocultas
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())

        # Capa de salida (predicción de la función de onda psi)
        layers.append(nn.Linear(hidden_layers[-1], 1))

        self.net = nn.Sequential(*layers)

        # Autovalor para entrenar de la función de pérdida física (Epsilon)
        self.epsilon = nn.Parameter(
            torch.tensor(0.0),
            requires_grad=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paso hacia adelante (forward pass) de la red.

        Args:
            x (torch.Tensor): Coordenada espacial.

        Returns:
            torch.Tensor: Predicción de la función de onda en x.
        """
        return self.net(x)
    
    
class PINNDampedPendulum(nn.Module):
    """
    PINN para el Péndulo Amortiguado Clásico (Problema Inverso).
    Aproxima la posición angular theta(t) y descubre la gravedad (g) 
    y el factor de amortiguamiento (mu).
    """
    def __init__(self, hidden_layers: list[int]):
        super(PINNDampedPendulum, self).__init__()
        
        layers = []
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.Tanh())
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)
        
        # Parámetros físicos a descubrir (g y mu)
        self.g = nn.Parameter(torch.tensor(5.0), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

class PINNTunnel(nn.Module):
    """
    PINN para la Ecuación de Schrödinger Dependiente del Tiempo (2D: x, t).
    Salida: dos componentes (parte real u y parte imaginaria v) de la función de onda.
    """
    def __init__(self, hidden_layers: list[int]):
        super(PINNTunnel, self).__init__()
        layers = []
        # Entrada: 2 neuronas (x, t)
        layers.append(nn.Linear(2, hidden_layers[0]))
        layers.append(nn.Tanh())
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())
        # Salida: 2 neuronas (u = Re(Psi), v = Im(Psi))
        layers.append(nn.Linear(hidden_layers[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Coordenada espacial (N, 1)
            t: Coordenada temporal (N, 1)
        Returns:
            u: Parte real de Psi (N, 1)
            v: Parte imaginaria de Psi (N, 1)
        """
        inp = torch.cat([x, t], dim=1)  # (N, 2)
        out = self.net(inp)             # (N, 2)
        u = out[:, 0:1]
        v = out[:, 1:2]
        return u, v
    
class PINNHeatInverse(nn.Module):
    """
    PINN para la Ecuación del Calor 1D — Problema Inverso.
    Aprende u(x,t) y descubre la difusividad térmica alpha.
    """
    def __init__(self, hidden_layers: list[int], alpha_init: float = 0.5):
        super(PINNHeatInverse, self).__init__()
        layers = []
        layers.append(nn.Linear(2, hidden_layers[0]))
        layers.append(nn.Tanh())
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)

        # Parámetro físico a descubrir
        self.alpha = nn.Parameter(
            torch.tensor(alpha_init), requires_grad=True
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)
    
class PINNDynamic(nn.Module):
    """
    PINN para problemas dinámicos temporales (EDOs clásicas).
    Sin parámetro epsilon — la salida es directamente la posición u(t).
    Usada para: Oscilador Clásico.
    """
    def __init__(self, hidden_layers: list[int]):
        super(PINNDynamic, self).__init__()
        layers = []
        layers.append(nn.Linear(1, hidden_layers[0]))
        layers.append(nn.Tanh())
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)