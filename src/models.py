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
