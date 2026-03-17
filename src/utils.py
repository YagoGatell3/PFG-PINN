import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch


def update_dynamic_weights(
    data_loss: torch.Tensor,
    ph_loss: torch.Tensor,
    bound_loss: torch.Tensor,
    last_layer_weight: torch.nn.Parameter,
    current_lambda_ph: float,
    current_lambda_bound: float,
    alpha: float = 0.9,
) -> tuple[float, float]:
    """
    Calcula y actualiza los pesos dinámicos de las funciones de pérdida.
    """
    # Cálculo de los gradientes de cada pérdida con respecto a los pesos de la última capa
    grad_data = torch.autograd.grad(data_loss, last_layer_weight, retain_graph=True)[0]
    grad_ph = torch.autograd.grad(ph_loss, last_layer_weight, retain_graph=True)[0]
    grad_bound = torch.autograd.grad(bound_loss, last_layer_weight, retain_graph=True)[
        0
    ]

    # Magnitudes de los gradientes
    max_grad_data = torch.max(torch.abs(grad_data))
    mean_grad_ph = torch.mean(torch.abs(grad_ph))
    mean_grad_bound = torch.mean(torch.abs(grad_bound))

    # Pesos objetivo para equilibrar las pérdidas
    hat_lambda_ph = max_grad_data / (mean_grad_ph + 1e-8)
    hat_lambda_bound = max_grad_data / (mean_grad_bound + 1e-8)

    # Actualización suave de los pesos con un factor de suavizado alpha
    new_lambda_ph = (1 - alpha) * current_lambda_ph + alpha * hat_lambda_ph.item()
    new_lambda_bound = (
        1 - alpha
    ) * current_lambda_bound + alpha * hat_lambda_bound.item()

    return new_lambda_ph, new_lambda_bound


def calculate_l2_error(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    """
    Calcula el error relativo L2 entre la predicción de la PINN y la solución exacta.

    Args:
        u_pred (torch.Tensor): Predicción del modelo.
        u_true (torch.Tensor): Solución analítica verdadera.

    Returns:
        float: Valor del error relativo L2.
    """
    error = torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)
    return error.item()


def save_experiment_results(
    config: dict, final_results: dict, history: dict, save_dir: str = "results"
):
    """
    Guarda los hiperparámetros, resultados finales y el historial de pérdida
    en un archivo JSON para su posterior análisis.
    """
    os.makedirs(save_dir, exist_ok=True)

    experimento = {
        "config": config,
        "resultados_finales": final_results,
        "historial": history,
    }

    # Generación del nombre básado en la configuración
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"exp_n{config['estado_n']}_{config['sampler']}_{timestamp}.json"
    ruta_completa = os.path.join(save_dir, nombre_archivo)

    with open(ruta_completa, "w") as f:
        json.dump(experimento, f, indent=4)

    print(f"Resultados guardados exitosamente en: {ruta_completa}")


def set_seed(seed: int = 42):
    """
    Fija la semilla aleatoria para garantizar la reproducibilidad de los experimentos.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_and_save_results(
    pinn_model: torch.nn.Module,
    x_train: torch.Tensor,
    u_train: torch.Tensor,
    x_eval: torch.Tensor,
    u_true: torch.Tensor,
    epoch: int,
    pinn_loss: float,
    n: int = 0,
    save_dir: str = "../img",
):
    """
    Genera y guarda la gráfica del estado actual de la predicción de la PINN.
    """
    # Asegurar que el directorio existe
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Desconectar del grafo computacional para poder plotear
    pinn_pred = pinn_model(x_eval).detach()

    # 1. Solución Analítica
    ax.plot(
        x_eval.detach().numpy(),
        u_true.detach().numpy(),
        label="Solución Analítica",
        color="blue",
        linewidth=2,
        alpha=0.5,
    )

    # 2. Predicción de la PINN
    ax.plot(
        x_eval.detach().numpy(),
        pinn_pred.numpy(),
        label="Predicción PINN",
        linestyle="--",
        color="black",
        linewidth=2,
    )

    # 3. Puntos de entrenamiento empíricos
    if x_train is not None and u_train is not None:
        ax.scatter(
            x_train.detach().numpy(),
            u_train.detach().numpy(),
            color="red",
            label="Datos de Entrenamiento",
            s=50,
            zorder=5,
        )

    ax.set_title(
        f"PINN (Oscilador Armónico) - Época {epoch} | Pérdida Total: {pinn_loss:.4e}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("ψ(x)")
    ax.set_ylim(-1, 1.5)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # Guardar la figura
    nombre_archivo = f"PINN_resultado_estado_n{n}_epoch_{epoch}.png"
    ruta_completa = os.path.join(save_dir, nombre_archivo)
    plt.savefig(ruta_completa, dpi=300)
    plt.close()

    print(f"Gráfica guardada en: {ruta_completa}")
