import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.numerical_methods import (
    solve_classical_oscillator_rk4,
    solve_damped_pendulum_rk4,
    solve_schrodinger_fdm,
    solve_tunnel_crank_nicolson,
)


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
    grad_data = torch.autograd.grad(
        data_loss, last_layer_weight, retain_graph=True, allow_unused=True
    )[0]
    grad_ph = torch.autograd.grad(
        ph_loss, last_layer_weight, retain_graph=True, allow_unused=True
    )[0]
    grad_bound = torch.autograd.grad(
        bound_loss, last_layer_weight, retain_graph=True, allow_unused=True
    )[0]

    if grad_data is None:
        return current_lambda_ph, current_lambda_bound

    # Magnitudes de los gradientes
    max_grad_data = torch.max(torch.abs(grad_data))

    # Actualización de lambda_ph si el gradiente existe (para evitar errores en casos donde ph_loss no se use)
    new_lambda_ph = current_lambda_ph
    if grad_ph is not None:
        # Usamos la media del gradiente de ph_loss para suavizar la actualización y evitar cambios bruscos
        mean_grad_ph = torch.mean(torch.abs(grad_ph))
        hat_lambda_ph = max_grad_data / (mean_grad_ph + 1e-8)
        new_lambda_ph = (1 - alpha) * current_lambda_ph + alpha * hat_lambda_ph.item()

    # Actualización de lambda_bound si el gradiente existe (para evitar errores en casos donde bound_loss no se use)
    new_lambda_bound = current_lambda_bound
    if grad_bound is not None:
        # Usamos la media del gradiente de bound_loss para suavizar la actualización y evitar cambios bruscos
        mean_grad_bound = torch.mean(torch.abs(grad_bound))
        hat_lambda_bound = max_grad_data / (mean_grad_bound + 1e-8)
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

    sistema = config.get("sistema", "qho")
    estado_n = config.get("estado_n", 0)

    ruta_directorio = os.path.join(save_dir, sistema, f"estado_{estado_n}")
    os.makedirs(ruta_directorio, exist_ok=True)

    experimento = {
        "config": config,
        "resultados_finales": final_results,
        "historial": history,
    }

    # Generación del nombre básado en la configuración
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"exp_{config['sampler']}_{timestamp}.json"
    ruta_completa = os.path.join(ruta_directorio, nombre_archivo)

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
    sistema: str = "qho",
):
    """
    Genera y guarda la gráfica del estado actual de la predicción de la PINN,
    adaptando los títulos y ejes al sistema físico correspondiente.
    """
    # Asegurar que el directorio existe
    ruta_directorio = os.path.join(save_dir, sistema, f"estado_{n}")
    os.makedirs(ruta_directorio, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Desconectar del grafo computacional para poder plotear
    pinn_pred = pinn_model(x_eval).detach()
    x_eval_np = x_eval.detach().numpy()
    u_true_np = u_true.detach().numpy()
    pred_np = pinn_pred.numpy()

    # 1. Solución Analítica
    ax.plot(
        x_eval_np,
        u_true_np,
        label="Solución Analítica",
        color="blue",
        linewidth=2,
        alpha=0.5,
    )

    # 2. Predicción de la PINN
    ax.plot(
        x_eval_np,
        pred_np,
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

    # --- LÓGICA DINÁMICA SEGÚN EL SISTEMA ---
    if sistema == "qho":
        titulo = f"PINN (Oscilador Armónico Cuántico) - Estado n={n} | Época {epoch}"
        ax.set_xlabel("x (Posición)")
        ax.set_ylabel("ψ(x) (Función de onda)")
        # Dejamos que matplotlib ajuste el Y automático o lo forzamos un poco más amplio
        ax.set_ylim(-1.5, 1.5)

    elif sistema == "pozo_infinito":
        titulo = f"PINN (Pozo Potencial Infinito) - Estado n={n} | Época {epoch}"
        ax.set_xlabel("x (Posición dentro del pozo)")
        ax.set_ylabel("ψ(x) (Función de onda)")
        ax.set_ylim(-2.0, 2.0)

    elif sistema == "oscilador_clasico":
        titulo = f"PINN (Oscilador Clásico) | Época {epoch}"
        ax.set_xlabel("t (Tiempo)")
        ax.set_ylabel("u(t) (Posición)")
        ax.set_ylim(-1.5, 1.5)

    elif sistema == "pendulo_inverso":
        titulo = f"PINN (Péndulo Inverso) | Época {epoch}"
        ax.set_xlabel("t (Tiempo)")
        ax.set_ylabel("θ(t) (Ángulo)")
        ax.set_ylim(-2.0, 2.0)

    else:
        titulo = f"PINN ({sistema}) | Época {epoch}"
        ax.set_xlabel("Entrada")
        ax.set_ylabel("Salida")

    # Título común con la pérdida
    ax.set_title(f"{titulo}\nPérdida Total: {pinn_loss:.4e}")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # Guardar la figura
    nombre_archivo = f"epoch_{epoch:05d}.png"
    ruta_completa = os.path.join(ruta_directorio, nombre_archivo)
    plt.savefig(ruta_completa, dpi=300)
    plt.close()


class Timer:
    """
    Contexto para medir tiempos de ejecución de forma limpia.

    Uso:
        with Timer() as t:
            ... código a medir ...
        print(t.elapsed)
    """

    def __enter__(self):
        self.start = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start


def measure_numerical_reference(
    sistema: str,
    x_or_t: np.ndarray,
    **kwargs,
) -> dict:
    """
    Ejecuta el método numérico de referencia para cada sistema y mide su tiempo.

    Args:
        sistema: Identificador del sistema físico.
                 Opciones: 'qho', 'pozo_infinito', 'oscilador_clasico',
                           'pendulo_inverso', 'tunnel', 'heat_inverse'
        x_or_t:  Array espacial (problemas estacionarios) o temporal (dinámicos).
        **kwargs: Parámetros físicos específicos del sistema.

    Returns:
        dict con claves:
            - 'solution': resultado del método numérico
            - 'time_s':   tiempo de ejecución en segundos
            - 'method':   nombre del método usado
    """
    with Timer() as t:
        if sistema == "qho":
            mass = kwargs.get("mass", 1.0)
            omega = kwargs.get("omega", 1.0)
            hbar = kwargs.get("hbar", 1.0)
            k = kwargs.get("k", 5)
            V = 0.5 * mass * omega**2 * x_or_t**2
            solution = solve_schrodinger_fdm(x_or_t, V, mass=mass, hbar=hbar, k=k)
            method = "FDM (Diferencias Finitas)"

        elif sistema == "pozo_infinito":
            mass = kwargs.get("mass", 1.0)
            hbar = kwargs.get("hbar", 1.0)
            k = kwargs.get("k", 5)
            V = np.zeros_like(x_or_t)  # V=0 dentro del pozo
            solution = solve_schrodinger_fdm(x_or_t, V, mass=mass, hbar=hbar, k=k)
            method = "FDM (Diferencias Finitas)"

        elif sistema == "oscilador_clasico":
            mass = kwargs.get("mass", 1.0)
            k_spring = kwargs.get("k", 1.0)
            u0 = kwargs.get("u0", 1.0)
            v0 = kwargs.get("v0", 0.0)
            solution = solve_classical_oscillator_rk4(
                x_or_t, mass=mass, k=k_spring, u0=u0, v0=v0
            )
            method = "RK4"

        elif sistema == "pendulo_inverso":
            g = kwargs.get("g", 9.81)
            mu = kwargs.get("mu", 0.5)
            L = kwargs.get("L", 1.0)
            theta0 = kwargs.get("theta0", np.pi / 4)
            omega0 = kwargs.get("omega0", 0.0)
            solution = solve_damped_pendulum_rk4(
                x_or_t, g=g, mu=mu, L=L, theta0=theta0, omega0=omega0
            )
            method = "RK4"

        elif sistema == "tunnel":
            solution = solve_tunnel_crank_nicolson(
                x_or_t,
                kwargs.get("t_array"),
                x0=kwargs.get("x0", -4.0),
                sigma=kwargs.get("sigma", 0.75),
                k0=kwargs.get("k0", 2.0),
                V0=kwargs.get("V0", 3.0),
                x_barrier_left=kwargs.get("x_barrier_left", 0.5),
                x_barrier_right=kwargs.get("x_barrier_right", 1.5),
                mass=kwargs.get("mass", 1.0),
                hbar=kwargs.get("hbar", 1.0),
            )
            method = "Crank-Nicolson"

        elif sistema == "heat_inverse":
            # Para el calor usamos la solución analítica de Fourier
            # que ya tenemos en exact_solutions — no necesita método numérico
            # pero medimos su tiempo igualmente para la comparativa
            import torch

            from src.exact_solutions import heat_exact

            x_t = torch.tensor(x_or_t, dtype=torch.float32).unsqueeze(1)
            t_arr = kwargs.get("t_array", np.linspace(0, 1, 100))
            alpha = kwargs.get("alpha", 0.1)
            L = kwargs.get("L", 1.0)
            # Evaluamos en todos los instantes temporales
            results = []
            for t_val in t_arr:
                t_t = torch.full_like(x_t, t_val)
                results.append(heat_exact(x_t, t_t, alpha=alpha, L=L).numpy())
            solution = np.array(results)
            method = "Serie de Fourier (Analítica)"

        else:
            raise ValueError(
                f"Sistema '{sistema}' no reconocido. "
                f"Opciones: 'qho', 'pozo_infinito', 'oscilador_clasico', "
                f"'pendulo_inverso', 'tunnel', 'heat_inverse'"
            )

    return {
        "solution": solution,
        "time_s": t.elapsed,
        "method": method,
    }


def get_device() -> torch.device:
    """Detecta automáticamente si hay GPU disponible."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device
