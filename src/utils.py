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
    Calcula y actualiza los pesos dinámicos de las funciones de pérdida físicas 
    y de frontera utilizando la retropropagación de sus gradientes.

    Args:
        data_loss (torch.Tensor): Tensor correspondiente a la pérdida de datos empíricos.
        ph_loss (torch.Tensor): Tensor correspondiente a la pérdida física (residuo).
        bound_loss (torch.Tensor): Tensor correspondiente a la pérdida de condiciones de frontera.
        last_layer_weight (torch.nn.Parameter): Pesos de la última capa de la red neuronal.
        current_lambda_ph (float): Peso actual asociado a la pérdida física.
        current_lambda_bound (float): Peso actual asociado a la pérdida de frontera.
        alpha (float, opcional): Factor de suavizado (momentum) para la actualización. Por defecto es 0.9.

    Returns:
        tuple[float, float]: Nuevos valores actualizados para lambda_ph y lambda_bound.
    """
    # Cálculo de los gradientes de cada pérdida respecto a los pesos de la última capa
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

    # Obtención de la magnitud máxima del gradiente de la pérdida principal (datos)
    max_grad_data = torch.max(torch.abs(grad_data))

    # Actualización de lambda_ph basada en la relación de gradientes
    new_lambda_ph = current_lambda_ph
    if grad_ph is not None:
        mean_grad_ph = torch.mean(torch.abs(grad_ph))
        hat_lambda_ph = max_grad_data / (mean_grad_ph + 1e-8)
        new_lambda_ph = (1 - alpha) * current_lambda_ph + alpha * hat_lambda_ph.item()

    # Actualización de lambda_bound basada en la relación de gradientes
    new_lambda_bound = current_lambda_bound
    if grad_bound is not None:
        mean_grad_bound = torch.mean(torch.abs(grad_bound))
        hat_lambda_bound = max_grad_data / (mean_grad_bound + 1e-8)
        new_lambda_bound = (
            1 - alpha
        ) * current_lambda_bound + alpha * hat_lambda_bound.item()

    return new_lambda_ph, new_lambda_bound


def calculate_l2_error(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    """
    Calcula la norma del error relativo L2 entre la predicción del modelo y la solución analítica.

    Args:
        u_pred (torch.Tensor): Predicciones generadas por la red neuronal.
        u_true (torch.Tensor): Valores exactos de la solución analítica.

    Returns:
        float: Valor numérico del error relativo L2.
    """
    error = torch.linalg.norm(u_pred - u_true) / torch.linalg.norm(u_true)
    return error.item()


def save_experiment_results(
    config: dict, final_results: dict, history: dict, save_dir: str = "results"
):
    """
    Serializa y almacena los hiperparámetros, resultados finales y el historial 
    de entrenamiento en un archivo JSON para garantizar la trazabilidad del experimento.

    Args:
        config (dict): Diccionario de configuración y metadatos del experimento.
        final_results (dict): Métricas y parámetros finales obtenidos.
        history (dict): Historial iterativo de las funciones de pérdida a lo largo del entrenamiento.
        save_dir (str, opcional): Directorio base de almacenamiento. Por defecto es "results".
    """
    sistema = config.get("sistema", "qho")
    estado_n = config.get("estado_n", 0)

    # Construcción de la ruta estructurada
    ruta_directorio = os.path.join(save_dir, sistema, f"estado_{estado_n}")
    os.makedirs(ruta_directorio, exist_ok=True)

    experimento = {
        "config": config,
        "resultados_finales": final_results,
        "historial": history,
    }

    # Generación de nomenclatura basada en el tipo de muestreo y marca temporal
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"exp_{config['sampler']}_{timestamp}.json"
    ruta_completa = os.path.join(ruta_directorio, nombre_archivo)

    with open(ruta_completa, "w") as f:
        json.dump(experimento, f, indent=4)

    print(f"Resultados guardados exitosamente en: {ruta_completa}")


def set_seed(seed: int = 42):
    """
    Fija la semilla en los motores de generación de números aleatorios para 
    garantizar la reproducibilidad estricta de los experimentos.

    Args:
        seed (int, opcional): Valor numérico de la semilla. Por defecto es 42.
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
    label: str = "PINN",
):
    """
    Genera y almacena una representación visual que contrasta la predicción actual 
    del modelo (PINN) frente a la solución analítica y los datos de entrenamiento.

    Args:
        pinn_model (torch.nn.Module): Modelo de red neuronal entrenado.
        x_train (torch.Tensor): Coordenadas de los datos de entrenamiento.
        u_train (torch.Tensor): Valores objetivo en los puntos de entrenamiento.
        x_eval (torch.Tensor): Coordenadas de evaluación continua.
        u_true (torch.Tensor): Valores analíticos exactos evaluados en x_eval.
        epoch (int): Número de época actual (para nomenclatura e información).
        pinn_loss (float): Valor de la pérdida total en la época actual.
        n (int, opcional): Estado cuántico (aplicable en sistemas cuánticos). Por defecto es 0.
        save_dir (str, opcional): Directorio base de exportación. Por defecto es "../img".
        sistema (str, opcional): Identificador del sistema físico evaluado. Por defecto es "qho".
        label (str, opcional): Etiqueta descriptiva del modelo. Por defecto es "PINN".
    """
    # Verificación y creación de la estructura de directorios
    ruta_directorio = os.path.join(save_dir, sistema, f"estado_{n}")
    os.makedirs(ruta_directorio, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Aislamiento del grafo computacional para evitar fugas de memoria en inferencia
    pinn_pred = pinn_model(x_eval).detach()
    x_eval_np = x_eval.detach().numpy()
    u_true_np = u_true.detach().numpy()
    pred_np = pinn_pred.numpy()

    # Trazado de la curva analítica de referencia
    ax.plot(
        x_eval_np,
        u_true_np,
        label="Solución Analítica",
        color="blue",
        linewidth=2,
        alpha=0.5,
    )

    # Trazado de la curva generada por inferencia de la PINN
    ax.plot(
        x_eval_np,
        pred_np,
        label="Predicción PINN",
        linestyle="--",
        color="black",
        linewidth=2,
    )

    # Proyección de los puntos de datos empíricos utilizados
    if x_train is not None and u_train is not None:
        ax.scatter(
            x_train.detach().numpy(),
            u_train.detach().numpy(),
            color="red",
            label="Datos de Entrenamiento",
            s=50,
            zorder=5,
        )

    # Parametrización semántica de los ejes en función del problema modelado
    if sistema == "qho":
        titulo = f"PINN (Oscilador Armónico Cuántico) - Estado n={n} | Época {epoch}"
        ax.set_xlabel("x (Posición)")
        ax.set_ylabel("ψ(x) (Función de onda)")
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

    # Inserción de metadatos de rendimiento en el gráfico
    ax.set_title(f"{titulo}\nPérdida Total: {pinn_loss:.4e}")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    # Exportación gráfica
    nombre_archivo = f"{label}_epoch_{epoch:05d}.png"
    ruta_completa = os.path.join(ruta_directorio, nombre_archivo)
    plt.savefig(ruta_completa, dpi=300)
    plt.close()


class Timer:
    """
    Gestor de contexto modular para la perfilación de tiempos de ejecución (benchmarking).

    Ejemplo de uso:
        with Timer() as t:
            operacion_pesada()
        print(f"Tiempo transcurrido: {t.elapsed} segundos")
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
    Ejecuta y perfilea (mide el tiempo) el solver numérico de referencia 
    asociado a un sistema físico particular.

    Args:
        sistema (str): Identificador del sistema (ej. 'qho', 'pozo_infinito', etc.).
        x_or_t (np.ndarray): Matriz de discretización del dominio espacial o temporal.
        **kwargs: Diccionario con la parametrización física específica del sistema seleccionado.

    Returns:
        dict: Diccionario que empaqueta:
            - 'solution' (np.ndarray): Los resultados de la simulación numérica.
            - 'time_s' (float): Tiempo de cómputo invertido en segundos.
            - 'method' (str): Metodología algorítmica utilizada.

    Raises:
        ValueError: Si el identificador del sistema no se encuentra registrado.
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
            V = np.zeros_like(x_or_t)  # Potencial estrictamente nulo dentro del pozo
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
            from src.numerical_methods import solve_heat_crank_nicolson

            t_arr = kwargs.get("t_array", np.linspace(0, 1, 100))
            alpha = kwargs.get("alpha", 0.1)
            L     = kwargs.get("L", 1.0)
            solution = solve_heat_crank_nicolson(
                x=x_or_t, t=t_arr, alpha=alpha, L=L,
            )
            method = "Crank-Nicolson"

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
    """
    Gestiona la selección del dispositivo computacional (hardware) sobre el que 
    correrán los tensores. (Actualmente hardcodeado a CPU).

    Returns:
        torch.device: Instancia de dispositivo PyTorch seleccionada.
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Usando dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def plot_comparison(
    x_eval: torch.Tensor,
    u_true: torch.Tensor,
    pred_pinn: torch.Tensor,
    pred_nn: torch.Tensor,
    pred_numerical: torch.Tensor,
    error_pinn: float,
    error_nn: float,
    error_numerical: float,
    numerical_label: str,
    sistema: str,
    x_train: torch.Tensor = None,
    train_region_end: float = None,
    save_dir: str = "img",
    estado_n: int = 0,
):
    """
    Exporta un gráfico comparativo multidimensional evaluando el rendimiento 
    final de los distintos modelos aproximadores y numéricos.

    Args:
        x_eval (torch.Tensor): Puntos del dominio continuo de evaluación (x o t).
        u_true (torch.Tensor): Solución analítica base del problema.
        pred_pinn (torch.Tensor): Perfil inferido por la red informada por la física (PINN).
        pred_nn (torch.Tensor): Perfil inferido por el modelo puramente empírico (NN Base).
        pred_numerical (torch.Tensor): Solución discreta generada por el método numérico tradicional.
        error_pinn (float): Evaluación del error relativo L2 asociado a la PINN.
        error_nn (float): Evaluación del error relativo L2 asociado a la NN Base.
        error_numerical (float): Evaluación del error relativo L2 asociado a la simulación discreta.
        numerical_label (str): Etiqueta algorítmica utilizada (ej. 'RK4', 'Crank-Nicolson').
        sistema (str): Identificador clave del sistema modelado.
        x_train (torch.Tensor, opcional): Puntos espaciales extraídos para entrenamiento empírico.
        train_region_end (float, opcional): Límite derecho del dominio de entrenamiento.
        save_dir (str, opcional): Ruta hacia la carpeta de exportación gráfica. Por defecto es "img".
        estado_n (int, opcional): Estado cuántico (reservar en 0 para modelos macroscópicos). Por defecto es 0.
    """
    ruta = os.path.join(save_dir, sistema, f"estado_{estado_n}")
    os.makedirs(ruta, exist_ok=True)

    # Conversión vectorial a formato unidimensional Numpy
    x_np      = x_eval.detach().numpy().flatten()
    u_true_np = u_true.detach().numpy().flatten()
    pinn_np   = pred_pinn.detach().numpy().flatten()
    nn_np     = pred_nn.detach().numpy().flatten()
    num_np    = pred_numerical.detach().numpy().flatten()

    # Mapeo descriptivo para ejes en función del problema
    etiquetas = {
        "oscilador_clasico": ("t (Tiempo)", "u(t) (Posición)"),
        "pendulo_inverso":   ("t (Tiempo)", "θ(t) (Ángulo)"),
        "qho":               ("x (Posición)", "ψ(x) (Función de onda)"),
        "pozo_infinito":     ("x (Posición)", "ψ(x) (Función de onda)"),
        "heat_inverse":      ("x (Posición)", "u(x,t)"),
    }
    xlabel, ylabel = etiquetas.get(sistema, ("Entrada", "Salida"))

    fig, ax = plt.subplots(figsize=(11, 5))

    # Trazado comparativo
    ax.plot(x_np, u_true_np, label="Solución analítica",
            color="blue", linewidth=2, alpha=0.6)
    ax.plot(x_np, num_np,  label=f"{numerical_label} (L2={error_numerical:.2e})",
            color="green",  linewidth=1.5, linestyle="--")
    ax.plot(x_np, nn_np,   label=f"NN pura (L2={error_nn:.2e})",
            color="orange", linewidth=1.5, linestyle="-.")
    ax.plot(x_np, pinn_np, label=f"PINN (L2={error_pinn:.2e})",
            color="red",    linewidth=1.5, linestyle=":")

    # Región sombreada ilustrando la cobertura de los datos de entrenamiento
    if train_region_end is not None:
        ax.axvspan(x_np[0], train_region_end, alpha=0.08,
                   color="gray", label="Zona de entrenamiento")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{sistema.replace('_', ' ').title()} — Comparativa: Analítica vs {numerical_label} vs NN vs PINN")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    ruta_completa = os.path.join(ruta, "comparativa_final.png")
    plt.savefig(ruta_completa, dpi=150)
    plt.close()
    print(f"Gráfica comparativa guardada en: {ruta_completa}")
    
    
def update_dynamic_weights_tunnel(
    ic_loss: torch.Tensor,
    ph_loss: torch.Tensor,
    bc_loss: torch.Tensor,
    norm_loss: torch.Tensor,
    data_loss: torch.Tensor,
    last_layer_weight: torch.nn.Parameter,
    current_lambda_ph: float,
    current_lambda_bc: float,
    current_lambda_norm: float,
    current_lambda_data: float,
    alpha: float = 0.9,
    lambda_min: float = 0.1,
    lambda_max: float = 50.0,
) -> tuple[float, float, float, float]:
    """
    Implementa un algoritmo de balanceo dinámico de gradientes específico para la 
    ecuación de Schrödinger dependiente del tiempo (efecto túnel).
    
    Utiliza el gradiente de la Condición Inicial (IC) como pivote base para reescalar 
    las penalizaciones del residuo físico, fronteras, datos y normalización, aplicando 
    recortes (clipping) para asegurar la estabilidad numérica.

    Args:
        ic_loss (torch.Tensor): Pérdida asociada a la condición inicial (paquete de ondas).
        ph_loss (torch.Tensor): Pérdida estructural proveniente de la PDE.
        bc_loss (torch.Tensor): Pérdida en las fronteras de Dirichlet.
        norm_loss (torch.Tensor): Pérdida que fuerza la probabilidad integradora unitaria.
        data_loss (torch.Tensor): Pérdida empírica basada en puntos colisionados explícitos.
        last_layer_weight (torch.nn.Parameter): Pesos en la capa de salida.
        current_lambda_... (float): Factores de penalización lambda correspondientes del epoch previo.
        alpha (float, opcional): Factor de momentum de promediado temporal. Por defecto es 0.9.
        lambda_min (float, opcional): Límite mínimo de saturación para los pesos. Por defecto es 0.1.
        lambda_max (float, opcional): Límite máximo de saturación para los pesos. Por defecto es 50.0.

    Returns:
        tuple[float, float, float, float]: Bloque actualizado de multiplicadores lambda 
        (ph, bc, norm, data).
    """
    # Extracción del gradiente de referencia (condición inicial)
    grad_ic = torch.autograd.grad(
        ic_loss, last_layer_weight, retain_graph=True, allow_unused=True
    )[0]

    if grad_ic is None:
        return current_lambda_ph, current_lambda_bc, current_lambda_norm, current_lambda_data

    max_grad_ic = torch.max(torch.abs(grad_ic))

    def _update(loss, current_lambda):
        """Subrutina interna para recalcular individualmente cada componente lambda."""
        grad = torch.autograd.grad(
            loss, last_layer_weight, retain_graph=True, allow_unused=True
        )[0]
        if grad is None:
            return current_lambda
        
        # Inclusión del epsilon term (1e-8) incrustado conceptualmente previniendo indeterminaciones
        mean_grad  = torch.mean(torch.abs(grad)) + 1e-8
        hat_lambda = max_grad_ic / mean_grad
        
        # Suavización temporal EMA y recorte (clipping) estricto de estabilidad
        new_lambda = (1 - alpha) * current_lambda + alpha * hat_lambda.item()
        return float(np.clip(new_lambda, lambda_min, lambda_max))

    # Actualizaciones funcionales modulares
    new_lambda_ph   = _update(ph_loss,   current_lambda_ph)
    new_lambda_bc   = _update(bc_loss,   current_lambda_bc)
    new_lambda_norm = _update(norm_loss, current_lambda_norm)
    
    # Manejo contingente si la pérdida de datos se anula (estrategia sin muestreo empírico)
    new_lambda_data = (
        _update(data_loss, current_lambda_data)
        if data_loss.item() > 0.0
        else current_lambda_data
    )

    return new_lambda_ph, new_lambda_bc, new_lambda_norm, new_lambda_data