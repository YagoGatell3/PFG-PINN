import os

import numpy as np
import torch
import torch.optim as optim

from src.exact_solutions import classical_oscillator
from src.loss_functions import (
    initial_condition_loss,
    physics_loss_classical_oscillator,
)
from src.models import PINNDynamic
from src.samplers import (
    generate_grid_points,
    generate_lhs_points,
)
from src.utils import (
    Timer,
    calculate_l2_error,
    measure_numerical_reference,
    plot_and_save_results,
    plot_comparison,
    save_experiment_results,
    set_seed,
    update_dynamic_weights,
)


def _train_model(
    use_physics: bool,
    t_zero: torch.Tensor,
    t_train: torch.Tensor,
    u_train: torch.Tensor,
    t_domain: torch.Tensor,
    t_eval: torch.Tensor,
    u_true: torch.Tensor,
    mass: float,
    k: float,
    u_0: float,
    v_0: float,
    epochs: int,
    lr: float,
    use_data: bool,
    use_dynamic_weights: bool,
    optimizer_name: str,
    hidden_layers: list,
    log_freq: int,
    seed: int,
    save_plots: bool = True,
) -> tuple[float, float, dict, torch.Tensor]:
    """
    Rutina principal de entrenamiento para los modelos de dinámica temporal (PINN o NN estándar).
    Recibe los tensores pregenerados para garantizar condiciones idénticas de evaluación 
    y convergencia entre los distintos métodos.

    Args:
        use_physics (bool): Indica si se debe optimizar el residuo de la ecuación de movimiento.
        t_zero (torch.Tensor): Tensor representando el instante temporal inicial (t=0).
        t_train (torch.Tensor): Puntos temporales de entrenamiento empírico (observaciones).
        u_train (torch.Tensor): Solución exacta evaluada en los puntos de entrenamiento.
        t_domain (torch.Tensor): Puntos de colocación para evaluar el residuo físico en el dominio continuo.
        t_eval (torch.Tensor): Puntos temporales de evaluación continua para métricas de validación.
        u_true (torch.Tensor): Solución analítica en los puntos de validación.
        mass (float): Masa del oscilador clásico.
        k (float): Constante elástica del oscilador.
        u_0 (float): Posición inicial del sistema.
        v_0 (float): Velocidad inicial del sistema.
        epochs (int): Número total de épocas de optimización.
        lr (float): Tasa de aprendizaje inicial.
        use_data (bool): Indica si se incluyen datos empíricos en la función de pérdida.
        use_dynamic_weights (bool): Habilita la actualización dinámica de los ponderadores de pérdida.
        optimizer_name (str): Optimizador seleccionado ('adam', 'lbfgs', o 'adam+lbfgs').
        hidden_layers (list): Arquitectura topológica del perceptrón multicapa.
        log_freq (int): Frecuencia de épocas para la impresión del progreso por consola.
        seed (int): Semilla para garantizar la reproducibilidad.
        save_plots (bool, opcional): Habilita el guardado de visualizaciones durante el entrenamiento. Por defecto es True.

    Returns:
        tuple[float, float, dict, torch.Tensor]: Tupla que empaqueta:
            - Error relativo L2 final.
            - Tiempo total de cómputo en segundos.
            - Diccionario con el historial completo de métricas y pérdidas.
            - Tensor con las inferencias generadas tras el entrenamiento.
    """
    set_seed(seed)
    model = PINNDynamic(hidden_layers=hidden_layers)

    # --- Configuración del motor de optimización ---
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=250)
    elif optimizer_name == "adam+lbfgs":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no reconocido. "
            f"Opciones válidas: 'adam', 'lbfgs', 'adam+lbfgs'."
        )

    # Inicialización de ponderadores dinámicos
    lambda_ph = 1.0
    lambda_ic = 1.0

    historial = {
        "epoch":      [],
        "total_loss": [],
        "data_loss":  [],
        "ph_loss":    [],
        "ic_loss":    [],
        "lambda_ph":  [],
        "lambda_ic":  [],
    }

    label = "PINN" if use_physics else "NN"

    with Timer() as timer:
        for epoch in range(1, epochs + 1):

            # Transición heurística de Adam a L-BFGS en el ecuador del proceso de entrenamiento
            if optimizer_name == "adam+lbfgs" and epoch == epochs // 2 + 1:
                print(f"\n[{label}] Transición al optimizador L-BFGS en época {epoch}\n")
                optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=250)

            def closure(lph=lambda_ph, lic=lambda_ic):
                optimizer.zero_grad()

                # Evaluación de la pérdida empírica (datos observables)
                if use_data:
                    data_loss = torch.mean((model(t_train) - u_train) ** 2)
                else:
                    data_loss = torch.tensor(0.0)

                # Evaluación de la regularización basada en la física y condiciones iniciales
                if use_physics:
                    ph_loss_val = physics_loss_classical_oscillator(
                        model, t_domain, mass=mass, k=k
                    )
                    ic_loss_val = initial_condition_loss(
                        model, t_zero, u_0=u_0, v_0=v_0
                    )
                else:
                    ph_loss_val = torch.tensor(0.0)
                    ic_loss_val = torch.tensor(0.0)

                # Ensamblaje de la función de coste global
                if not use_physics:
                    total = data_loss
                elif use_data:
                    total = data_loss + lph * ph_loss_val + lic * ic_loss_val
                else:
                    total = lph * ph_loss_val + 10.0 * ic_loss_val

                total.backward()
                return total

            # --- Ejecución del paso de optimización ---
            if optimizer_name == "lbfgs" or (optimizer_name == "adam+lbfgs" and epoch >= epochs // 2 + 1):
                result = optimizer.step(closure)
                total_loss = result if result is not None else torch.tensor(0.0)
                
                # Estimación de métricas para el historial (sin afectar el grafo computacional)
                with torch.no_grad():
                    data_loss   = torch.mean((model(t_train) - u_train) ** 2) if use_data else torch.tensor(0.0)
                ph_loss_val = torch.tensor(0.0)
                ic_loss_val = torch.tensor(0.0)
            else:
                optimizer.zero_grad()

                if use_data:
                    data_loss = torch.mean((model(t_train) - u_train) ** 2)
                else:
                    data_loss = torch.tensor(0.0)

                if use_physics:
                    ph_loss_val = physics_loss_classical_oscillator(
                        model, t_domain, mass=mass, k=k
                    )
                    ic_loss_val = initial_condition_loss(
                        model, t_zero, u_0=u_0, v_0=v_0
                    )
                else:
                    ph_loss_val = torch.tensor(0.0)
                    ic_loss_val = torch.tensor(0.0)

                # Recalibración dinámica de hiperparámetros de pérdida
                if use_dynamic_weights and use_data and use_physics:
                    lambda_ph, lambda_ic = update_dynamic_weights(
                        data_loss,
                        ph_loss_val,
                        ic_loss_val,
                        model.net[-1].weight,
                        lambda_ph,
                        lambda_ic,
                    )
                else:
                    lambda_ph, lambda_ic = 1.0, 1.0

                if not use_physics:
                    total_loss = data_loss
                elif use_data:
                    total_loss = (
                        data_loss
                        + lambda_ph * ph_loss_val
                        + lambda_ic * ic_loss_val
                    )
                else:
                    total_loss = lambda_ph * ph_loss_val + 10.0 * ic_loss_val

                total_loss.backward()
                optimizer.step()

            # --- Monitorización y almacenamiento del estado ---
            if epoch % log_freq == 0 or epoch == epochs:
                print(f"[{label}] Época {epoch:05d} | Pérdida: {total_loss.item():.4e}")
                if use_dynamic_weights and use_data and use_physics:
                    print(
                        f"            | Pesos dinámicos -> Física: {lambda_ph:.4f} "
                        f"| Cond. Inicial: {lambda_ic:.4f}"
                    )

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())
                historial["ph_loss"].append(ph_loss_val.item())
                historial["ic_loss"].append(ic_loss_val.item())
                historial["lambda_ph"].append(lambda_ph)
                historial["lambda_ic"].append(lambda_ic)

                if save_plots:
                    plot_and_save_results(
                        model,
                        t_train if use_data else None,
                        u_train if use_data else None,
                        t_eval,
                        u_true,
                        epoch,
                        total_loss.item(),
                        n=0,
                        save_dir="img",
                        sistema="oscilador_clasico",
                        label=label,
                    )

    pred_eval = model(t_eval).detach()
    error_l2  = calculate_l2_error(pred_eval, u_true)

    return error_l2, timer.elapsed, historial, pred_eval


def main(
    t_max: float = 10.0,
    mass: float = 1.0,
    k: float = 2.0,
    u_0: float = 1.0,
    v_0: float = 0.0,
    epochs: int = 5000,
    lr: float = 0.001,
    num_domain_points: int = 500,
    num_train_points: int = 15,
    train_region: float = 0.2,
    sampler: str = "lhs",
    log_freq: int = 1000,
    use_data: bool = True,
    use_dynamic_weights: bool = False,
    optimizer_name: str = "adam",
    hidden_layers: list = None,
    seed: int = 42,
    save_plots: bool = True,
):
    """
    Orquesta la ejecución integral de un experimento para la dinámica del Oscilador Armónico Clásico.
    Sintetiza la generación de datos, el entrenamiento comparativo (PINN vs NN estándar), la contrastación 
    numérica contra métodos explícitos (RK4) y el empaquetado final de los resultados.

    Args:
        t_max (float, opcional): Límite superior del dominio temporal evaluado. Por defecto es 10.0.
        mass (float, opcional): Masa del sistema. Por defecto es 1.0.
        k (float, opcional): Constante elástica. Por defecto es 2.0.
        u_0 (float, opcional): Posición inicial del sistema en t=0. Por defecto es 1.0.
        v_0 (float, opcional): Velocidad inicial del sistema en t=0. Por defecto es 0.0.
        epochs (int, opcional): Número total de iteraciones de optimización. Por defecto es 5000.
        lr (float, opcional): Tasa de aprendizaje inicial. Por defecto es 0.001.
        num_domain_points (int, opcional): Número de nodos de colocación en el continuo. Por defecto es 500.
        num_train_points (int, opcional): Tamaño de la muestra de datos observacionales. Por defecto es 15.
        train_region (float, opcional): Fracción inicial del dominio cubierta por observaciones empíricas. Por defecto es 0.2.
        sampler (str, opcional): Algoritmo de discretización de colocación ('lhs' o 'grid'). Por defecto es "lhs".
        log_freq (int, opcional): Frecuencia de volcado a consola del progreso del entrenamiento. Por defecto es 1000.
        use_data (bool, opcional): Integra la función de coste basada en datos empíricos. Por defecto es True.
        use_dynamic_weights (bool, opcional): Habilita ponderación adaptativa de los términos de pérdida. Por defecto es False.
        optimizer_name (str, opcional): Motor de búsqueda del gradiente. Por defecto es "adam".
        hidden_layers (list, opcional): Arquitectura de las capas ocultas de la red neuronal. Por defecto es [32, 32, 32].
        seed (int, opcional): Semilla algorítmica para asegurar reproducibilidad total. Por defecto es 42.
        save_plots (bool, opcional): Bandera de control de exportación gráfica de resultados. Por defecto es True.
    """
    if hidden_layers is None:
        hidden_layers = [32, 32, 32]

    # --- 1. Configuración de reproducibilidad y sistema de archivos ---
    set_seed(seed)
    os.makedirs("img", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    config_exp = {
        "sistema":             "oscilador_clasico",
        "estado_n":            0,
        "t_max":               t_max,
        "mass":                mass,
        "k":                   k,
        "epochs":              epochs,
        "lr":                  lr,
        "num_domain_points":   num_domain_points,
        "num_train_points":    num_train_points,
        "train_region":        train_region,
        "sampler":             sampler,
        "use_data":            use_data,
        "use_dynamic_weights": use_dynamic_weights,
        "optimizer":           optimizer_name,
        "hidden_layers":       hidden_layers,
        "seed":                seed,
    }

    print("=" * 60)
    print("Oscilador Clásico — Experimentación y Benchmark")
    print(
        f"Muestreo: {sampler} | Colocación: {num_domain_points} nodos | "
        f"Train: {num_train_points} obs | Cobertura: {int(train_region*100)}% | "
        f"Épocas: {epochs} | Motor Opt.: {optimizer_name} | "
        f"lr: {lr} | Pesos adaptativos: {use_dynamic_weights} | Semilla: {seed}"
    )
    print("=" * 60)

    # --- 2. Generación unificada de tensores base compartidos ---
    t_zero  = torch.tensor([[0.0]], requires_grad=True)
    t_train = generate_grid_points(0.0, t_max * train_region, num_train_points, requires_grad=False)
    u_train = classical_oscillator(t_train, mass=mass, k=k, u_0=u_0, v_0=v_0)
    t_eval  = generate_grid_points(0.0, t_max, 500, requires_grad=False)
    u_true  = classical_oscillator(t_eval, mass=mass, k=k, u_0=u_0, v_0=v_0)

    if sampler == "lhs":
        t_domain = generate_lhs_points(0.0, t_max, num_domain_points)
    else:
        t_domain = generate_grid_points(0.0, t_max, num_domain_points)

    shared = dict(
        t_zero=t_zero, t_train=t_train, u_train=u_train,
        t_domain=t_domain, t_eval=t_eval, u_true=u_true,
        mass=mass, k=k, u_0=u_0, v_0=v_0,
        epochs=epochs, lr=lr, use_data=use_data,
        use_dynamic_weights=use_dynamic_weights,
        optimizer_name=optimizer_name, hidden_layers=hidden_layers,
        log_freq=log_freq, seed=seed, save_plots=save_plots,
    )

    # --- 3. Despliegue del Modelo Informado por la Física (PINN) ---
    print("\n--- Modelado PINN (Regulación Física Activa) ---")
    error_pinn, time_pinn, hist_pinn, pred_pinn = _train_model(use_physics=True, **shared)

    # --- 4. Despliegue del Modelo Empírico (NN Estándar) ---
    print("\n--- Modelado NN Pura (Basado exclusivamente en Datos) ---")
    error_nn, time_nn, hist_nn, pred_nn = _train_model(use_physics=False, **shared)

    # --- 5. Resolución de Referencia Numérica (Método de Runge-Kutta de 4º Orden) ---
    print("\n--- Resolución Discreta Estándar (Método RK4) ---")
    t_np = np.linspace(0.0, t_max, 500)
    ref  = measure_numerical_reference(
        sistema="oscilador_clasico",
        x_or_t=t_np,
        mass=mass, k=k, u0=u_0, v0=v_0,
    )
    u_rk4     = torch.tensor(ref["solution"][0], dtype=torch.float32).unsqueeze(1)
    error_rk4 = calculate_l2_error(u_rk4, u_true)
    time_rk4  = ref["time_s"]

    # --- 6. Exportación de Perfil Comparativo Final ---
    if save_plots:
        plot_comparison(
            x_eval=t_eval,
            u_true=u_true,
            pred_pinn=pred_pinn,
            pred_nn=pred_nn,
            pred_numerical=u_rk4,
            error_pinn=error_pinn,
            error_nn=error_nn,
            error_numerical=error_rk4,
            numerical_label="RK4",
            sistema="oscilador_clasico",
            train_region_end=t_max * train_region,
            save_dir="img",
            estado_n=0,
        )

    # --- 7. Consolidación Estructural de Resultados ---
    final_results = {
        "pinn": {"error_L2": error_pinn, "time_s": time_pinn},
        "nn":   {"error_L2": error_nn,   "time_s": time_nn},
        "rk4":  {"error_L2": error_rk4,  "time_s": time_rk4, "method": ref["method"]},
    }

    historial_completo = {
        "pinn": hist_pinn,
        "nn":   hist_nn,
    }

    print(f"\n{'=' * 60}")
    print("RESUMEN DE PRECISIÓN METODOLÓGICA")
    print(f"{'Metodología':<12} {'Error L2':>12} {'Cómputo (s)':>12}")
    print(f"{'RK4 Explícito':<12} {error_rk4:>12.4e} {time_rk4:>12.4f}")
    print(f"{'NN Estándar':<12} {error_nn:>12.4e} {time_nn:>12.2f}")
    print(f"{'PINN Analítica':<12} {error_pinn:>12.4e} {time_pinn:>12.2f}")
    print("=" * 60)

    save_experiment_results(config_exp, final_results, historial_completo)
    
    
if __name__ == "__main__":
 
    SEEDS = [42, 123, 7, 99, 2024, 314, 17, 56, 88, 200]
 
    # ----------------------------------------------------------------
    # Configuración Paramétrica BASE
    # ----------------------------------------------------------------
    BASE = dict(
        epochs=10000,
        lr=0.001,
        num_domain_points=500,
        num_train_points=15,
        train_region=0.2,
        sampler="lhs",
        use_data=True,
        use_dynamic_weights=False,
        optimizer_name="adam",
        hidden_layers=[32, 32, 32],
        log_freq=10000,
        save_plots=True,   # Control de exportación gráfica global
    )
 
    # ----------------------------------------------------------------
    # Hiperespacio del Estudio de Sensibilidad
    # Cada diccionario superpone sus valores sobre la configuración BASE
    # ----------------------------------------------------------------
    variaciones = [
        # --- Nodo Control (BASE) ---
        {},
 
        # --- Eje 1: Densidad de nodos de colocación espacial ---
        {"num_domain_points": 50},
        {"num_domain_points": 100},
        {"num_domain_points": 250},
        {"num_domain_points": 1000},
 
        # --- Eje 2: Estrategias de muestreo del dominio ---
        {"sampler": "grid"},
 
        # --- Eje 3: Algoritmos de optimización de segundo orden ---
        {"optimizer_name": "adam+lbfgs"},
 
        # --- Eje 4: Capacidad estructural de la red neuronal ---
        {"hidden_layers": [64, 64, 64]},
 
        # --- Eje 5: Mecanismos de regularización dinámica ---
        {"use_dynamic_weights": True},
 
        # --- Eje 6: Tasa de aprendizaje (Learning rate) ---
        {"lr": 0.01},
 
        # --- Eje 7: Escasez de datos empíricos (Control de región: 20%) ---
        {"num_train_points": 3},
        {"num_train_points": 5},
        {"num_train_points": 10},
        {"num_train_points": 30},
 
        # --- Eje 8: Extensión observacional (Expansión al 50% del dominio continuo) ---
        {"train_region": 0.5},
 
        # --- Eje 9: Intersección (Escasez vs Extensión observacional) ---
        {"num_train_points": 3,  "train_region": 0.5},
        {"num_train_points": 5,  "train_region": 0.5},
        {"num_train_points": 10, "train_region": 0.5},
        {"num_train_points": 30, "train_region": 0.5},
    ]
 
    total_configs = len(variaciones)
    total_runs    = total_configs * len(SEEDS)
 
    print(f"\n{'=' * 60}")
    print(f"ANÁLISIS DE SENSIBILIDAD METODOLÓGICA — Oscilador Clásico")
    print(f"Batería experimental: {total_configs} Configuraciones | {len(SEEDS)} Semillas | {total_runs} Ejecuciones totales")
    print(f"{'=' * 60}\n")
 
    for i, variacion in enumerate(variaciones, start=1):
        if i < 8:  # Omisión de configuraciones previamente procesadas
            continue
        config = {**BASE, **variacion}
        print(f"\n{'#' * 60}")
        print(f"Evaluación de Configuración {i}/{total_configs}: {variacion if variacion else 'Modelo BASE'}")
        print(f"{'#' * 60}")

        for j, seed in enumerate(SEEDS, start=1):
            print(f"\n  Instancia Estocástica (Seed) {j}/{len(SEEDS)}: {seed}")
            main(**config, seed=seed)