import math
import os

import numpy as np
import torch
import torch.optim as optim
from scipy.interpolate import interp1d

from src.exact_solutions import psi_infinite_well
from src.loss_functions import (
    boundary_loss,
    normalization_loss,
    orthogonality_loss,
    physics_loss_infinite_well,
)
from src.models import PINNWell
from src.samplers import (
    generate_boundary_points,
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
    estado_n: int,
    L: float,
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    x_train: torch.Tensor,
    u_train: torch.Tensor,
    x_domain: torch.Tensor,
    x_eval: torch.Tensor,
    u_true: torch.Tensor,
    epochs: int,
    lr: float,
    use_data: bool,
    use_dynamic_weights: bool,
    use_orthogonality: bool,
    optimizer_name: str,
    hidden_layers: list,
    log_freq: int,
    seed: int,
    save_plots: bool = True,
) -> tuple[float, float, dict, torch.Tensor]:
    """
    Rutina principal de entrenamiento para los modelos neuronales (PINN o NN estándar).
    Recibe los tensores pregenerados para garantizar condiciones idénticas de 
    evaluación y convergencia entre los distintos métodos.

    Args:
        use_physics (bool): Indica si se debe optimizar el residuo de la PDE (física).
        estado_n (int): Nivel de energía cuántico a simular.
        L (float): Longitud del dominio espacial del pozo.
        x_left (torch.Tensor): Coordenada de la frontera izquierda.
        x_right (torch.Tensor): Coordenada de la frontera derecha.
        x_train (torch.Tensor): Puntos de entrenamiento empírico (observaciones).
        u_train (torch.Tensor): Solución exacta evaluada en los puntos de entrenamiento.
        x_domain (torch.Tensor): Puntos de colocación para evaluar el residuo físico.
        x_eval (torch.Tensor): Puntos de evaluación continua para métricas de validación.
        u_true (torch.Tensor): Solución analítica en los puntos de validación.
        epochs (int): Número total de épocas de optimización.
        lr (float): Tasa de aprendizaje inicial.
        use_data (bool): Indica si se incluyen datos empíricos en la función de pérdida.
        use_dynamic_weights (bool): Habilita la actualización dinámica de los ponderadores de pérdida.
        use_orthogonality (bool): Fuerza el aprendizaje ortogonal respecto a estados previos.
        optimizer_name (str): Optimizador seleccionado ('adam', 'lbfgs', o 'adam+lbfgs').
        hidden_layers (list): Arquitectura del perceptrón multicapa.
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
    
    # Estimación inicial del autovalor basada en la formulación analítica
    epsilon_exacto = float(estado_n ** 2 * math.pi ** 2 / (2.0 * L ** 2))
    epsilon_init = epsilon_exacto - 0.05 * epsilon_exacto
    
    model = PINNWell(hidden_layers=hidden_layers, epsilon_init=epsilon_init)
    label = "PINN" if use_physics else "NN"

    historial = {
        "epoch":        [],
        "total_loss":   [],
        "data_loss":    [],
        "ph_loss":      [],
        "bound_loss":   [],
        "ortho_loss":   [],
        "epsilon":      [],
        "norm_loss":    [],
        "lambda_ph":    [],
        "lambda_bound": [],
    }

    # --- Caso degenerado: Modelo empírico sin provisión de datos ---
    if not use_physics and not use_data:
        print(f"[{label}] Omitida: Modelo sin formulación física ni datos empíricos disponibles.")
        pred_eval = model(x_eval).detach()
        error_l2  = calculate_l2_error(pred_eval, u_true)
        return error_l2, 0.0, historial, pred_eval

    # --- Configuración del motor de optimización ---
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=500)
    elif optimizer_name == "adam+lbfgs":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no reconocido. "
            f"Opciones válidas: 'adam', 'lbfgs', 'adam+lbfgs'."
        )

    # Inicialización de ponderadores dinámicos
    lambda_ph    = 1.0
    lambda_bound = 1.0

    with Timer() as timer:
        for epoch in range(1, epochs + 1):

            # Transición heurística de Adam a L-BFGS en el ecuador del proceso
            if optimizer_name == "adam+lbfgs" and epoch == epochs // 2 + 1:
                print(f"\n[{label}] Transición al optimizador L-BFGS en época {epoch}\n")
                optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=500)

            def closure(lph=lambda_ph, lbound=lambda_bound):
                optimizer.zero_grad()

                # Evaluación de la pérdida de datos empíricos
                if use_data:
                    data_loss = torch.mean((model(x_train) - u_train) ** 2)
                else:
                    data_loss = torch.tensor(0.0)

                ph_loss_val    = torch.tensor(0.0)
                bound_loss_val = torch.tensor(0.0)
                ortho_loss_val = torch.tensor(0.0)
                norm_loss_val  = torch.tensor(0.0)

                # Evaluación de los componentes de pérdida informados por la física
                if use_physics:
                    ph_loss_val    = physics_loss_infinite_well(model, x_domain)
                    bound_loss_val = boundary_loss(model, x_left, x_right)
                    norm_loss_val  = normalization_loss(model, x_domain, domain_length=L)

                    if use_orthogonality and estado_n > 1:
                        psi_prev = psi_infinite_well(x_domain, n=estado_n - 1, L=L).detach()
                        ortho_loss_val = orthogonality_loss(
                            model, x_domain, psi_prev, domain_length=L
                        )

                # Ensamblaje de la función de coste global
                if not use_physics:
                    total = data_loss
                elif use_data:
                    total = (
                        data_loss
                        + lph * ph_loss_val
                        + lbound * bound_loss_val
                        + norm_loss_val
                        + ortho_loss_val
                    )
                else:
                    total = (
                        lph * ph_loss_val
                        + lbound * bound_loss_val
                        + 10.0 * norm_loss_val
                        + 10.0 * ortho_loss_val
                    )

                total.backward()
                return total

            # --- Ejecución del paso de optimización ---
            if optimizer_name == "lbfgs" or (
                optimizer_name == "adam+lbfgs" and epoch >= epochs // 2 + 1
            ):
                result     = optimizer.step(closure)
                total_loss = result if result is not None else torch.tensor(0.0)
                
                with torch.no_grad():
                    data_loss = (
                        torch.mean((model(x_train) - u_train) ** 2)
                        if use_data
                        else torch.tensor(0.0)
                    )
                ph_loss_val    = torch.tensor(0.0)
                bound_loss_val = torch.tensor(0.0)
                ortho_loss_val = torch.tensor(0.0)
                norm_loss_val  = torch.tensor(0.0)
            else:
                optimizer.zero_grad()

                if use_data:
                    data_loss = torch.mean((model(x_train) - u_train) ** 2)
                else:
                    data_loss = torch.tensor(0.0)

                ph_loss_val    = torch.tensor(0.0)
                bound_loss_val = torch.tensor(0.0)
                ortho_loss_val = torch.tensor(0.0)
                norm_loss_val  = torch.tensor(0.0)

                if use_physics:
                    ph_loss_val    = physics_loss_infinite_well(model, x_domain)
                    bound_loss_val = boundary_loss(model, x_left, x_right)
                    norm_loss_val  = normalization_loss(model, x_domain, domain_length=L)

                    if use_orthogonality and estado_n > 1:
                        psi_prev = psi_infinite_well(x_domain, n=estado_n - 1, L=L).detach()
                        ortho_loss_val = orthogonality_loss(
                            model, x_domain, psi_prev, domain_length=L
                        )

                # Recalibración dinámica de hiperparámetros de pérdida
                if use_dynamic_weights and use_data and use_physics:
                    lambda_ph, lambda_bound = update_dynamic_weights(
                        data_loss,
                        ph_loss_val,
                        bound_loss_val,
                        model.net[-1].weight,
                        lambda_ph,
                        lambda_bound,
                    )
                else:
                    lambda_ph, lambda_bound = 1.0, 1.0

                if not use_physics:
                    total_loss = data_loss
                elif use_data:
                    total_loss = (
                        data_loss
                        + lambda_ph * ph_loss_val
                        + lambda_bound * bound_loss_val
                        + norm_loss_val
                        + ortho_loss_val
                    )
                else:
                    total_loss = (
                        lambda_ph * ph_loss_val
                        + lambda_bound * bound_loss_val
                        + 10.0 * norm_loss_val
                        + 10.0 * ortho_loss_val
                    )

                total_loss.backward()
                optimizer.step()

            # --- Monitorización y almacenamiento del estado ---
            if epoch % log_freq == 0 or epoch == epochs:
                print(
                    f"[{label}] Época {epoch:05d} | Pérdida: {total_loss.item():.4e} "
                    f"| Epsilon: {epsilon_exacto:.4f}"
                )
                if use_dynamic_weights and use_data and use_physics:
                    print(
                        f"| Pesos dinámicos -> Física: {lambda_ph:.4f} "
                        f"| Frontera: {lambda_bound:.4f}"
                    )

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())
                historial["ph_loss"].append(ph_loss_val.item())
                historial["bound_loss"].append(bound_loss_val.item())
                historial["ortho_loss"].append(ortho_loss_val.item())
                historial["norm_loss"].append(norm_loss_val.item())
                historial["epsilon"].append(model.epsilon.item())
                historial["lambda_ph"].append(lambda_ph)
                historial["lambda_bound"].append(lambda_bound)

                if save_plots:
                    plot_and_save_results(
                        model,
                        x_train if use_data else None,
                        u_train if use_data else None,
                        x_eval,
                        u_true,
                        epoch,
                        total_loss.item(),
                        n=estado_n,
                        save_dir="img",
                        sistema="pozo_infinito",
                        label=label,
                    )

    pred_eval = model(x_eval).detach()
    error_l2  = calculate_l2_error(pred_eval, u_true)

    return error_l2, timer.elapsed, historial, pred_eval


def main(
    estado_n: int = 2,
    L: float = 1.0,
    epochs: int = 5000,
    lr: float = 0.001,
    num_domain_points: int = 500,
    num_train_points: int = 10,
    train_region: float = 0.2,
    sampler: str = "lhs",
    log_freq: int = 1000,
    use_data: bool = True,
    use_dynamic_weights: bool = False,
    use_orthogonality: bool = False,
    optimizer_name: str = "adam",
    hidden_layers: list = None,
    seed: int = 42,
    save_plots: bool = True,
):
    """
    Orquesta la ejecución integral de un experimento para el sistema de Pozo de Potencial Infinito 1D.
    Sintetiza la generación de datos, el entrenamiento comparativo (PINN vs NN estándar), la contrastación 
    numérica contra métodos finitos (FDM) y el empaquetado de resultados.

    Args:
        estado_n (int, opcional): Nivel de energía cuántico a resolver. Por defecto es 2.
        L (float, opcional): Dominio espacial del sistema cuántico. Por defecto es 1.0.
        epochs (int, opcional): Iteraciones de optimización del modelo. Por defecto es 5000.
        lr (float, opcional): Tasa de aprendizaje inicial. Por defecto es 0.001.
        num_domain_points (int, opcional): Cantidad de puntos de colocación internos. Por defecto es 500.
        num_train_points (int, opcional): Tamaño de la muestra de datos empíricos. Por defecto es 10.
        train_region (float, opcional): Fracción del dominio cubierta por observaciones. Por defecto es 0.2.
        sampler (str, opcional): Algoritmo de discretización de colocación ('lhs' o 'grid'). Por defecto es "lhs".
        log_freq (int, opcional): Frecuencia de escritura del progreso. Por defecto es 1000.
        use_data (bool, opcional): Integra la función de coste basada en datos empíricos. Por defecto es True.
        use_dynamic_weights (bool, opcional): Habilita ponderación adaptativa de los términos de pérdida. Por defecto es False.
        use_orthogonality (bool, opcional): Incorpora regularización de ortogonalidad cuántica. Por defecto es False.
        optimizer_name (str, opcional): Motor de búsqueda del gradiente. Por defecto es "adam".
        hidden_layers (list, opcional): Topología estructural de la red neuronal. Por defecto es [32, 32, 32].
        seed (int, opcional): Entropía controlada para reproducibilidad. Por defecto es 42.
        save_plots (bool, opcional): Habilita la serialización gráfica automática. Por defecto es True.
    """
    if hidden_layers is None:
        hidden_layers = [32, 32, 32]

    # --- 1. Configuración de reproducibilidad y sistema de archivos ---
    set_seed(seed)
    os.makedirs("img", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    exact_epsilon = (estado_n ** 2 * math.pi ** 2) / (2.0 * L ** 2)

    config_exp = {
        "sistema":             "pozo_infinito",
        "estado_n":            estado_n,
        "L":                   L,
        "epochs":              epochs,
        "lr":                  lr,
        "num_domain_points":   num_domain_points,
        "num_train_points":    num_train_points,
        "train_region":        train_region,
        "sampler":             sampler,
        "use_data":            use_data,
        "use_dynamic_weights": use_dynamic_weights,
        "use_orthogonality":   use_orthogonality,
        "optimizer":           optimizer_name,
        "hidden_layers":       hidden_layers,
        "seed":                seed,
        "epsilon_exacto":      exact_epsilon,
    }

    print("=" * 60)
    print(f"Pozo Infinito — Experimento Analítico (Estado n={estado_n})")
    print(
        f"Muestreo: {sampler} | Colocación: {num_domain_points} nodos | "
        f"Train: {num_train_points} obs | Cobertura: {train_region*100}% | "
        f"Épocas: {epochs} | Motor Opt.: {optimizer_name} | "
        f"lr: {lr} | Pesos adaptativos: {use_dynamic_weights} | "
        f"Ortogonalidad: {use_orthogonality} | Semilla: {seed}"
    )
    print("=" * 60)

    # --- 2. Generación unificada de tensores base compartidos ---
    x_min, x_max = 0.0, L
    x_left, x_right = generate_boundary_points(x_min, x_max)

    x_train_end = x_min + train_region * (x_max - x_min)
    margin = 1e-3 * (x_max - x_min)
    x_train = generate_grid_points(x_min + margin, x_train_end, num_train_points, requires_grad=False)
    u_train = psi_infinite_well(x_train, n=estado_n, L=L)

    x_eval = generate_grid_points(x_min, x_max, 500, requires_grad=False)
    u_true = psi_infinite_well(x_eval, n=estado_n, L=L)

    if sampler == "lhs":
        x_domain = generate_lhs_points(x_min, x_max, num_domain_points)
    else:
        x_domain = generate_grid_points(x_min, x_max, num_domain_points)

    shared = dict(
        estado_n=estado_n, L=L,
        x_left=x_left, x_right=x_right,
        x_train=x_train, u_train=u_train,
        x_domain=x_domain, x_eval=x_eval, u_true=u_true,
        epochs=epochs, lr=lr, use_data=use_data,
        use_dynamic_weights=use_dynamic_weights,
        use_orthogonality=use_orthogonality,
        optimizer_name=optimizer_name, hidden_layers=hidden_layers,
        log_freq=log_freq, seed=seed, save_plots=save_plots,
    )

    # --- 3. Despliegue del Modelo Informado por la Física (PINN) ---
    print("\n--- Modelado PINN (Regulación Física Activa) ---")
    error_pinn, time_pinn, hist_pinn, pred_pinn = _train_model(use_physics=True, **shared)

    # --- 4. Despliegue del Modelo Empírico (NN Estándar) ---
    print("\n--- Modelado NN Pura (Basado exclusivamente en Datos) ---")
    error_nn, time_nn, hist_nn, pred_nn = _train_model(use_physics=False, **shared)

    # --- 5. Resolución de Referencia Numérica (Método de Diferencias Finitas) ---
    print("\n--- Resolución Discreta (Método FDM) ---")
    x_np = np.linspace(x_min, x_max, 1000)
    ref  = measure_numerical_reference(
        sistema="pozo_infinito",
        x_or_t=x_np,
        mass=1.0, hbar=1.0, k=estado_n + 1,
    )
    eigenvalues_fdm, eigenvectors_fdm = ref["solution"]
    epsilon_fdm = eigenvalues_fdm[estado_n - 1]
    psi_fdm     = eigenvectors_fdm[:, estado_n - 1]

    # Corrección de la convención de fase (signo) para comparativa analítica L2
    x_eval_np    = x_eval.detach().numpy().flatten()
    u_true_np    = u_true.detach().numpy().flatten()
    fdm_interp   = interp1d(x_np, psi_fdm, kind="cubic")
    psi_fdm_eval = fdm_interp(x_eval_np)
    
    if np.dot(psi_fdm_eval, u_true_np) < 0:
        psi_fdm_eval *= -1

    u_fdm     = torch.tensor(psi_fdm_eval, dtype=torch.float32).unsqueeze(1)
    error_fdm = calculate_l2_error(u_fdm, u_true)
    time_fdm  = ref["time_s"]

    # --- 6. Exportación de Perfil Comparativo Final ---
    if save_plots:
        plot_comparison(
            x_eval=x_eval,
            u_true=u_true,
            pred_pinn=pred_pinn,
            pred_nn=pred_nn,
            pred_numerical=u_fdm,
            error_pinn=error_pinn,
            error_nn=error_nn,
            error_numerical=error_fdm,
            numerical_label="FDM",
            sistema="pozo_infinito",
            train_region_end=x_train_end,
            save_dir="img",
            estado_n=estado_n,
        )

    # --- 7. Consolidación Estructural de Resultados ---
    final_results = {
        "pinn": {
            "error_L2":        error_pinn,
            "time_s":          time_pinn,
            "epsilon_exacto":  exact_epsilon,
            "epsilon_learned": hist_pinn["epsilon"][-1] if hist_pinn["epsilon"] else float("nan"),
        },
        "nn": {
            "error_L2": error_nn if error_nn is not None else float("nan"),
            "time_s":   time_nn,
        },
        "fdm": {
            "error_L2":    error_fdm,
            "time_s":      time_fdm,
            "epsilon_fdm": float(epsilon_fdm),
            "method":      ref["method"],
        },
    }

    historial_completo = {
        "pinn": hist_pinn,
        "nn":   hist_nn,
    }

    print(f"\n{'=' * 60}")
    print(f"RESUMEN DE PRECISIÓN (Pozo Infinito | Estado n={estado_n})")
    print(f"{'Metodología':<15} {'Error L2':>15} {'Cómputo (s)':>15}")
    print(f"{'FDM Discreto':<15} {error_fdm:>15.4e} {time_fdm:>15.4f}")
    nn_str = f"{error_nn:>15.4e}" if error_nn is not None else f"{'N/A':>15}"
    print(f"{'NN Estándar':<15} {nn_str} {time_nn:>15.2f}")
    print(f"{'PINN Analítica':<15} {error_pinn:>15.4e} {time_pinn:>15.2f}")
    
    epsilon_learned = hist_pinn["epsilon"][-1] if hist_pinn["epsilon"] else float("nan")
    print(f"\nValor Teórico E: {exact_epsilon:.4f} | Convergencia PINN: {epsilon_learned:.4f} | Aproximación FDM: {float(epsilon_fdm):.4f}")
    print("=" * 60)

    save_experiment_results(config_exp, final_results, historial_completo)


if __name__ == "__main__":

    SEEDS = [42, 123, 7, 99, 2024, 314, 17, 56, 88, 200]

    # ----------------------------------------------------------------
    # Configuración Paramétrica BASE (Estado Excitado n=2)
    # ----------------------------------------------------------------
    BASE = dict(
        estado_n=2, L=1.0,
        epochs=10000,
        lr=0.001,
        num_domain_points=500, num_train_points=15,
        train_region=0.2, sampler="lhs",
        use_data=True, use_dynamic_weights=False,
        use_orthogonality=False,
        optimizer_name="adam", hidden_layers=[32, 32, 32],
        log_freq=10000, save_plots=True,
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

        # --- Eje 8: Extensión observacional (Expansión al 50% del dominio) ---
        {"train_region": 0.5},

        # --- Eje 9: Intersección (Escasez vs Extensión observacional) ---
        {"num_train_points": 3,  "train_region": 0.5},
        {"num_train_points": 5,  "train_region": 0.5},
        {"num_train_points": 10, "train_region": 0.5},
        {"num_train_points": 30, "train_region": 0.5},

        # --- Eje 10: Complejidad Cuántica (Oscilaciones paramétricas) ---
        {"estado_n": 1},
        {"estado_n": 3},
        {"estado_n": 4},

        # --- Eje 11: Descubrimiento Zero-Shot (Regulación física pura) ---
        {"use_data": False},
    ]

    total_configs = len(variaciones)
    total_runs    = total_configs * len(SEEDS)

    print(f"\n{'=' * 60}")
    print(f"ANÁLISIS DE SENSIBILIDAD METODOLÓGICA — Pozo de Potencial Infinito 1D")
    print(f"Batería experimental: {total_configs} Configuraciones | {len(SEEDS)} Semillas | {total_runs} Ejecuciones totales")
    print(f"{'=' * 60}\n")

    for i, variacion in enumerate(variaciones, start=1):
        if i < 9:
            continue
        
        config = {**BASE, **variacion}
        print(f"\n{'#' * 60}")
        print(f"Evaluación de Configuración {i}/{total_configs}: {variacion if variacion else 'Modelo BASE'}")
        print(f"{'#' * 60}")

        for j, seed in enumerate(SEEDS, start=1):
            print(f"\n  Instancia Estocástica (Seed) {j}/{len(SEEDS)}: {seed}")
            main(**config, seed=seed)