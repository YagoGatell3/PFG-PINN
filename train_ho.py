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
    Entrena un modelo (PINN o NN pura) y devuelve error L2, tiempo, historial
    y predicciones finales. Recibe los datos ya generados para garantizar
    condiciones idénticas entre métodos.
    """
    set_seed(seed)
    model = PINNDynamic(hidden_layers=hidden_layers)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=250)
    elif optimizer_name == "adam+lbfgs":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no reconocido. "
            f"Usa 'adam', 'lbfgs' o 'adam+lbfgs'."
        )

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

            # Cambio de fase Adam → L-BFGS en la mitad del entrenamiento
            if optimizer_name == "adam+lbfgs" and epoch == epochs // 2 + 1:
                print(f"\n[{label}] Cambiando a L-BFGS en época {epoch}\n")
                optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=250)

            def closure(lph=lambda_ph, lic=lambda_ic):
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

                if not use_physics:
                    total = data_loss
                elif use_data:
                    total = data_loss + lph * ph_loss_val + lic * ic_loss_val
                else:
                    total = lph * ph_loss_val + 10.0 * ic_loss_val

                total.backward()
                return total

            # Paso del optimizador
            if optimizer_name == "lbfgs" or (optimizer_name == "adam+lbfgs" and epoch >= epochs // 2 + 1):
                result = optimizer.step(closure)
                total_loss = result if result is not None else torch.tensor(0.0)
                # Para el historial: data_loss sí se puede calcular sin grad,
                # ph_loss e ic_loss no (necesitan autodiferenciación) — usamos 0.0
                # como placeholder, el total_loss sí es correcto
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

            if epoch % log_freq == 0 or epoch == epochs:
                print(f"[{label}] Época {epoch:05d} | Pérdida: {total_loss.item():.4e}")
                if use_dynamic_weights and use_data and use_physics:
                    print(
                        f"            | Pesos -> Física: {lambda_ph:.4f} "
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
    if hidden_layers is None:
        hidden_layers = [32, 32, 32]

    # 1. Seed y directorios
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
    print("Oscilador Clásico — Experimento completo")
    print(
        f"Sampler: {sampler} | Puntos dominio: {num_domain_points} | "
        f"Puntos train: {num_train_points} | Región: {int(train_region*100)}% | "
        f"Épocas: {epochs} | Optimizador: {optimizer_name} | "
        f"lr: {lr} | Pesos din.: {use_dynamic_weights} | Seed: {seed}"
    )
    print("=" * 60)

    # 2. Datos compartidos — generados UNA sola vez con la seed ya fijada
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

    # 3. Entrenar PINN
    print("\n--- PINN (con física) ---")
    error_pinn, time_pinn, hist_pinn, pred_pinn = _train_model(use_physics=True, **shared)

    # 4. Entrenar NN pura (mismos datos, misma seed → inicialización idéntica)
    print("\n--- NN pura (sin física) ---")
    error_nn, time_nn, hist_nn, pred_nn = _train_model(use_physics=False, **shared)

    # 5. Referencia numérica RK4
    print("\n--- RK4 ---")
    t_np = np.linspace(0.0, t_max, 500)
    ref  = measure_numerical_reference(
        sistema="oscilador_clasico",
        x_or_t=t_np,
        mass=mass, k=k, u0=u_0, v0=v_0,
    )
    u_rk4     = torch.tensor(ref["solution"][0], dtype=torch.float32).unsqueeze(1)
    error_rk4 = calculate_l2_error(u_rk4, u_true)
    time_rk4  = ref["time_s"]

    # 6. Gráfica comparativa final
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

    # 7. Resultados finales unificados en un único JSON
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
    print("RESULTADOS FINALES")
    print(f"{'Método':<12} {'Error L2':>12} {'Tiempo (s)':>12}")
    print(f"{'RK4':<12} {error_rk4:>12.4e} {time_rk4:>12.4f}")
    print(f"{'NN pura':<12} {error_nn:>12.4e} {time_nn:>12.2f}")
    print(f"{'PINN':<12} {error_pinn:>12.4e} {time_pinn:>12.2f}")
    print("=" * 60)

    save_experiment_results(config_exp, final_results, historial_completo)
    
    
if __name__ == "__main__":
 
    SEEDS = [42, 123, 7, 99, 2024, 314, 17, 56, 88, 200]
 
    # ----------------------------------------------------------------
    # Configuración BASE
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
        save_plots=True,   # sin gráficas durante el estudio masivo
    )
 
    # ----------------------------------------------------------------
    # Todas las configuraciones del estudio de sensibilidad
    # Cada entrada es un dict con los parámetros que difieren de BASE
    # ----------------------------------------------------------------
    variaciones = [
        # --- BASE ---
        {},
 
        # --- Eje 1: Puntos de colocación ---
        {"num_domain_points": 50},
        {"num_domain_points": 100},
        {"num_domain_points": 250},
        {"num_domain_points": 1000},
 
        # --- Eje 2: Sampler ---
        {"sampler": "grid"},
 
        # --- Eje 3: Optimizador ---
        {"optimizer_name": "adam+lbfgs"},
 
        # --- Eje 4: Arquitectura ---
        {"hidden_layers": [64, 64, 64]},
 
        # --- Eje 5: Pesos dinámicos ---
        {"use_dynamic_weights": True},
 
        # --- Eje 6: Learning rate ---
        {"lr": 0.01},
 
        # --- Eje 7: Puntos de entrenamiento (región 20%) ---
        {"num_train_points": 3},
        {"num_train_points": 5},
        {"num_train_points": 10},
        {"num_train_points": 30},
 
        # --- Eje 8: Región de entrenamiento (15 puntos base) ---
        {"train_region": 0.5},
 
        # --- Eje 9: Cruce puntos × región 50% ---
        {"num_train_points": 3,  "train_region": 0.5},
        {"num_train_points": 5,  "train_region": 0.5},
        {"num_train_points": 10, "train_region": 0.5},
        {"num_train_points": 30, "train_region": 0.5},
    ]
 
    total_configs = len(variaciones)
    total_runs    = total_configs * len(SEEDS)
 
    print(f"\n{'=' * 60}")
    print(f"ESTUDIO DE SENSIBILIDAD — Oscilador Clásico")
    print(f"Configuraciones: {total_configs} | Seeds: {len(SEEDS)} | Total ejecuciones: {total_runs}")
    print(f"{'=' * 60}\n")
 
    for i, variacion in enumerate(variaciones, start=1):
        if i < 8:  # saltar configuraciones ya ejecutadas
            continue
        config = {**BASE, **variacion}
        print(f"\n{'#' * 60}")
        print(f"Configuración {i}/{total_configs}: {variacion if variacion else 'BASE'}")
        print(f"{'#' * 60}")

        for j, seed in enumerate(SEEDS, start=1):
            print(f"\n  Seed {j}/{len(SEEDS)}: {seed}")
            main(**config, seed=seed)