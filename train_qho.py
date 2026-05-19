import os

import numpy as np
import torch
import torch.optim as optim
from scipy.interpolate import interp1d

from src.exact_solutions import psi_QHO
from src.loss_functions import (
    boundary_loss,
    normalization_loss,
    orthogonality_loss,
    physics_loss_QHO,
)
from src.models import PINN
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
    Entrena un modelo (PINN o NN pura) y devuelve error L2, tiempo, historial
    y predicciones finales. Recibe los datos ya generados para garantizar
    condiciones idénticas entre métodos.
    """
    set_seed(seed)

    epsilon_init   = 0
    model = PINN(hidden_layers=hidden_layers)
    with torch.no_grad():
        model.epsilon.fill_(epsilon_init)

    label = "PINN" if use_physics else "NN"

    historial = {
        "epoch":        [],
        "total_loss":   [],
        "data_loss":    [],
        "ph_loss":      [],
        "bound_loss":   [],
        "ortho_loss":   [],
        "norm_loss":    [],
        "epsilon":      [],
        "lambda_ph":    [],
        "lambda_bound": [],
    }

    # ── Caso degenerado: NN pura sin datos ─────────────────────────────────
    if not use_physics and not use_data:
        print(f"[{label}] Omitida: sin datos ni física no hay nada que optimizar.")
        pred_eval = model(x_eval).detach()
        error_l2  = calculate_l2_error(pred_eval, u_true)
        return error_l2, 0.0, historial, pred_eval

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=500)
    elif optimizer_name == "adam+lbfgs":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no reconocido. "
            f"Usa 'adam', 'lbfgs' o 'adam+lbfgs'."
        )

    lambda_ph    = 1.0
    lambda_bound = 1.0

    with Timer() as timer:
        for epoch in range(1, epochs + 1):

            # Cambio de fase Adam → L-BFGS en la mitad del entrenamiento
            if optimizer_name == "adam+lbfgs" and epoch == epochs // 2 + 1:
                print(f"\n[{label}] Cambiando a L-BFGS en época {epoch}\n")
                optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=500)

            def closure(lph=lambda_ph, lbound=lambda_bound):
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
                    ph_loss_val    = physics_loss_QHO(model, x_domain)
                    bound_loss_val = boundary_loss(model, x_left, x_right)

                    if use_orthogonality and estado_n > 0:
                        psi_prev = psi_QHO(x_domain, n=estado_n - 1).detach()
                        ortho_loss_val = orthogonality_loss(
                            model, x_domain, psi_prev, domain_length=20.0
                        )
                    if not use_data:
                        norm_loss_val = normalization_loss(
                            model, x_domain, domain_length=20.0
                        )

                if not use_physics:
                    total = data_loss
                elif use_data:
                    total = (
                        data_loss
                        + lph * ph_loss_val
                        + lbound * bound_loss_val
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

            # Paso del optimizador
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
                    ph_loss_val    = physics_loss_QHO(model, x_domain)
                    bound_loss_val = boundary_loss(model, x_left, x_right)

                    if use_orthogonality and estado_n > 0:
                        psi_prev = psi_QHO(x_domain, n=estado_n - 1).detach()
                        ortho_loss_val = orthogonality_loss(
                            model, x_domain, psi_prev, domain_length=20.0
                        )
                    if not use_data:
                        norm_loss_val = normalization_loss(
                            model, x_domain, domain_length=20.0
                        )

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

            if epoch % log_freq == 0 or epoch == epochs:
                print(
                    f"[{label}] Época {epoch:05d} | Pérdida: {total_loss.item():.4e} "
                    f"| Epsilon: {model.epsilon.item():.4f}"
                )
                if use_dynamic_weights and use_data and use_physics:
                    print(
                        f"| Pesos -> Física: {lambda_ph:.4f} "
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
                        sistema="qho",
                        label=label,
                    )

    pred_eval = model(x_eval).detach()
    error_l2  = calculate_l2_error(pred_eval, u_true)

    return error_l2, timer.elapsed, historial, pred_eval


def main(
    estado_n: int = 0,
    epochs: int = 5000,
    lr: float = 0.001,
    num_domain_points: int = 500,
    num_train_points: int = 15,
    train_region: float = 0.5,
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
    if hidden_layers is None:
        hidden_layers = [32, 32, 32]

    # 1. Seed y directorios
    set_seed(seed)
    os.makedirs("img", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    exact_epsilon = float(estado_n + 0.5)

    config_exp = {
        "sistema":             "qho",
        "estado_n":            estado_n,
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
    print(f"QHO — Experimento completo (n={estado_n})")
    print(
        f"Sampler: {sampler} | Puntos dominio: {num_domain_points} | "
        f"Puntos train: {num_train_points} | Región: {train_region} | "
        f"Épocas: {epochs} | Optimizador: {optimizer_name} | "
        f"lr: {lr} | Pesos din.: {use_dynamic_weights} | "
        f"Ortogonalidad: {use_orthogonality} | Seed: {seed}"
    )
    print("=" * 60)

    # 2. Datos compartidos — generados UNA sola vez con la seed ya fijada
    x_min, x_max = -10.0, 10.0
    x_left, x_right = generate_boundary_points(x_min, x_max)

    x_train_end = x_min + train_region * (x_max - x_min)
    margin = 1e-3 * (x_max - x_min)
    x_train = generate_grid_points(x_min + margin, x_train_end, num_train_points, requires_grad=False)
    u_train = psi_QHO(x_train, n=estado_n)

    x_eval = generate_grid_points(x_min, x_max, 500, requires_grad=False)
    u_true = psi_QHO(x_eval, n=estado_n)

    if sampler == "lhs":
        x_domain = generate_lhs_points(x_min, x_max, num_domain_points)
    else:
        x_domain = generate_grid_points(x_min, x_max, num_domain_points)

    shared = dict(
        estado_n=estado_n,
        x_left=x_left, x_right=x_right,
        x_train=x_train, u_train=u_train,
        x_domain=x_domain, x_eval=x_eval, u_true=u_true,
        epochs=epochs, lr=lr, use_data=use_data,
        use_dynamic_weights=use_dynamic_weights,
        use_orthogonality=use_orthogonality,
        optimizer_name=optimizer_name, hidden_layers=hidden_layers,
        log_freq=log_freq, seed=seed, save_plots=save_plots,
    )

    # 3. Entrenar PINN
    print("\n--- PINN (con física) ---")
    error_pinn, time_pinn, hist_pinn, pred_pinn = _train_model(use_physics=True, **shared)

    # 4. Entrenar NN pura (omitida automáticamente si use_data=False)
    print("\n--- NN pura (sin física) ---")
    error_nn, time_nn, hist_nn, pred_nn = _train_model(use_physics=False, **shared)

    # 5. Referencia numérica FDM
    print("\n--- FDM ---")
    x_np = np.linspace(x_min, x_max, 1000)
    ref  = measure_numerical_reference(
        sistema="qho", x_or_t=x_np, mass=1.0, omega=1.0, hbar=1.0, k=estado_n + 1
    )
    eigenvalues_fdm, eigenvectors_fdm = ref["solution"]
    epsilon_fdm = eigenvalues_fdm[estado_n]
    psi_fdm     = eigenvectors_fdm[:, estado_n]

    # Corrección de signo
    x_eval_np    = x_eval.detach().numpy().flatten()
    u_true_np    = u_true.detach().numpy().flatten()
    fdm_interp   = interp1d(x_np, psi_fdm, kind="cubic")
    psi_fdm_eval = fdm_interp(x_eval_np)
    if np.dot(psi_fdm_eval, u_true_np) < 0:
        psi_fdm_eval *= -1

    u_fdm     = torch.tensor(psi_fdm_eval, dtype=torch.float32).unsqueeze(1)
    error_fdm = calculate_l2_error(u_fdm, u_true)
    time_fdm  = ref["time_s"]

    # 6. Gráfica comparativa final
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
            sistema="qho",
            train_region_end=x_train_end,
            save_dir="img",
            estado_n=estado_n,
        )

    # 7. Resultados finales
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
    print(f"RESULTADOS FINALES (QHO | n={estado_n})")
    print(f"{'Método':<12} {'Error L2':>12} {'Tiempo (s)':>12}")
    print(f"{'FDM':<12} {error_fdm:>12.4e} {time_fdm:>12.4f}")
    nn_str = f"{error_nn:>12.4e}" if error_nn is not None else f"{'N/A':>12}"
    print(f"{'NN pura':<12} {nn_str} {time_nn:>12.2f}")
    print(f"{'PINN':<12} {error_pinn:>12.4e} {time_pinn:>12.2f}")
    epsilon_learned = hist_pinn["epsilon"][-1] if hist_pinn["epsilon"] else float("nan")
    print(f"\nAutovalor exacto: {exact_epsilon:.4f} | PINN: {epsilon_learned:.4f} | FDM: {float(epsilon_fdm):.4f}")
    print("=" * 60)

    save_experiment_results(config_exp, final_results, historial_completo)

if __name__ == "__main__":

    SEEDS = [42, 123, 7, 99, 2024, 314, 17, 56, 88, 200]

    # ----------------------------------------------------------------
    # Configuración BASE  (n=1, train_region=0.5)
    # ----------------------------------------------------------------
    BASE = dict(
        estado_n=1, 
        epochs=10000,
        lr=0.001,
        num_domain_points=500, num_train_points=15,
        train_region=0.5, sampler="lhs",
        use_data=True, use_dynamic_weights=False,
        use_orthogonality=False,
        optimizer_name="adam", hidden_layers=[32, 32, 32],
        log_freq=10000, save_plots=False,
    )
    # ----------------------------------------------------------------
    # Todas las configuraciones del estudio de sensibilidad
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

        # --- Eje 7: Puntos de entrenamiento (región 50%) ---
        {"num_train_points": 3},
        {"num_train_points": 5},
        {"num_train_points": 10},
        {"num_train_points": 30},

        # --- Eje 8: Región de entrenamiento (15 puntos base) ---
        {"train_region": 0.3},

        # --- Eje 9: Cruce puntos × región 0.3 ---
        {"num_train_points": 3,  "train_region": 0.3},
        {"num_train_points": 5,  "train_region": 0.3},
        {"num_train_points": 10, "train_region": 0.3},
        {"num_train_points": 30, "train_region": 0.3},

        # --- Eje 10: Estado cuántico (solo configuración BASE) ---
        {"estado_n": 0},
        {"estado_n": 2},
        {"estado_n": 3},

        # --- Eje 11: Sin datos (solo física) ---
        {"use_data": False},

        # --- Eje 12: Ortogonalidad ---
        {"use_orthogonality": True},
    ]

    total_configs = len(variaciones)
    total_runs    = total_configs * len(SEEDS)

    print(f"\n{'=' * 60}")
    print(f"ESTUDIO DE SENSIBILIDAD — QHO")
    print(f"Configuraciones: {total_configs} | Seeds: {len(SEEDS)} | Total ejecuciones: {total_runs}")
    print(f"{'=' * 60}\n")

    for i, variacion in enumerate(variaciones, start=1):
        config = {**BASE, **variacion}
        print(f"\n{'#' * 60}")
        print(f"Configuración {i}/{total_configs}: {variacion if variacion else 'BASE'}")
        print(f"{'#' * 60}")

        for j, seed in enumerate(SEEDS, start=1):
            print(f"\n  Seed {j}/{len(SEEDS)}: {seed}")
            main(**config, seed=seed)
