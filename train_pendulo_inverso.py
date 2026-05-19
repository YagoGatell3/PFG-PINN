import math
import os

import numpy as np
import torch
import torch.optim as optim

from src.loss_functions import physics_loss_damped_pendulum
from src.models import PINNDampedPendulum
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


def _train_pinn(
    t_train: torch.Tensor,
    u_train: torch.Tensor,
    t_domain: torch.Tensor,
    t_eval: torch.Tensor,
    u_true: torch.Tensor,
    L: float,
    g_true: float,
    mu_true: float,
    epochs: int,
    lr: float,
    use_dynamic_weights: bool,
    optimizer_name: str,
    hidden_layers: list,
    log_freq: int,
    seed: int,
    save_plots: bool = True,
) -> tuple[float, float, dict, torch.Tensor]:
    """
    Entrena la PINN para el problema inverso: descubre g y mu a partir
    de datos (posiblemente ruidosos) y la ecuación del movimiento.
    """
    set_seed(seed)
    model = PINNDampedPendulum(hidden_layers=hidden_layers)
    label = "PINN"

    historial = {
        "epoch":      [],
        "total_loss": [],
        "data_loss":  [],
        "ph_loss":    [],
        "g_pred":     [],
        "mu_pred":    [],
        "lambda_ph":  [],
    }

    if optimizer_name == "adam":
        optimizer = optim.Adam([
            {"params": model.net.parameters(), "lr": lr},
            {"params": [model.g, model.mu],    "lr": lr * 20},
        ])
    elif optimizer_name == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=50)
    elif optimizer_name == "adam+lbfgs":
        optimizer = optim.Adam([
            {"params": model.net.parameters(), "lr": lr},
            {"params": [model.g, model.mu],    "lr": lr * 20},
        ])
    else:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no reconocido. "
            f"Usa 'adam', 'lbfgs' o 'adam+lbfgs'."
        )

    lambda_ph = 1.0

    with Timer() as timer:
        for epoch in range(1, epochs + 1):

            if optimizer_name == "adam+lbfgs" and epoch == int(epochs * 0.85) + 1:
                print(f"\n[{label}] Cambiando a L-BFGS en época {epoch}\n")
                optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=50)

            def closure(lph=lambda_ph):
                optimizer.zero_grad()
                data_loss   = torch.mean((model(t_train) - u_train) ** 2)
                ph_loss_val = physics_loss_damped_pendulum(model, t_domain, L=L)
                total = data_loss + lph * ph_loss_val
                total.backward()
                return total

            if optimizer_name == "lbfgs" or (
                optimizer_name == "adam+lbfgs" and epoch >= int(epochs * 0.85) + 1
            ):
                result     = optimizer.step(closure)
                total_loss = result if result is not None else torch.tensor(0.0)
                with torch.no_grad():
                    data_loss = torch.mean((model(t_train) - u_train) ** 2)
                ph_loss_val = torch.tensor(0.0)
            else:
                optimizer.zero_grad()
                data_loss   = torch.mean((model(t_train) - u_train) ** 2)
                ph_loss_val = physics_loss_damped_pendulum(model, t_domain, L=L)

                if use_dynamic_weights:
                    dummy_bound = torch.tensor(0.0, requires_grad=True)
                    lambda_ph, _ = update_dynamic_weights(
                        data_loss, ph_loss_val, dummy_bound,
                        model.net[-1].weight, lambda_ph, 1.0,
                    )
                else:
                    lambda_ph = 1.0

                total_loss = data_loss + lambda_ph * ph_loss_val
                total_loss.backward()
                # Gradient clipping para evitar explosiones antes de entrar en L-BFGS
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if epoch % log_freq == 0 or epoch == epochs:
                g_pred  = model.g.item()
                mu_pred = model.mu.item()
                print(
                    f"[{label}] Época {epoch:05d} | Pérdida: {total_loss.item():.4e} "
                    f"| g: {g_pred:.4f} (real: {g_true:.4f}) "
                    f"| mu: {mu_pred:.4f} (real: {mu_true:.4f})"
                )
                if use_dynamic_weights:
                    print(f"            | Peso Física: {lambda_ph:.4f}")

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())
                historial["ph_loss"].append(ph_loss_val.item())
                historial["g_pred"].append(g_pred)
                historial["mu_pred"].append(mu_pred)
                historial["lambda_ph"].append(lambda_ph)

                if save_plots:
                    plot_and_save_results(
                        model, t_train, u_train, t_eval, u_true,
                        epoch, total_loss.item(), n=0,
                        save_dir="img", sistema="pendulo_inverso", label=label,
                    )

    pred_eval = model(t_eval).detach()
    error_l2  = calculate_l2_error(pred_eval, u_true)

    return error_l2, timer.elapsed, historial, pred_eval


def _train_nn(
    t_train: torch.Tensor,
    u_train: torch.Tensor,
    t_eval: torch.Tensor,
    u_true: torch.Tensor,
    epochs: int,
    lr: float,
    optimizer_name: str,
    hidden_layers: list,
    log_freq: int,
    seed: int,
    save_plots: bool = True,
) -> tuple[float, float, dict, torch.Tensor]:
    """
    Entrena una NN pura (sin física) con los mismos datos ruidosos.
    No conoce g ni mu — solo ajusta los datos.
    Sirve como baseline para comparar con la PINN.
    """
    set_seed(seed)
    model = PINNDampedPendulum(hidden_layers=hidden_layers)
    label = "NN"

    historial = {
        "epoch":      [],
        "total_loss": [],
        "data_loss":  [],
    }

    if optimizer_name in ("adam", "adam+lbfgs"):
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=50)
    else:
        raise ValueError(f"Optimizador '{optimizer_name}' no reconocido.")

    with Timer() as timer:
        for epoch in range(1, epochs + 1):

            if optimizer_name == "adam+lbfgs" and epoch == int(epochs * 0.85) + 1:
                print(f"\n[{label}] Cambiando a L-BFGS en época {epoch}\n")
                optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=50)

            def closure():
                optimizer.zero_grad()
                data_loss = torch.mean((model(t_train) - u_train) ** 2)
                data_loss.backward()
                return data_loss

            if optimizer_name == "lbfgs" or (
                optimizer_name == "adam+lbfgs" and epoch >= int(epochs * 0.85) + 1
            ):
                result     = optimizer.step(closure)
                total_loss = result if result is not None else torch.tensor(0.0)
                with torch.no_grad():
                    data_loss = torch.mean((model(t_train) - u_train) ** 2)
            else:
                optimizer.zero_grad()
                data_loss  = torch.mean((model(t_train) - u_train) ** 2)
                total_loss = data_loss
                total_loss.backward()
                optimizer.step()

            if epoch % log_freq == 0 or epoch == epochs:
                print(f"[{label}] Época {epoch:05d} | Pérdida: {total_loss.item():.4e}")

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())

                if save_plots:
                    plot_and_save_results(
                        model, t_train, u_train, t_eval, u_true,
                        epoch, total_loss.item(), n=0,
                        save_dir="img", sistema="pendulo_inverso", label=label,
                    )

    pred_eval = model(t_eval).detach()
    error_l2  = calculate_l2_error(pred_eval, u_true)

    return error_l2, timer.elapsed, historial, pred_eval


def main(
    t_max: float = 20.0,
    L: float = 1.0,
    g_true: float = 9.81,
    mu_true: float = 0.5,
    theta0: float = math.pi / 4.0,
    omega0: float = 0.0,
    noise_std: float = 0.05,
    epochs: int = 15000,
    lr: float = 0.001,
    num_domain_points: int = 2000,
    num_train_points: int = 80,
    train_region: float = 1.0,
    sampler: str = "lhs",
    log_freq: int = 1000,
    use_dynamic_weights: bool = False,
    optimizer_name: str = "adam",
    hidden_layers: list = None,
    seed: int = 42,
    save_plots: bool = True,
):
    if hidden_layers is None:
        hidden_layers = [32, 32, 32, 32, 32]

    # 1. Seed y directorios
    set_seed(seed)
    os.makedirs("img", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    config_exp = {
        "sistema":             "pendulo_inverso",
        "t_max":               t_max,
        "L":                   L,
        "g_true":              g_true,
        "mu_true":             mu_true,
        "theta0":              theta0,
        "omega0":              omega0,
        "noise_std":           noise_std,
        "epochs":              epochs,
        "lr":                  lr,
        "num_domain_points":   num_domain_points,
        "num_train_points":    num_train_points,
        "train_region":        train_region,
        "sampler":             sampler,
        "use_dynamic_weights": use_dynamic_weights,
        "optimizer":           optimizer_name,
        "hidden_layers":       hidden_layers,
        "seed":                seed,
    }

    print("=" * 60)
    print("Péndulo Amortiguado — Problema Inverso")
    print(
        f"Sampler: {sampler} | Puntos dominio: {num_domain_points} | "
        f"Puntos train: {num_train_points} | Región: {train_region} | "
        f"Épocas: {epochs} | Optimizador: {optimizer_name} | "
        f"lr: {lr} | Pesos din.: {use_dynamic_weights} | "
        f"Ruido: {noise_std} | Seed: {seed}"
    )
    print(f"Buscando g={g_true} y mu={mu_true}")
    print("=" * 60)

    # 2. Ground truth con RK4
    t_rk4 = np.linspace(0.0, t_max, 1000)
    ref_rk4 = measure_numerical_reference(
        sistema="pendulo_inverso",
        x_or_t=t_rk4,
        g=g_true, mu=mu_true, L=L, theta0=theta0, omega0=omega0,
    )
    theta_rk4 = ref_rk4["solution"][0]

    t_eval = torch.tensor(t_rk4, dtype=torch.float32).unsqueeze(1)
    u_true = torch.tensor(theta_rk4, dtype=torch.float32).unsqueeze(1)

    # 3. Datos de entrenamiento (ground truth + ruido, hasta train_region)
    t_train_end = t_max * train_region
    idx_all     = np.where(t_rk4 <= t_train_end)[0]
    idx_train   = idx_all[
        np.linspace(0, len(idx_all) - 1, num_train_points).astype(int)
    ]

    t_train_np     = t_rk4[idx_train]
    theta_train_np = theta_rk4[idx_train]

    np.random.seed(seed)
    noise             = np.random.normal(0, noise_std, size=theta_train_np.shape)
    theta_train_noisy = theta_train_np + noise

    t_train = torch.tensor(t_train_np,        dtype=torch.float32).unsqueeze(1)
    u_train = torch.tensor(theta_train_noisy, dtype=torch.float32).unsqueeze(1)

    # 4. Puntos de colocación (todo el dominio temporal)
    if sampler == "lhs":
        t_domain = generate_lhs_points(0.0, t_max, num_domain_points)
    else:
        t_domain = generate_grid_points(0.0, t_max, num_domain_points)

    # 5. Entrenar PINN (descubre g y mu)
    print("\n--- PINN (problema inverso) ---")
    error_pinn, time_pinn, hist_pinn, pred_pinn = _train_pinn(
        t_train=t_train, u_train=u_train,
        t_domain=t_domain, t_eval=t_eval, u_true=u_true,
        L=L, g_true=g_true, mu_true=mu_true,
        epochs=epochs, lr=lr,
        use_dynamic_weights=use_dynamic_weights,
        optimizer_name=optimizer_name, hidden_layers=hidden_layers,
        log_freq=log_freq, seed=seed, save_plots=save_plots,
    )

    # 6. Entrenar NN pura (mismos datos ruidosos, sin física)
    print("\n--- NN pura (sin física, mismos datos) ---")
    error_nn, time_nn, hist_nn, pred_nn = _train_nn(
        t_train=t_train, u_train=u_train,
        t_eval=t_eval, u_true=u_true,
        epochs=epochs, lr=lr,
        optimizer_name=optimizer_name, hidden_layers=hidden_layers,
        log_freq=log_freq, seed=seed, save_plots=save_plots,
    )

    # 7. RK4 ya calculado
    print("\n--- RK4 (referencia numérica) ---")
    time_rk4 = ref_rk4["time_s"]

    # 8. Gráfica comparativa final
    if save_plots:
        plot_comparison(
            x_eval=t_eval,
            u_true=u_true,
            pred_pinn=pred_pinn,
            pred_nn=pred_nn,
            pred_numerical=u_true,
            error_pinn=error_pinn,
            error_nn=error_nn,
            error_numerical=0.0,
            numerical_label="RK4",
            sistema="pendulo_inverso",
            train_region_end=t_max * train_region,
            save_dir="img",
            estado_n=0,
        )

    # 9. Resultados finales
    g_final  = hist_pinn["g_pred"][-1]  if hist_pinn["g_pred"]  else float("nan")
    mu_final = hist_pinn["mu_pred"][-1] if hist_pinn["mu_pred"] else float("nan")

    final_results = {
        "pinn": {
            "error_L2":          error_pinn,
            "time_s":            time_pinn,
            "g_true":            g_true,
            "g_learned":         g_final,
            "error_relativo_g":  abs(g_true  - g_final)  / g_true,
            "mu_true":           mu_true,
            "mu_learned":        mu_final,
            "error_relativo_mu": abs(mu_true - mu_final) / mu_true,
        },
        "nn": {
            "error_L2": error_nn,
            "time_s":   time_nn,
        },
        "rk4": {
            "time_s": time_rk4,
            "method": ref_rk4["method"],
        },
    }

    historial_completo = {
        "pinn": hist_pinn,
        "nn":   hist_nn,
    }

    print(f"\n{'=' * 60}")
    print("RESULTADOS FINALES (Péndulo Inverso)")
    print(f"{'Parámetro':<12} {'Real':>10} {'Predicho':>10} {'Error (%)':>10}")
    print(f"{'g':<12} {g_true:>10.4f} {g_final:>10.4f} "
          f"{abs(g_true - g_final) / g_true * 100:>10.2f}")
    print(f"{'mu':<12} {mu_true:>10.4f} {mu_final:>10.4f} "
          f"{abs(mu_true - mu_final) / mu_true * 100:>10.2f}")
    print(f"\n{'Método':<12} {'Error L2':>12} {'Tiempo (s)':>12}")
    print(f"{'RK4':<12} {'0.0000e+00':>12} {time_rk4:>12.4f}")
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
        t_max=20.0, L=1.0,
        g_true=9.81, mu_true=0.5,
        theta0=math.pi / 2.0,
        omega0=0.0,
        noise_std=0.05,
        epochs=10000,
        lr=0.001,
        num_domain_points=2000,
        num_train_points=15,
        train_region=0.5,
        sampler="lhs",
        use_dynamic_weights=False,
        optimizer_name="adam+lbfgs",
        hidden_layers=[32, 32, 32, 32, 32],
        log_freq=10000,
        save_plots=False,
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
        {"num_domain_points": 500},
        {"num_domain_points": 1000},

        # --- Eje 2: Sampler ---
        {"sampler": "grid"},

        # --- Eje 3: Optimizador ---
        {"optimizer_name": "adam"},

        # --- Eje 4: Arquitectura ---
        {"hidden_layers": [64, 64, 64, 64, 64]},

        # --- Eje 5: Pesos dinámicos ---
        {"use_dynamic_weights": True},

        # --- Eje 6: Learning rate ---
        {"lr": 0.01},

        # --- Eje 7: Ruido ---
        {"noise_std": 0.0},
        {"noise_std": 0.1},
        {"noise_std": 0.2},

        # --- Eje 8: Puntos de entrenamiento (región 0.5) ---
        {"num_train_points": 5},
        {"num_train_points": 10},
        {"num_train_points": 30},

        # --- Eje 9: Región de entrenamiento (15 puntos base) ---
        {"train_region": 0.3},
        {"train_region": 0.7},

        # --- Eje 10: Cruce puntos × región 0.7 ---
        {"num_train_points": 5,  "train_region": 0.7},
        {"num_train_points": 10, "train_region": 0.7},
        {"num_train_points": 30, "train_region": 0.7},
    ]

    total_configs = len(variaciones)
    total_runs    = total_configs * len(SEEDS)

    print(f"\n{'=' * 60}")
    print(f"ESTUDIO DE SENSIBILIDAD — Péndulo Inverso")
    print(f"Configuraciones: {total_configs} | Seeds: {len(SEEDS)} | Total ejecuciones: {total_runs}")
    print(f"{'=' * 60}\n")

    for i, variacion in enumerate(variaciones, start=1):
        config = {**BASE, **variacion}
        if i < 13:  # saltar configuraciones ya ejecutadas
            continue
        print(f"\n{'#' * 60}")
        print(f"Configuración {i}/{total_configs}: {variacion if variacion else 'BASE'}")
        print(f"{'#' * 60}")

        for j, seed in enumerate(SEEDS, start=1):
            if i == 13 and j <= 4:  # saltar seeds ya ejecutadas de la config 13
                continue
            print(f"\n  Seed {j}/{len(SEEDS)}: {seed}")
            main(**config, seed=seed)