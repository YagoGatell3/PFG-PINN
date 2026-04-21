import os
import numpy as np
import torch
import torch.optim as optim

from src.exact_solutions import heat_exact
from src.loss_functions import (
    boundary_loss_heat,
    initial_condition_loss_heat,
    physics_loss_heat_inverse,
)
from src.models import PINNHeatInverse
from src.utils import (
    calculate_l2_error, 
    save_experiment_results, 
    set_seed, 
    Timer, 
    measure_numerical_reference)


import matplotlib.pyplot as plt
from scipy.stats import qmc


def sample_collocation_2d(
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    n_points: int,
    sampler: str = "lhs",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Muestrea puntos de colocación (x, t) en el dominio 2D.
    """
    if sampler == "lhs":
        engine = qmc.LatinHypercube(d=2)
        pts    = engine.random(n=n_points)
        pts[:, 0] = x_min + pts[:, 0] * (x_max - x_min)
        pts[:, 1] = t_min + pts[:, 1] * (t_max - t_min)
    else:
        nx = int(np.sqrt(n_points))
        nt = nx
        xv, tv = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(t_min, t_max, nt),
        )
        pts = np.stack([xv.ravel(), tv.ravel()], axis=1)

    x = torch.tensor(pts[:, 0:1], dtype=torch.float32, requires_grad=True)
    t = torch.tensor(pts[:, 1:2], dtype=torch.float32, requires_grad=True)
    return x, t


def plot_heat_inverse_results(
    model: torch.nn.Module,
    x_eval: torch.Tensor,
    t_snapshots: list[float],
    alpha_true: float,
    L: float,
    epoch: int,
    loss: float,
    save_dir: str,
):
    """
    Compara u(x,t) PINN vs solución analítica en varios instantes
    e incluye el valor de alpha predicho en el título.
    """
    n_snaps = len(t_snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4), sharey=True)
    if n_snaps == 1:
        axes = [axes]

    x_np = x_eval.detach().numpy().flatten()

    for i, t_val in enumerate(t_snapshots):
        t_tensor = torch.full((len(x_eval), 1), t_val)
        with torch.no_grad():
            u_exact = heat_exact(x_eval, t_tensor, alpha=alpha_true, L=L)
            u_pinn  = model(x_eval, t_tensor)

        axes[i].plot(
            x_np, u_exact.numpy().flatten(),
            label="Analítica", color="blue", linewidth=2, alpha=0.7
        )
        axes[i].plot(
            x_np, u_pinn.numpy().flatten(),
            label="PINN", linestyle="--", color="black", linewidth=2
        )
        axes[i].set_title(f"t = {t_val:.3f}")
        axes[i].set_xlabel("x")
        axes[i].set_ylim(-0.1, 1.1)
        axes[i].legend(fontsize=8)
        axes[i].grid(True)

    axes[0].set_ylabel("u(x,t)  Temperatura")
    alpha_pred = model.alpha.item()
    error_alpha = abs(alpha_true - alpha_pred) / alpha_true * 100
    fig.suptitle(
        f"Calor Inverso — Época {epoch} | Pérdida: {loss:.4e}\n"
        f"α real: {alpha_true:.4f} | α pred: {alpha_pred:.4f} "
        f"| Error: {error_alpha:.2f}%",
        fontsize=11
    )
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:05d}.png"), dpi=150)
    plt.close()


def main(
    L: float = 1.0,
    t_max: float = 1.0,
    alpha_true: float = 0.1,      # Valor real que la red debe descubrir
    alpha_init: float = 0.5,      # Inicialización deliberadamente lejos
    noise_std: float = 0.0,       # Ruido en las observaciones
    epochs: int = 15000,
    lr: float = 1e-3,
    num_collocation: int = 3000,
    num_ic_points: int = 300,
    num_bc_points: int = 200,
    num_data_points: int = 100,   # Observaciones empíricas para anclar alpha
    sampler: str = "lhs",
    log_freq: int = 2000,
    warmup_epochs: int = 2000,
):
    set_seed(42)
    os.makedirs("img/heat_inverse", exist_ok=True)
    os.makedirs("results/heat_inverse", exist_ok=True)

    config_exp = {
        "sistema":          "heat_inverse",
        "sampler":          sampler,
        "estado_n":         0,
        "L":                L,
        "t_max":            t_max,
        "alpha_true":       alpha_true,
        "alpha_init":       alpha_init,
        "noise_std":        noise_std,
        "epochs":           epochs,
        "lr":               lr,
        "num_collocation":  num_collocation,
        "num_ic_points":    num_ic_points,
        "num_bc_points":    num_bc_points,
        "num_data_points":  num_data_points,
        "warmup_epochs":    warmup_epochs,
    }

    print("--- Iniciando Entrenamiento (Ecuación del Calor — Problema Inverso) ---")
    print(f"Buscando α={alpha_true} | Inicialización: α={alpha_init} | Ruido: {noise_std}\n")

    # 1. Modelo y optimizador
    model = PINNHeatInverse(hidden_layers=[32, 32, 32], alpha_init=alpha_init)
    optimizer = optim.Adam([
        {'params': model.net.parameters(),  'lr': lr},
        {'params': [model.alpha],           'lr': lr * 5},  # alpha converge más rápido con lr mayor
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5000, gamma=0.5
    )

    # 2. Puntos fijos de condición inicial (t=0)
    x_ic = torch.linspace(0.0, L, num_ic_points).unsqueeze(1)
    t_ic = torch.zeros(num_ic_points, 1)

    # 3. Puntos fijos de frontera
    t_bc = torch.rand(num_bc_points, 1) * t_max

    # 4. Datos empíricos con solución analítica (simulan mediciones reales)
    #    Distribuidos en todo el dominio espacio-temporal
    np.random.seed(42)
    x_data_np = np.random.uniform(0.0, L,     num_data_points)
    t_data_np = np.random.uniform(0.0, t_max, num_data_points)
    x_data = torch.tensor(x_data_np, dtype=torch.float32).unsqueeze(1)
    t_data = torch.tensor(t_data_np, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        u_data_clean = heat_exact(x_data, t_data, alpha=alpha_true, L=L)

    # Ruido gaussiano opcional
    if noise_std > 0.0:
        noise  = torch.randn_like(u_data_clean) * noise_std
        u_data = u_data_clean + noise
    else:
        u_data = u_data_clean

    # 5. Puntos de evaluación y visualización
    t_snap_vals = [0.0, t_max * 0.25, t_max * 0.5, t_max]
    x_eval      = torch.linspace(0.0, L, 300).unsqueeze(1)

    historial = {
        "epoch":      [],
        "total_loss": [],
        "data_loss":  [],
        "ph_loss":    [],
        "ic_loss":    [],
        "bc_loss":    [],
        "alpha_pred": [],
    }

    # 6. Bucle de entrenamiento
    # Entrenamiento PINN medido con Timer
    with Timer() as t_pinn:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # Remuestreo cada 10 épocas para equilibrar coste y exploración
            if epoch % 10 == 1:
                x_col, t_col = sample_collocation_2d(
                    0.0, L, 0.0, t_max, num_collocation, sampler=sampler
                )

            # A) Pérdida de datos — ancla alpha al valor correcto
            u_pred_data = model(x_data, t_data)
            data_loss   = torch.mean((u_pred_data - u_data) ** 2)

            # B) Condición inicial
            ic_loss = initial_condition_loss_heat(model, x_ic, t_ic, L=L)

            # C) Condición de frontera
            bc_loss = boundary_loss_heat(model, t_bc, x_min=0.0, x_max=L)

            # D) Residuo físico con alpha entrenable — curriculum
            ph_loss = physics_loss_heat_inverse(model, x_col, t_col)

            if epoch < warmup_epochs:
                ph_ramp = 0.0
            else:
                ph_ramp = min(1.0, (epoch - warmup_epochs) / 1000.0)

            # E) Pérdida total
            total_loss = data_loss + ic_loss + bc_loss + ph_ramp * ph_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Clipping para que alpha no diverja
            with torch.no_grad():
                model.alpha.clamp_(1e-4, 10.0)

            # 7. Monitorización
            if epoch % log_freq == 0 or epoch == epochs:
                alpha_pred  = model.alpha.item()
                error_alpha = abs(alpha_true - alpha_pred) / alpha_true * 100

                print(
                    f"Época {epoch:05d} | Total: {total_loss.item():.4e} "
                    f"| Datos: {data_loss.item():.4e} "
                    f"| Física: {ph_loss.item():.4e} (×{ph_ramp:.2f})"
                )
                print(
                    f"            | α real: {alpha_true:.4f} "
                    f"-> α pred: {alpha_pred:.4f} "
                    f"| Error: {error_alpha:.2f}%"
                )

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())
                historial["ph_loss"].append(ph_loss.item())
                historial["ic_loss"].append(ic_loss.item())
                historial["bc_loss"].append(bc_loss.item())
                historial["alpha_pred"].append(alpha_pred)

                plot_heat_inverse_results(
                    model, x_eval, t_snap_vals,
                    alpha_true=alpha_true, L=L,
                    epoch=epoch, loss=total_loss.item(),
                    save_dir="img/heat_inverse",
                )
                

    # 8. Evaluación final
    import numpy as np
    x_np   = np.linspace(0.0, L, 300)
    t_np   = np.linspace(0.0, t_max, 50)
    ref_fourier = measure_numerical_reference(
        sistema="heat_inverse",
        x_or_t=x_np,
        t_array=t_np,
        alpha=alpha_true, L=L,
    )

    errors = []
    for t_val in t_snap_vals:
        t_tensor = torch.full((len(x_eval), 1), t_val)
        with torch.no_grad():
            u_pinn  = model(x_eval, t_tensor)
            u_exact = heat_exact(x_eval, t_tensor, alpha=alpha_true, L=L)
        err = calculate_l2_error(u_pinn, u_exact)
        errors.append(err)
        print(f"  t={t_val:.3f} | Error L2: {err:.4e}")

    alpha_final = model.alpha.item()
    print(f"\nDifusividad térmica (α): Real = {alpha_true:.4f} | Predicha = {alpha_final:.4f}")
    print(f"Error relativo α:        {abs(alpha_true - alpha_final)/alpha_true*100:.2f}%")
    print(f"Tiempo PINN:             {t_pinn.elapsed:.2f}s")
    print(f"Tiempo Serie Fourier:    {ref_fourier['time_s']:.4f}s")
    print(f"Speedup (Fourier/PINN):  {ref_fourier['time_s']/t_pinn.elapsed:.4f}x\n")

    final_results = {
        "error_L2_t0":           errors[0],
        "error_L2_t025":         errors[1],
        "error_L2_t050":         errors[2],
        "error_L2_tmax":         errors[3],
        "error_L2_mean":         float(np.mean(errors)),
        "alpha_final":           alpha_final,
        "error_relativo_alpha":  abs(alpha_true - alpha_final) / alpha_true,
        "pinn_time_s":           t_pinn.elapsed,
        "numerical_time_s":      ref_fourier["time_s"],
        "numerical_method":      ref_fourier["method"],
        "speedup":               ref_fourier["time_s"] / t_pinn.elapsed,
    }

    save_experiment_results(config_exp, final_results, historial)


if __name__ == "__main__":
    main(
        alpha_true=0.1,
        alpha_init=0.5,
        epochs=6000,
        sampler="lhs",
        log_freq=2000,
        warmup_epochs=1000,
        noise_std=0.2,
    )