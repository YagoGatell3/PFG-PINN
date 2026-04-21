import os
import numpy as np
import torch
import torch.optim as optim

from src.models import PINNTunnel
from src.loss_functions import (
    physics_loss_tunnel,
    initial_condition_loss_tunnel,
    boundary_loss_tunnel,
    normalization_loss_tunnel,
)
from src.exact_solutions import psi_tunnel_initial
from src.numerical_methods import solve_tunnel_crank_nicolson
from src.samplers import generate_lhs_points, generate_grid_points
from src.utils import set_seed, calculate_l2_error, save_experiment_results, Timer, measure_numerical_reference


def sample_collocation_2d(
    x_min, x_max, t_min, t_max, n_points
) -> tuple[torch.Tensor, torch.Tensor]:
    """Muestrea puntos (x, t) con LHS en el dominio 2D."""
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2)
    pts = sampler.random(n=n_points)
    pts[:, 0] = x_min + pts[:, 0] * (x_max - x_min)
    pts[:, 1] = t_min + pts[:, 1] * (t_max - t_min)
    x = torch.tensor(pts[:, 0:1], dtype=torch.float32, requires_grad=True)
    t = torch.tensor(pts[:, 1:2], dtype=torch.float32, requires_grad=True)
    return x, t


def plot_tunnel_results(model, x_eval, t_snapshots, prob_cn, epoch, loss, save_dir):
    """
    Compara densidad de probabilidad PINN vs Crank-Nicolson
    en varios instantes temporales.
    """
    import matplotlib.pyplot as plt

    n_snaps = len(t_snapshots)
    fig, axes = plt.subplots(n_snaps, 1, figsize=(10, 3 * n_snaps))
    if n_snaps == 1:
        axes = [axes]

    x_np = x_eval.detach().numpy().flatten()

    for i, t_val in enumerate(t_snapshots):
        t_tensor = torch.full((len(x_eval), 1), t_val)
        with torch.no_grad():
            u, v = model(x_eval, t_tensor)
        prob_pinn = (u**2 + v**2).numpy().flatten()

        axes[i].plot(x_np, prob_cn[i], label="Crank-Nicolson", color="blue", alpha=0.7)
        axes[i].plot(x_np, prob_pinn, label="PINN", linestyle="--", color="black")
        axes[i].axvspan(0.5, 1.5, alpha=0.15, color="red", label="Barrera V₀")
        axes[i].set_title(f"t = {t_val:.2f}")
        axes[i].set_ylabel("|Ψ|²")
        axes[i].legend(fontsize=8)
        axes[i].set_ylim(0, None)

    axes[-1].set_xlabel("x")
    fig.suptitle(f"Efecto Túnel — Época {epoch} | Pérdida: {loss:.4e}", fontsize=12)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:05d}.png"), dpi=150)
    plt.close()


def main(
    x_min: float = -10.0,
    x_max: float = 10.0,
    t_max: float = 3.0,
    V0: float = 3,
    x_barrier_left: float = 0.5,
    x_barrier_right: float = 1.5,
    x0: float = -4.0,
    sigma: float = 0.75,
    k0: float = 2.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    epochs: int = 20000,
    lr: float = 1e-3,
    num_collocation: int = 5000,
    num_ic_points: int = 500,
    num_bc_points: int = 200,
    log_freq: int = 2000,
    warmup_epochs: int = 3000,
):
    set_seed(42)
    os.makedirs("img/tunnel", exist_ok=True)
    os.makedirs("results/tunnel", exist_ok=True)

    print("--- Iniciando Entrenamiento (Efecto Túnel Cuántico) ---")
    print(f"Barrera: V₀={V0} en x∈[{x_barrier_left}, {x_barrier_right}] | k₀={k0} | E≈{k0**2/2:.2f}")
    print(f"Condición túnel: E < V₀ → {k0**2/2:.2f} < {V0} → {'✓ SÍ hay túnel' if k0**2/2 < V0 else '✗ NO hay túnel'}\n")

    # 1. Modelo y optimizador
    model = PINNTunnel(hidden_layers=[64, 64, 64, 64])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    # 2. Puntos de condición inicial (t=0)
    x_ic = generate_grid_points(x_min, x_max, num_ic_points, requires_grad=False)
    t_ic = torch.zeros(num_ic_points, 1)

    # 3. Puntos de frontera temporal (muestreados aleatoriamente en t)
    t_bc = torch.rand(num_bc_points, 1) * t_max

    # 4. Ground truth numérico (Crank-Nicolson)
    x_cn = np.linspace(x_min, x_max, 300)
    t_cn = np.linspace(0.0, t_max, 100)
    print("Calculando solución de referencia (Crank-Nicolson)...")
    ref_cn = measure_numerical_reference(
        sistema="tunnel",
        x_or_t=x_cn,
        t_array=t_cn,
        x0=x0, sigma=sigma, k0=k0,
        V0=V0, x_barrier_left=x_barrier_left,
        x_barrier_right=x_barrier_right,
        mass=mass, hbar=hbar,
    )
    prob_cn_full = ref_cn["solution"]
    print("Referencia calculada.\n")

    # Instantes para visualizar
    t_snap_vals = [0.0, t_max * 0.33, t_max * 0.66, t_max]
    t_snap_idx  = [np.argmin(np.abs(t_cn - tv)) for tv in t_snap_vals]
    prob_cn_snaps = [prob_cn_full[i] for i in t_snap_idx]
    x_eval = torch.tensor(x_cn, dtype=torch.float32).unsqueeze(1)

    historial = {
        "epoch": [], "total_loss": [],
        "ph_loss": [], "ic_loss": [], "bc_loss": [], "norm_loss": []
    }

    # 5. Bucle de entrenamiento
    with Timer as t_pinn:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # Remuestreo de puntos de colocación cada época (mejora la exploración)
            x_col, t_col = sample_collocation_2d(x_min, x_max, 0.0, t_max, num_collocation)

            # A) Condición inicial
            ic_loss = initial_condition_loss_tunnel(
                model, x_ic, t_ic, x0=x0, sigma=sigma, k0=k0
            )

            # B) Condición de frontera
            bc_loss = boundary_loss_tunnel(model, t_bc, x_min=x_min, x_max=x_max)

            # C) Residuo físico (con curriculum)
            ph_loss = physics_loss_tunnel(
                model, x_col, t_col,
                V0=V0, x_barrier_left=x_barrier_left,
                x_barrier_right=x_barrier_right,
                mass=mass, hbar=hbar
            )

            # D) Normalización
            norm_loss = normalization_loss_tunnel(
                model, x_ic, t_ic, domain_length=(x_max - x_min)
            )

            # E) Curriculum: la física entra gradualmente
            if epoch < warmup_epochs:
                ph_ramp = 0.0
            else:
                ph_ramp = min(1.0, (epoch - warmup_epochs) / 2000.0)

            total_loss = (
                ic_loss
                + bc_loss
                + ph_ramp * ph_loss
                + 0.1 * norm_loss
            )

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % log_freq == 0 or epoch == epochs:
                print(
                    f"Época {epoch:05d} | Total: {total_loss.item():.4e} "
                    f"| IC: {ic_loss.item():.4e} | BC: {bc_loss.item():.4e} "
                    f"| Física: {ph_loss.item():.4e} (×{ph_ramp:.2f}) "
                    f"| Norma: {norm_loss.item():.4e}"
                )

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["ph_loss"].append(ph_loss.item())
                historial["ic_loss"].append(ic_loss.item())
                historial["bc_loss"].append(bc_loss.item())
                historial["norm_loss"].append(norm_loss.item())

                plot_tunnel_results(
                    model, x_eval, t_snap_vals,
                    prob_cn_snaps, epoch, total_loss.item(),
                    save_dir="img/tunnel"
                )

    # 6. Evaluación final
    t_final = torch.full((len(x_eval), 1), t_max)
    with torch.no_grad():
        u_f, v_f = model(x_eval, t_final)
    prob_pinn_final = (u_f**2 + v_f**2).numpy().flatten()
    prob_cn_final   = prob_cn_full[-1]

    prob_pinn_t = torch.tensor(prob_pinn_final).unsqueeze(1)
    prob_cn_t   = torch.tensor(prob_cn_final).unsqueeze(1)
    error_l2    = calculate_l2_error(prob_pinn_t, prob_cn_t)

    dx     = x_cn[1] - x_cn[0]
    T_cn   = np.sum(prob_cn_final[x_cn > x_barrier_right]) * dx
    T_pinn = np.sum(prob_pinn_final[x_cn > x_barrier_right]) * dx

    print(f"\n--- RESULTADOS FINALES (Efecto Túnel) ---")
    print(f"Error L2 en |Ψ|²:       {error_l2:.4e}")
    print(f"Coef. transmisión CN:   {T_cn:.4f}")
    print(f"Coef. transmisión PINN: {T_pinn:.4f}")
    print(f"Tiempo PINN:            {t_pinn.elapsed:.2f}s")
    print(f"Tiempo Crank-Nicolson:  {ref_cn['time_s']:.4f}s")
    print(f"Speedup (CN/PINN):      {ref_cn['time_s']/t_pinn.elapsed:.4f}x\n")

    final_results = {
        "error_L2":          error_l2,
        "T_cn":              float(T_cn),
        "T_pinn":            float(T_pinn),
        "pinn_time_s":       t_pinn.elapsed,
        "numerical_time_s":  ref_cn["time_s"],
        "numerical_method":  ref_cn["method"],
        "speedup":           ref_cn["time_s"] / t_pinn.elapsed,
    }

    save_experiment_results(
        {
            "sistema":   "tunnel",
            "sampler":   "lhs",
            "estado_n":  0,
            "V0":        V0,
            "k0":        k0,
            "epochs":    epochs,
        },
        final_results,
        historial,
        save_dir="results"
    )


if __name__ == "__main__":
    main(
        epochs=20000,
        log_freq=2000,
        num_collocation=5000,
        warmup_epochs=3000,
    )