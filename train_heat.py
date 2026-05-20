import os

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import qmc

from src.exact_solutions import heat_exact
from src.loss_functions import (
    boundary_loss_heat,
    initial_condition_loss_heat,
    physics_loss_heat_inverse,
)
from src.models import PINNHeatInverse
from src.utils import (
    Timer,
    calculate_l2_error,
    measure_numerical_reference,
    save_experiment_results,
    set_seed,
    update_dynamic_weights,
)


# ── Sampler 2D ────────────────────────────────────────────────────────────────

def sample_collocation_2d(
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    n_points: int,
    sampler: str = "lhs",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Genera puntos de colocación (x, t) dentro del dominio espaciotemporal 2D.

    Args:
        x_min, x_max: Límites espaciales del dominio.
        t_min, t_max: Límites temporales del dominio.
        n_points: Número total de puntos a generar.
        sampler: Estrategia de muestreo ('lhs' para Latin Hypercube, 'grid' para malla regular).

    Returns:
        Tupla de tensores (x, t) con requires_grad=True para el cálculo de derivadas físicas.
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


# ── Visualizaciones ───────────────────────────────────────────────────────────

def _plot_heatmaps(
    model: torch.nn.Module,
    alpha_true: float,
    L: float,
    t_max: float,
    epoch: int,
    label: str,
    x_train: torch.Tensor = None,
    t_train: torch.Tensor = None,
):
    """
    Genera y guarda una comparativa 2D (Heatmaps) mostrando:
    1. La solución analítica exacta.
    2. La predicción del modelo espacial.
    3. El mapa de error absoluto entre ambos.
    """
    import matplotlib.pyplot as plt

    nx, nt = 200, 200
    x_np = np.linspace(0.0, L,     nx)
    t_np = np.linspace(0.0, t_max, nt)
    XX, TT = np.meshgrid(x_np, t_np)

    x_flat = torch.tensor(XX.ravel(), dtype=torch.float32).unsqueeze(1)
    t_flat = torch.tensor(TT.ravel(), dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        u_exact_flat = heat_exact(x_flat, t_flat, alpha=alpha_true, L=L)
        u_pred_flat  = model(x_flat, t_flat)

    U_exact = u_exact_flat.numpy().reshape(nt, nx)
    U_pred  = u_pred_flat.numpy().reshape(nt, nx)
    U_err   = np.abs(U_exact - U_pred)

    vmin = min(U_exact.min(), U_pred.min())
    vmax = max(U_exact.max(), U_pred.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Analítica
    im0 = axes[0].pcolormesh(x_np, t_np, U_exact, cmap="viridis",
                             vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(im0, ax=axes[0], label="u(x,t)")
    axes[0].set_title("Analítica", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    
    # Puntos de entrenamiento superpuestos (si se proporcionan)
    if x_train is not None and t_train is not None:
        axes[0].scatter(
            x_train.detach().numpy().flatten(),
            t_train.detach().numpy().flatten(),
            c="red", s=20, zorder=5, label="Train", alpha=0.9,
            edgecolors="white", linewidths=0.5,
        )
        axes[0].legend(fontsize=9, loc="upper right")

    # Predicción
    im1 = axes[1].pcolormesh(x_np, t_np, U_pred, cmap="viridis",
                             vmin=vmin, vmax=vmax, shading="auto")
    fig.colorbar(im1, ax=axes[1], label="u(x,t)")
    axes[1].set_title(label, fontsize=13, fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")

    # Error absoluto
    im2 = axes[2].pcolormesh(x_np, t_np, U_err, cmap="hot", shading="auto")
    fig.colorbar(im2, ax=axes[2], label="|u_exact − u_pred|")
    axes[2].set_title("Error absoluto", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")

    plt.tight_layout()

    save_dir = f"img/heat_inverse/{label.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"heatmaps_epoch_{epoch:05d}.png"), dpi=150)
    plt.close()


def _plot_surface_3d(
    model: torch.nn.Module,
    alpha_true: float,
    L: float,
    t_max: float,
    epoch: int,
    label: str,
):
    """
    Genera y guarda una proyección 3D superponiendo la solución analítica
    y la predicción de la red en todo el dominio.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.patches import Patch

    nx, nt = 80, 80
    x_np = np.linspace(0.0, L,     nx)
    t_np = np.linspace(0.0, t_max, nt)
    XX, TT = np.meshgrid(x_np, t_np)

    x_flat = torch.tensor(XX.ravel(), dtype=torch.float32).unsqueeze(1)
    t_flat = torch.tensor(TT.ravel(), dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        u_exact_flat = heat_exact(x_flat, t_flat, alpha=alpha_true, L=L)
        u_pred_flat  = model(x_flat, t_flat)

    U_exact = u_exact_flat.numpy().reshape(nt, nx)
    U_pred  = u_pred_flat.numpy().reshape(nt, nx)

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    ax.plot_surface(XX, TT, U_exact, alpha=0.35, color="#1f77b4",
                    edgecolor="none", antialiased=True)
    ax.plot_surface(XX, TT, U_pred,  alpha=0.35, color="#ff7f0e",
                    edgecolor="none", antialiased=True)

    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x,t)")
    ax.view_init(elev=25, azim=-135)

    legend_elements = [
        Patch(facecolor="#1f77b4", alpha=0.35, label="Analítica"),
        Patch(facecolor="#ff7f0e", alpha=0.35, label=label),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    save_dir = f"img/heat_inverse/{label.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"surface3d_epoch_{epoch:05d}.png"), dpi=150)
    plt.close()


def _plot_alpha_convergence(
    historial: dict,
    alpha_true: float,
    label: str,
):
    """
    Grafica la evolución del parámetro físico alpha (difusividad térmica)
    descubierto por el modelo a lo largo de las épocas de entrenamiento.
    """
    import matplotlib.pyplot as plt

    epochs_hist = historial["epoch"]
    alpha_hist  = historial["alpha_pred"]

    if len(epochs_hist) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs_hist, alpha_hist, color="black", linewidth=2, label="α predicho")
    ax.axhline(alpha_true, color="blue", linewidth=1.5,
               linestyle="--", label=f"α real = {alpha_true}")
    ax.set_xlabel("Época")
    ax.set_ylabel("α (Difusividad térmica)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    save_dir = f"img/heat_inverse/{label.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "alpha_convergence.png"), dpi=150)
    plt.close()


# ── Evaluación ────────────────────────────────────────────────────────────────

def _eval_snapshots(
    model: torch.nn.Module,
    t_snap_vals: list,
    alpha_true: float,
    L: float,
    n_eval: int = 300,
) -> list:
    """
    Calcula el error L2 discretizado espacialmente evaluando el modelo 
    en momentos temporales concretos (snapshots).
    """
    x_eval = torch.linspace(0.0, L, n_eval).unsqueeze(1)
    errors = []
    for t_val in t_snap_vals:
        t_tensor = torch.full((n_eval, 1), t_val)
        with torch.no_grad():
            u_pred  = model(x_eval, t_tensor)
            u_exact = heat_exact(x_eval, t_tensor, alpha=alpha_true, L=L)
        errors.append(calculate_l2_error(u_pred, u_exact))
    return errors


# ── Entrenamiento PINN ────────────────────────────────────────────────────────

def _train_pinn(
    x_train: torch.Tensor,
    t_train: torch.Tensor,
    u_train: torch.Tensor,
    x_ic: torch.Tensor,
    t_ic: torch.Tensor,
    t_bc: torch.Tensor,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    alpha_true: float,
    L: float,
    t_max: float,
    t_snap_vals: list,
    epochs: int,
    lr: float,
    alpha_init: float,
    warmup_epochs: int,
    use_dynamic_weights: bool,
    optimizer_name: str,
    hidden_layers: list,
    log_freq: int,
    seed: int,
    save_plots: bool = True,
) -> tuple[float, float, dict]:
    """
    Entrena la red neuronal informada por la física (PINN) para el problema inverso
    de la ecuación del calor. Descubre la difusividad (alpha) simultáneamente
    al ajuste del campo de temperaturas a partir de datos dispersos.
    """
    set_seed(seed)
    model = PINNHeatInverse(hidden_layers=hidden_layers, alpha_init=alpha_init)
    label = "PINN"

    historial = {
        "epoch":      [],
        "total_loss": [],
        "data_loss":  [],
        "ph_loss":    [],
        "ic_loss":    [],
        "bc_loss":    [],
        "alpha_pred": [],
        "lambda_ph":  [],
    }

    # Configuración del optimizador: alpha utiliza un factor multiplicador en su LR
    if optimizer_name == "adam":
        optimizer = optim.Adam([
            {"params": model.net.parameters(), "lr": lr},
            {"params": [model.alpha],          "lr": lr * 5},
        ])
    elif optimizer_name == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=50)
    elif optimizer_name == "adam+lbfgs":
        optimizer = optim.Adam([
            {"params": model.net.parameters(), "lr": lr},
            {"params": [model.alpha],          "lr": lr * 5},
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

            # Rampa de calentamiento (warmup) para introducir la física progresivamente
            if epoch < warmup_epochs:
                ph_ramp = 0.0
            else:
                ph_ramp = min(1.0, (epoch - warmup_epochs) / 1000.0)

            def closure(ramp=ph_ramp, lph=lambda_ph):
                optimizer.zero_grad()
                data_loss = torch.mean((model(x_train, t_train) - u_train) ** 2)
                ic_loss   = initial_condition_loss_heat(model, x_ic, t_ic, L=L)
                bc_loss   = boundary_loss_heat(model, t_bc, x_min=0.0, x_max=L)
                ph_loss   = physics_loss_heat_inverse(model, x_col, t_col)
                total = data_loss + ic_loss + bc_loss + ramp * lph * ph_loss
                total.backward()
                return total

            if optimizer_name == "lbfgs" or (
                optimizer_name == "adam+lbfgs" and epoch >= int(epochs * 0.85) + 1
            ):
                result     = optimizer.step(closure)
                total_loss = result if result is not None else torch.tensor(0.0)
                with torch.no_grad():
                    data_loss = torch.mean((model(x_train, t_train) - u_train) ** 2)
                ic_loss = torch.tensor(0.0)
                bc_loss = torch.tensor(0.0)
                ph_loss = torch.tensor(0.0)
            else:
                optimizer.zero_grad()

                data_loss = torch.mean((model(x_train, t_train) - u_train) ** 2)
                ic_loss   = initial_condition_loss_heat(model, x_ic, t_ic, L=L)
                bc_loss   = boundary_loss_heat(model, t_bc, x_min=0.0, x_max=L)
                ph_loss   = physics_loss_heat_inverse(model, x_col, t_col)

                if use_dynamic_weights:
                    lambda_ph, _ = update_dynamic_weights(
                        data_loss, ph_loss, bc_loss,
                        model.net[-1].weight,
                        lambda_ph, 1.0,
                    )
                else:
                    lambda_ph = 1.0

                total_loss = data_loss + ic_loss + bc_loss + ph_ramp * lambda_ph * ph_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Aseguramos estabilidad física del parámetro
            with torch.no_grad():
                model.alpha.clamp_(1e-4, 10.0)

            if epoch % log_freq == 0 or epoch == epochs:
                alpha_pred  = model.alpha.item()
                error_alpha = abs(alpha_true - alpha_pred) / alpha_true * 100
                print(
                    f"[{label}] Época {epoch:05d} | Total: {total_loss.item():.4e} "
                    f"| Datos: {data_loss.item():.4e} "
                    f"| Física: {ph_loss.item():.4e} (×{ph_ramp:.2f})"
                )
                print(
                    f"                   | α real: {alpha_true:.4f} "
                    f"-> α pred: {alpha_pred:.4f} "
                    f"| Error: {error_alpha:.2f}%"
                )
                if use_dynamic_weights:
                    print(f"                   | λ_ph: {lambda_ph:.4f}")

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())
                historial["ph_loss"].append(ph_loss.item())
                historial["ic_loss"].append(ic_loss.item())
                historial["bc_loss"].append(bc_loss.item())
                historial["alpha_pred"].append(alpha_pred)
                historial["lambda_ph"].append(lambda_ph)

                if save_plots:
                    _plot_heatmaps(
                        model, alpha_true=alpha_true, L=L, t_max=t_max,
                        epoch=epoch, label=label,
                        x_train=x_train, t_train=t_train,
                    )
                    _plot_surface_3d(
                        model, alpha_true=alpha_true, L=L, t_max=t_max,
                        epoch=epoch, label=label,
                    )
                    _plot_alpha_convergence(
                        historial, alpha_true=alpha_true, label=label,
                    )

    errors        = _eval_snapshots(model, t_snap_vals, alpha_true, L)
    error_l2_mean = float(np.mean(errors))

    return error_l2_mean, timer.elapsed, historial


# ── Entrenamiento NN pura ─────────────────────────────────────────────────────

def _train_nn(
    x_train: torch.Tensor,
    t_train: torch.Tensor,
    u_train: torch.Tensor,
    alpha_true: float,
    L: float,
    t_max: float,
    t_snap_vals: list,
    epochs: int,
    lr: float,
    alpha_init: float,
    optimizer_name: str,
    hidden_layers: list,
    log_freq: int,
    seed: int,
    save_plots: bool = True,
) -> tuple[float, float, dict]:
    """
    Entrena un modelo puramente de datos (NN estándar sin restricciones físicas)
    para establecer un marco de comparación (baseline).
    """
    set_seed(seed)
    model = PINNHeatInverse(hidden_layers=hidden_layers, alpha_init=alpha_init)
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
                data_loss = torch.mean((model(x_train, t_train) - u_train) ** 2)
                data_loss.backward()
                return data_loss

            if optimizer_name == "lbfgs" or (
                optimizer_name == "adam+lbfgs" and epoch >= int(epochs * 0.85) + 1
            ):
                result     = optimizer.step(closure)
                total_loss = result if result is not None else torch.tensor(0.0)
                with torch.no_grad():
                    data_loss = torch.mean((model(x_train, t_train) - u_train) ** 2)
            else:
                optimizer.zero_grad()
                data_loss  = torch.mean((model(x_train, t_train) - u_train) ** 2)
                total_loss = data_loss
                total_loss.backward()
                optimizer.step()

            if epoch % log_freq == 0 or epoch == epochs:
                print(f"[{label}] Época {epoch:05d} | Pérdida: {total_loss.item():.4e}")

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())

                if save_plots:
                    _plot_heatmaps(
                        model, alpha_true=alpha_true, L=L, t_max=t_max,
                        epoch=epoch, label=label,
                        x_train=x_train, t_train=t_train,
                    )
                    _plot_surface_3d(
                        model, alpha_true=alpha_true, L=L, t_max=t_max,
                        epoch=epoch, label=label,
                    )

    errors        = _eval_snapshots(model, t_snap_vals, alpha_true, L)
    error_l2_mean = float(np.mean(errors))

    return error_l2_mean, timer.elapsed, historial


# ── Controlador Principal ─────────────────────────────────────────────────────

def main(
    L: float = 1.0,
    t_max: float = 1.0,
    alpha_true: float = 0.1,
    alpha_init: float = 0.5,
    noise_std: float = 0.0,
    epochs: int = 15000,
    lr: float = 1e-3,
    num_collocation: int = 3000,
    num_train_points: int = 20,
    num_ic_points: int = 100,
    num_bc_points: int = 100,
    sampler: str = "lhs",
    log_freq: int = 2000,
    warmup_epochs: int = 2000,
    use_dynamic_weights: bool = False,
    optimizer_name: str = "adam",
    hidden_layers: list = None,
    seed: int = 42,
    save_plots: bool = True,
):
    """
    Función orquestadora del experimento. Controla la inicialización del 
    problema físico, la generación de puntos de muestreo (con/sin ruido), 
    la ejecución del entrenamiento y la evaluación frente a solucionadores 
    tradicionales (Crank-Nicolson).
    """
    if hidden_layers is None:
        hidden_layers = [32, 32, 32]

    set_seed(seed)
    os.makedirs("img/heat_inverse", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    config_exp = {
        "sistema":             "heat_inverse",
        "L":                   L,
        "t_max":               t_max,
        "alpha_true":          alpha_true,
        "alpha_init":          alpha_init,
        "noise_std":           noise_std,
        "epochs":              epochs,
        "lr":                  lr,
        "num_collocation":     num_collocation,
        "num_train_points":    num_train_points,
        "num_ic_points":       num_ic_points,
        "num_bc_points":       num_bc_points,
        "sampler":             sampler,
        "warmup_epochs":       warmup_epochs,
        "use_dynamic_weights": use_dynamic_weights,
        "optimizer":           optimizer_name,
        "hidden_layers":       hidden_layers,
        "seed":                seed,
    }

    print("=" * 60)
    print("Ecuación del Calor — Problema Inverso")
    print(
        f"Sampler: {sampler} | Colocación: {num_collocation} | "
        f"Train: {num_train_points} | IC: {num_ic_points} | BC: {num_bc_points} | "
        f"Épocas: {epochs} | Optimizador: {optimizer_name} | "
        f"lr: {lr} | Warmup: {warmup_epochs} | "
        f"Pesos din.: {use_dynamic_weights} | "
        f"Ruido: {noise_std} | Seed: {seed}"
    )
    print(f"Buscando α={alpha_true} | Inicialización: α={alpha_init}")
    print("=" * 60)

    # 1. Muestreo de datos de entrenamiento aleatorios en el dominio
    np.random.seed(seed)
    x_train_np = np.random.uniform(0.0, L,     num_train_points)
    t_train_np = np.random.uniform(0.0, t_max, num_train_points)
    x_train = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1)
    t_train = torch.tensor(t_train_np, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        u_train_clean = heat_exact(x_train, t_train, alpha=alpha_true, L=L)

    # Introducción de ruido gaussiano
    if noise_std > 0.0:
        torch.manual_seed(seed)
        u_train = u_train_clean + torch.randn_like(u_train_clean) * noise_std
    else:
        u_train = u_train_clean

    # 2. Generación de las fronteras: Condición inicial (t=0)
    x_ic = torch.linspace(0.0, L, num_ic_points).unsqueeze(1)
    t_ic = torch.zeros(num_ic_points, 1)

    # 3. Generación de las fronteras: Condiciones de contorno espaciales (x=0, x=L)
    torch.manual_seed(seed)
    t_bc = torch.rand(num_bc_points, 1) * t_max

    # 4. Muestreo del dominio estructural para la evaluación de residuos (física)
    x_col, t_col = sample_collocation_2d(
        0.0, L, 0.0, t_max, num_collocation, sampler=sampler
    )

    # 5. Segmentos temporales para evaluación de errores
    t_snap_vals = [0.0, t_max * 0.25, t_max * 0.5, t_max]

    shared = dict(
        x_train=x_train, t_train=t_train, u_train=u_train,
        alpha_true=alpha_true, L=L, t_max=t_max,
        t_snap_vals=t_snap_vals,
        epochs=epochs, lr=lr, alpha_init=alpha_init,
        optimizer_name=optimizer_name, hidden_layers=hidden_layers,
        log_freq=log_freq, seed=seed, save_plots=save_plots,
    )

    # 6. Lanzamiento de la red neuronal física (PINN)
    print("\n--- PINN (problema inverso) ---")
    error_pinn, time_pinn, hist_pinn = _train_pinn(
        x_ic=x_ic, t_ic=t_ic, t_bc=t_bc,
        x_col=x_col, t_col=t_col,
        warmup_epochs=warmup_epochs,
        use_dynamic_weights=use_dynamic_weights,
        **shared,
    )

    # 7. Lanzamiento del baseline sin regularización física (NN)
    print("\n--- NN pura (sin física, mismos datos) ---")
    error_nn, time_nn, hist_nn = _train_nn(**shared)

    # 8. Resolución estándar por el método de Crank-Nicolson
    print("\n--- Crank-Nicolson (referencia numérica) ---")
    x_np = np.linspace(0.0, L, 300)
    t_np = np.linspace(0.0, t_max, 100)
    ref_cn = measure_numerical_reference(
        sistema="heat_inverse",
        x_or_t=x_np,
        t_array=t_np,
        alpha=alpha_true, L=L,
    )
    time_cn = ref_cn["time_s"]

    sol_cn    = ref_cn["solution"]
    cn_errors = []
    for t_val in t_snap_vals:
        idx_t     = int(np.argmin(np.abs(t_np - t_val)))
        u_cn_at_t = torch.tensor(sol_cn[idx_t], dtype=torch.float32).unsqueeze(1)
        x_cn      = torch.tensor(x_np,          dtype=torch.float32).unsqueeze(1)
        t_cn      = torch.full_like(x_cn, t_val)
        with torch.no_grad():
            u_exact = heat_exact(x_cn, t_cn, alpha=alpha_true, L=L)
        cn_errors.append(calculate_l2_error(u_cn_at_t, u_exact))
    error_cn = float(np.mean(cn_errors))

    # 9. Recopilación de métricas e impresiones finales
    alpha_final = hist_pinn["alpha_pred"][-1] if hist_pinn["alpha_pred"] else float("nan")

    final_results = {
        "pinn": {
            "error_L2_mean":        error_pinn,
            "time_s":               time_pinn,
            "alpha_true":           alpha_true,
            "alpha_learned":        alpha_final,
            "error_relativo_alpha": abs(alpha_true - alpha_final) / alpha_true,
        },
        "nn": {
            "error_L2_mean": error_nn,
            "time_s":        time_nn,
        },
        "crank_nicolson": {
            "error_L2_mean": error_cn,
            "time_s":        time_cn,
            "method":        ref_cn["method"],
        },
    }

    historial_completo = {
        "pinn": hist_pinn,
        "nn":   hist_nn,
    }

    print(f"\n{'=' * 60}")
    print("RESULTADOS FINALES (Ecuación del Calor | Problema Inverso)")
    print(f"{'Parámetro':<15} {'Real':>10} {'Predicho':>10} {'Error (%)':>10}")
    print(f"{'alpha':<15} {alpha_true:>10.4f} {alpha_final:>10.4f} "
          f"{abs(alpha_true - alpha_final) / alpha_true * 100:>10.2f}")
    print(f"\n{'Método':<15} {'Error L2':>12} {'Tiempo (s)':>12}")
    print(f"{'Crank-Nicolson':<15} {error_cn:>12.4e} {time_cn:>12.4f}")
    print(f"{'NN pura':<15} {error_nn:>12.4e} {time_nn:>12.2f}")
    print(f"{'PINN':<15} {error_pinn:>12.4e} {time_pinn:>12.2f}")
    print("=" * 60)

    save_experiment_results(config_exp, final_results, historial_completo)


if __name__ == "__main__":

    SEEDS = [42, 123, 7, 99, 2024, 314, 17, 56, 88, 200] 

    # ----------------------------------------------------------------
    # Configuración BASE del experimento
    # ----------------------------------------------------------------
    BASE = dict(
        L=1.0, t_max=1.0,
        alpha_true=0.1, alpha_init=0.5,
        noise_std=0.0,
        epochs=10000,
        lr=1e-3,
        num_collocation=3000,
        num_train_points=20,
        num_ic_points=100,
        num_bc_points=100,
        sampler="lhs",
        warmup_epochs=2000,
        use_dynamic_weights=False,
        optimizer_name="adam",
        hidden_layers=[32, 32, 32],
        log_freq=10000,
        save_plots=False,
    )

    # ----------------------------------------------------------------
    # Matriz de configuraciones para el estudio de sensibilidad
    # ----------------------------------------------------------------
    variaciones = [
        # --- BASE ---
        {},

        # --- Eje 1: Puntos de colocación ---
        {"num_collocation": 500},
        {"num_collocation": 1000},
        {"num_collocation": 5000},

        # --- Eje 2: Sampler ---
        {"sampler": "grid"},

        # --- Eje 3: Optimizador ---
        {"optimizer_name": "adam+lbfgs"},

        # --- Eje 4: Arquitectura ---
        {"hidden_layers": [64, 64, 64]},

        # --- Eje 5: Learning rate ---
        {"lr": 0.01},

        # --- Eje 6: Warmup epochs ---
        {"warmup_epochs": 0},
        {"warmup_epochs": 500},
        {"warmup_epochs": 5000},

        # --- Eje 7: Pesos dinámicos ---
        {"use_dynamic_weights": True},

        # --- Eje 8: Puntos de entrenamiento ---
        {"num_train_points": 5},
        {"num_train_points": 10},
        {"num_train_points": 30},

        # --- Eje 9: Ruido ---
        {"noise_std": 0.05},
        {"noise_std": 0.1},
        {"noise_std": 0.2},

        # --- Eje 10: Alpha true ---
        {"alpha_true": 0.01},
        {"alpha_true": 0.3},
        
        # --- Eje 11: Alpha init ---
        {"alpha_init": 0.05},   
        {"alpha_init": 0.8},    
    ]

    total_configs = len(variaciones)
    total_runs    = total_configs * len(SEEDS)

    print(f"\n{'=' * 60}")
    print(f"ESTUDIO DE SENSIBILIDAD — Ecuación del Calor")
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