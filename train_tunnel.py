import os

import numpy as np
import torch
import torch.optim as optim
from scipy.stats import qmc

from src.loss_functions import (
    boundary_loss_tunnel,
    data_loss_tunnel,
    initial_condition_loss_tunnel,
    normalization_loss_tunnel,
    physics_loss_tunnel,
)
from src.models import PINNTunnel
from src.utils import (
    Timer,
    calculate_l2_error,
    measure_numerical_reference,
    save_experiment_results,
    set_seed,
    update_dynamic_weights_tunnel,
)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Detecta y devuelve el dispositivo de cómputo disponible (GPU si está 
    disponible, de lo contrario CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Usando: {device}")
    return device


# ── Sampler 2D ────────────────────────────────────────────────────────────────

def sample_collocation_2d(
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    n_points: int,
    sampler: str = "lhs",
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Muestrea puntos de colocación (x, t) en el dominio espaciotemporal 2D.

    Args:
        x_min, x_max: Límites espaciales.
        t_min, t_max: Límites temporales.
        n_points: Número total de puntos a generar.
        sampler: Método de muestreo ('lhs' para Latin Hypercube o malla regular).
        device: Dispositivo donde se alojarán los tensores resultantes.

    Returns:
        Tupla de tensores (x, t) con requires_grad=True.
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

    x = torch.tensor(pts[:, 0:1], dtype=torch.float32,
                     requires_grad=True, device=device)
    t = torch.tensor(pts[:, 1:2], dtype=torch.float32,
                     requires_grad=True, device=device)
    return x, t


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_tunnel_results(
    model_pinn: torch.nn.Module,
    model_nn: torch.nn.Module,
    x_eval: torch.Tensor,
    t_snap_vals: list,
    prob_cn_snaps: list,
    x_barrier_left: float,
    x_barrier_right: float,
    epoch: int,
    loss: float,
    save_dir: str,
    filename: str = None,
):
    """
    Genera y guarda una comparativa gráfica de la densidad de probabilidad |Ψ|² 
    entre la PINN, la NN pura y la solución de referencia (Crank-Nicolson) 
    para diferentes instantes temporales (snapshots).
    """
    import matplotlib.pyplot as plt

    n_snaps  = len(t_snap_vals)
    fig, axes = plt.subplots(1, n_snaps, figsize=(5 * n_snaps, 4), sharey=False)
    if n_snaps == 1:
        axes = [axes]

    x_np   = x_eval.cpu().detach().numpy().flatten()
    device = x_eval.device

    for i, t_val in enumerate(t_snap_vals):
        t_tensor = torch.full((len(x_eval), 1), t_val, device=device)
        with torch.no_grad():
            u_p, v_p = model_pinn(x_eval, t_tensor)
            u_n, v_n = model_nn(x_eval,   t_tensor)

        # Cálculo de la densidad de probabilidad |Ψ|² = u² + v²
        prob_pinn = (u_p ** 2 + v_p ** 2).cpu().numpy().flatten()
        prob_nn   = (u_n ** 2 + v_n ** 2).cpu().numpy().flatten()

        axes[i].plot(x_np, prob_cn_snaps[i],
                     label="Crank-Nicolson", color="blue", linewidth=2, alpha=0.7)
        axes[i].plot(x_np, prob_pinn,
                     label="PINN", linestyle="--", color="black", linewidth=2)
        axes[i].plot(x_np, prob_nn,
                     label="NN", linestyle=":", color="gray", linewidth=1.5)
        
        # Sombreado de la barrera de potencial V₀
        axes[i].axvspan(x_barrier_left, x_barrier_right,
                        alpha=0.15, color="red", label="Barrera V₀")
        
        axes[i].set_title(f"t = {t_val:.2f}", fontsize=12)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("|Ψ|²")
        axes[i].legend(fontsize=8)
        axes[i].set_ylim(0, None)
        axes[i].grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fname = filename if filename else f"epoch_{epoch:05d}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


# ── Evaluación ────────────────────────────────────────────────────────────────

def _eval_error(
    model: torch.nn.Module,
    x_eval: torch.Tensor,
    prob_cn_full: np.ndarray,
    t_cn: np.ndarray,
    t_snap_vals: list,
) -> float:
    """
    Calcula el error L2 medio de la densidad de probabilidad |Ψ|² 
    entre las predicciones del modelo y la referencia de Crank-Nicolson.
    """
    device = x_eval.device
    errors = []
    for t_val in t_snap_vals:
        idx_t    = int(np.argmin(np.abs(t_cn - t_val)))
        t_tensor = torch.full((len(x_eval), 1), t_val, device=device)
        with torch.no_grad():
            u, v = model(x_eval, t_tensor)
        prob_pred = (u ** 2 + v ** 2).cpu()
        prob_ref  = torch.tensor(prob_cn_full[idx_t], dtype=torch.float32).unsqueeze(1)
        errors.append(calculate_l2_error(prob_pred, prob_ref))
    return float(np.mean(errors))


# ── Entrenamiento PINN ────────────────────────────────────────────────────────

def _train_pinn(
    x_ic: torch.Tensor,
    t_ic: torch.Tensor,
    t_bc: torch.Tensor,
    x_col: torch.Tensor,
    t_col: torch.Tensor,
    x_eval: torch.Tensor,
    prob_cn_full: np.ndarray,
    t_cn: np.ndarray,
    t_snap_vals: list,
    prob_cn_snaps: list,
    x_min: float,
    x_max: float,
    x_barrier_left: float,
    x_barrier_right: float,
    x0: float,
    sigma: float,
    k0: float,
    V0: float,
    mass: float,
    hbar: float,
    t_max: float,
    epochs: int,
    lr: float,
    warmup_epochs: int,
    optimizer_name: str,
    hidden_layers: list,
    log_freq: int,
    seed: int,
    device: torch.device,
    save_plots: bool = True,
    model_nn: torch.nn.Module = None,
    x_train: torch.Tensor = None,
    t_train: torch.Tensor = None,
    prob_train: torch.Tensor = None,
    lambda_data: float = 0.1,
    use_dynamic_weights: bool = False,
) -> tuple[float, float, dict, torch.nn.Module]:
    """
    Entrena la PINN para simular el efecto túnel cuántico (Ecuación de Schrödinger).
    La función de pérdida incluye:
    - IC (Condiciones iniciales)
    - BC (Condiciones de contorno)
    - Residuos físicos (TDSE)
    - Conservación de la probabilidad (Normalización)
    - (Opcional) Datos dispersos de referencia (Data Loss)
    """
    set_seed(seed)
    model = PINNTunnel(hidden_layers=hidden_layers).to(device)
    label = "PINN"

    use_data_pinn = x_train is not None

    historial = {
        "epoch":        [],
        "total_loss":   [],
        "data_loss":    [],
        "ph_loss":      [],
        "ic_loss":      [],
        "bc_loss":      [],
        "norm_loss":    [],
        "lambda_ph":    [],
        "lambda_bc":    [],
        "lambda_norm":  [],
        "lambda_data":  [],
    }

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=50)
    elif optimizer_name == "adam+lbfgs":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no reconocido. "
            f"Usa 'adam', 'lbfgs' o 'adam+lbfgs'."
        )

    # ── Inicialización de pesos dinámicos ─────────────────────────────────
    lam_ph   = 1.0
    lam_bc   = 1.0
    lam_norm = 1.0
    lam_data = lambda_data

    with Timer() as timer:
        for epoch in range(1, epochs + 1):

            # Transición a L-BFGS si se usa la estrategia combinada
            if optimizer_name == "adam+lbfgs" and epoch == int(epochs * 0.85) + 1:
                print(f"\n[{label}] Cambiando a L-BFGS en época {epoch}\n")
                optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=20)

            # Rampa de calentamiento para introducir la penalización física progresivamente
            if epoch < warmup_epochs:
                ph_ramp = 0.0
            else:
                ph_ramp = min(1.0, (epoch - warmup_epochs) / 2000.0)

            # ── Rama L-BFGS ───────────────────────────────────────────────
            is_lbfgs = optimizer_name == "lbfgs" or (
                optimizer_name == "adam+lbfgs" and epoch >= int(epochs * 0.85) + 1
            )

            if is_lbfgs:
                def closure(ramp=ph_ramp):
                    optimizer.zero_grad()
                    ic_loss   = initial_condition_loss_tunnel(
                        model, x_ic, t_ic, x0=x0, sigma=sigma, k0=k0
                    )
                    bc_loss   = boundary_loss_tunnel(
                        model, t_bc, x_min=x_min, x_max=x_max
                    )
                    ph_loss   = physics_loss_tunnel(
                        model, x_col, t_col,
                        V0=V0, x_barrier_left=x_barrier_left,
                        x_barrier_right=x_barrier_right,
                        mass=mass, hbar=hbar,
                    )
                    norm_loss = normalization_loss_tunnel(
                        model, x_ic, t_ic, domain_length=(x_max - x_min)
                    )
                    d_loss = (
                        data_loss_tunnel(model, x_train, t_train, prob_train)
                        if use_data_pinn
                        else torch.tensor(0.0, device=device)
                    )
                    # L-BFGS utiliza los últimos pesos dinámicos calculados (fijos)
                    total = (
                        ic_loss
                        + lam_bc   * bc_loss
                        + ramp * lam_ph * ph_loss
                        + lam_norm * norm_loss
                        + lam_data * d_loss
                    )
                    total.backward()
                    return total

                result     = optimizer.step(closure)
                total_loss = result if result is not None else torch.tensor(0.0, device=device)

                with torch.no_grad():
                    ic_loss   = initial_condition_loss_tunnel(
                        model, x_ic, t_ic, x0=x0, sigma=sigma, k0=k0
                    )
                    bc_loss   = boundary_loss_tunnel(
                        model, t_bc, x_min=x_min, x_max=x_max
                    )
                    norm_loss = normalization_loss_tunnel(
                        model, x_ic, t_ic, domain_length=(x_max - x_min)
                    )
                    d_loss = (
                        data_loss_tunnel(model, x_train, t_train, prob_train)
                        if use_data_pinn
                        else torch.tensor(0.0, device=device)
                    )
                ph_loss = torch.tensor(0.0, device=device)

            # ── Rama Adam ─────────────────────────────────────────────────
            else:
                optimizer.zero_grad()

                ic_loss   = initial_condition_loss_tunnel(
                    model, x_ic, t_ic, x0=x0, sigma=sigma, k0=k0
                )
                bc_loss   = boundary_loss_tunnel(
                    model, t_bc, x_min=x_min, x_max=x_max
                )
                ph_loss   = physics_loss_tunnel(
                    model, x_col, t_col,
                    V0=V0, x_barrier_left=x_barrier_left,
                    x_barrier_right=x_barrier_right,
                    mass=mass, hbar=hbar,
                )
                norm_loss = normalization_loss_tunnel(
                    model, x_ic, t_ic, domain_length=(x_max - x_min)
                )
                d_loss = (
                    data_loss_tunnel(model, x_train, t_train, prob_train)
                    if use_data_pinn
                    else torch.tensor(0.0, device=device)
                )

                # ── Pesos dinámicos (solo en Adam y si ph_ramp > 0) ───────
                if use_dynamic_weights and ph_ramp > 0.0:
                    lam_ph, lam_bc, lam_norm, lam_data = update_dynamic_weights_tunnel(
                        ic_loss=ic_loss,
                        ph_loss=ph_loss,
                        bc_loss=bc_loss,
                        norm_loss=norm_loss,
                        data_loss=d_loss,
                        last_layer_weight=model.net[-1].weight,
                        current_lambda_ph=lam_ph,
                        current_lambda_bc=lam_bc,
                        current_lambda_norm=lam_norm,
                        current_lambda_data=lam_data,
                    )

                total_loss = (
                    ic_loss
                    + lam_bc   * bc_loss
                    + ph_ramp * lam_ph * ph_loss
                    + lam_norm * norm_loss
                    + lam_data * d_loss
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # ── Registro de métricas ──────────────────────────────────────
            if epoch % log_freq == 0 or epoch == epochs:
                print(
                    f"[{label}] Época {epoch:05d} | Total: {total_loss.item():.4e} "
                    f"| IC: {ic_loss.item():.4e} | BC: {bc_loss.item():.4e} "
                    f"| Física: {ph_loss.item():.4e} (×{ph_ramp:.2f}) "
                    f"| Norma: {norm_loss.item():.4e} "
                    f"| Datos: {d_loss.item():.4e}"
                )
                if use_dynamic_weights:
                    print(
                        f"           | λ_ph={lam_ph:.3f} λ_bc={lam_bc:.3f} "
                        f"λ_norm={lam_norm:.3f} λ_data={lam_data:.3f}"
                    )

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(d_loss.item())
                historial["ph_loss"].append(ph_loss.item())
                historial["ic_loss"].append(ic_loss.item())
                historial["bc_loss"].append(bc_loss.item())
                historial["norm_loss"].append(norm_loss.item())
                historial["lambda_ph"].append(lam_ph)
                historial["lambda_bc"].append(lam_bc)
                historial["lambda_norm"].append(lam_norm)
                historial["lambda_data"].append(lam_data)

                if save_plots and model_nn is not None:
                    _plot_tunnel_results(
                        model, model_nn, x_eval,
                        t_snap_vals, prob_cn_snaps,
                        x_barrier_left, x_barrier_right,
                        epoch, total_loss.item(),
                        save_dir="img/tunnel/pinn",
                    )

    error_l2 = _eval_error(model, x_eval, prob_cn_full, t_cn, t_snap_vals)
    return error_l2, timer.elapsed, historial, model


# ── Entrenamiento NN pura ─────────────────────────────────────────────────────

def _train_nn(
    x_train: torch.Tensor,
    t_train: torch.Tensor,
    prob_train: torch.Tensor,
    x_eval: torch.Tensor,
    prob_cn_full: np.ndarray,
    t_cn: np.ndarray,
    t_snap_vals: list,
    epochs: int,
    lr: float,
    optimizer_name: str,
    hidden_layers: list,
    log_freq: int,
    seed: int,
    device: torch.device,
    save_plots: bool = True,
) -> tuple[float, float, dict, torch.nn.Module]:
    """
    Entrena un modelo puramente de datos (NN baseline) ajustando 
    la predicción (u² + v²) directamente a la probabilidad observada de Crank-Nicolson, 
    sin forzar el cumplimiento de la ecuación de Schrödinger.
    """
    set_seed(seed)
    model = PINNTunnel(hidden_layers=hidden_layers).to(device)
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
                optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=20)

            def closure():
                optimizer.zero_grad()
                u, v      = model(x_train, t_train)
                prob_pred = u ** 2 + v ** 2
                data_loss = torch.mean((prob_pred - prob_train) ** 2)
                data_loss.backward()
                return data_loss

            if optimizer_name == "lbfgs" or (
                optimizer_name == "adam+lbfgs" and epoch >= int(epochs * 0.85) + 1
            ):
                result     = optimizer.step(closure)
                total_loss = result if result is not None else torch.tensor(0.0, device=device)
                with torch.no_grad():
                    u, v      = model(x_train, t_train)
                    data_loss = torch.mean((u ** 2 + v ** 2 - prob_train) ** 2)
            else:
                optimizer.zero_grad()
                u, v       = model(x_train, t_train)
                prob_pred  = u ** 2 + v ** 2
                data_loss  = torch.mean((prob_pred - prob_train) ** 2)
                total_loss = data_loss
                total_loss.backward()
                optimizer.step()

            if epoch % log_freq == 0 or epoch == epochs:
                print(f"[{label}] Época {epoch:05d} | Pérdida: {total_loss.item():.4e}")
                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())

    error_l2 = _eval_error(model, x_eval, prob_cn_full, t_cn, t_snap_vals)
    return error_l2, timer.elapsed, historial, model


# ── Controlador Principal ─────────────────────────────────────────────────────

def main(
    x_min: float = -10.0,
    x_max: float = 10.0,
    t_max: float = 3.0,
    V0: float = 3.0,
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
    num_train_points: int = 100,
    sampler: str = "lhs",
    log_freq: int = 2000,
    warmup_epochs: int = 5000,
    optimizer_name: str = "adam",
    hidden_layers: list = None,
    seed: int = 42,
    save_plots: bool = True,
    save_final_plot: bool = True,
    use_data_pinn: bool = False,
    lambda_data: float = 0.1,
    use_dynamic_weights: bool = False,
):
    """
    Función orquestadora. Prepara el entorno para el experimento del Efecto Túnel:
    - Resuelve numéricamente mediante Crank-Nicolson como 'ground truth'.
    - Interpola datos para el muestreo de entrenamiento.
    - Lanza y evalúa el entrenamiento tanto de la NN baseline como de la PINN.
    - Calcula el coeficiente de transmisión probabilístico.
    """
    if hidden_layers is None:
        hidden_layers = [64, 64, 64, 64]

    set_seed(seed)
    device = get_device()

    os.makedirs("img/tunnel", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    config_exp = {
        "sistema":          "tunnel",
        "x_min":            x_min,
        "x_max":            x_max,
        "t_max":            t_max,
        "V0":               V0,
        "x_barrier_left":   x_barrier_left,
        "x_barrier_right":  x_barrier_right,
        "x0":               x0,
        "sigma":            sigma,
        "k0":               k0,
        "mass":             mass,
        "hbar":             hbar,
        "epochs":           epochs,
        "lr":               lr,
        "num_collocation":  num_collocation,
        "num_ic_points":    num_ic_points,
        "num_bc_points":    num_bc_points,
        "num_train_points": num_train_points,
        "sampler":          sampler,
        "warmup_epochs":    warmup_epochs,
        "optimizer":        optimizer_name,
        "hidden_layers":    hidden_layers,
        "seed":             seed,
        "use_data_pinn":    use_data_pinn,
        "lambda_data":      lambda_data,
        "use_dynamic_weights": use_dynamic_weights,
    }

    print("=" * 60)
    print("Efecto Túnel Cuántico — TDSE")
    print(
        f"Sampler: {sampler} | Colocación: {num_collocation} | "
        f"IC: {num_ic_points} | BC: {num_bc_points} | Train: {num_train_points} | "
        f"Épocas: {epochs} | Optimizador: {optimizer_name} | lr: {lr}"
    )
    print(
        f"Warmup: {warmup_epochs} | Seed: {seed} | "
        f"PINN con datos: {use_data_pinn}| "
        f"Pesos dinámicos: {use_dynamic_weights}"
    )
    print(
        f"Barrera: V₀={V0} en x∈[{x_barrier_left}, {x_barrier_right}] | "
        f"k₀={k0} | E≈{k0**2/2:.2f} | "
        f"{'Hay túnel' if k0**2/2 < V0 else 'No hay túnel'}"
    )
    print("=" * 60)

    # 1. Resolución de la referencia numérica mediante Crank-Nicolson
    print("\nCalculando solución de referencia (Crank-Nicolson)...")
    Nx_cn  = 500
    dx_cn  = (x_max - x_min) / (Nx_cn - 1)
    dt_max = 0.5 * mass * dx_cn / (hbar * k0)
    Nt_cn  = int(np.ceil(t_max / dt_max)) + 1
    x_cn   = np.linspace(x_min, x_max, Nx_cn)
    t_cn   = np.linspace(0.0, t_max, Nt_cn)
    
    ref_cn       = measure_numerical_reference(
        sistema="tunnel",
        x_or_t=x_cn, t_array=t_cn,
        x0=x0, sigma=sigma, k0=k0,
        V0=V0, x_barrier_left=x_barrier_left,
        x_barrier_right=x_barrier_right,
        mass=mass, hbar=hbar,
    )
    prob_cn_full = ref_cn["solution"]
    time_cn      = ref_cn["time_s"]
    print("Referencia calculada.\n")

    t_snap_vals   = [0.0, t_max * 0.33, t_max * 0.66, t_max]
    t_snap_idx    = [np.argmin(np.abs(t_cn - tv)) for tv in t_snap_vals]
    prob_cn_snaps = [prob_cn_full[i] for i in t_snap_idx]
    x_eval        = torch.tensor(x_cn, dtype=torch.float32).unsqueeze(1).to(device)

    # 2. Condiciones iniciales del paquete de ondas (t=0)
    x_ic_np = np.linspace(x_min, x_max, num_ic_points)
    x_ic    = torch.tensor(x_ic_np, dtype=torch.float32).unsqueeze(1).to(device)
    t_ic    = torch.zeros(num_ic_points, 1, device=device)

    # 3. Condiciones de contorno (Dirichlet)
    torch.manual_seed(seed)
    t_bc = torch.rand(num_bc_points, 1, device=device) * t_max

    # 4. Puntos de colocación para evaluar el residuo físico
    x_col, t_col = sample_collocation_2d(
        x_min, x_max, 0.0, t_max, num_collocation,
        sampler=sampler, device=device,
    )

    # 5. Muestreo de datos de entrenamiento desde la referencia interpolada
    np.random.seed(seed)
    x_train_np = np.random.uniform(x_min, x_max, num_train_points)
    t_train_np = np.random.uniform(0.0,   t_max,  num_train_points)

    from scipy.interpolate import RegularGridInterpolator
    interp_cn     = RegularGridInterpolator(
        (t_cn, x_cn), prob_cn_full,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    prob_train_np = interp_cn(np.stack([t_train_np, x_train_np], axis=1))

    x_train    = torch.tensor(x_train_np,    dtype=torch.float32).unsqueeze(1).to(device)
    t_train    = torch.tensor(t_train_np,    dtype=torch.float32).unsqueeze(1).to(device)
    prob_train = torch.tensor(prob_train_np, dtype=torch.float32).unsqueeze(1).to(device)

    # 6. Lanzamiento de la red neuronal estándar (NN)
    print("\n--- NN pura (datos CN, sin física) ---")
    error_nn, time_nn, hist_nn, model_nn = _train_nn(
        x_train=x_train, t_train=t_train, prob_train=prob_train,
        x_eval=x_eval, prob_cn_full=prob_cn_full, t_cn=t_cn,
        t_snap_vals=t_snap_vals,
        epochs=epochs, lr=lr,
        optimizer_name=optimizer_name, hidden_layers=hidden_layers,
        log_freq=log_freq, seed=seed, device=device, save_plots=save_plots,
    )

    # 7. Lanzamiento de la red física (PINN)
    print("\n--- PINN (física" + (" + datos)" if use_data_pinn else ")") + " ---")
    error_pinn, time_pinn, hist_pinn, model_pinn = _train_pinn(
        x_ic=x_ic, t_ic=t_ic, t_bc=t_bc,
        x_col=x_col, t_col=t_col,
        x_eval=x_eval, prob_cn_full=prob_cn_full, t_cn=t_cn,
        t_snap_vals=t_snap_vals, prob_cn_snaps=prob_cn_snaps,
        x_min=x_min, x_max=x_max,
        x_barrier_left=x_barrier_left, x_barrier_right=x_barrier_right,
        x0=x0, sigma=sigma, k0=k0, V0=V0, mass=mass, hbar=hbar,
        t_max=t_max, epochs=epochs, lr=lr,
        warmup_epochs=warmup_epochs,
        optimizer_name=optimizer_name, hidden_layers=hidden_layers,
        log_freq=log_freq, seed=seed, device=device, save_plots=save_plots,
        model_nn=model_nn,
        x_train=x_train if use_data_pinn else None,
        t_train=t_train if use_data_pinn else None,
        prob_train=prob_train if use_data_pinn else None,
        lambda_data=lambda_data,
        use_dynamic_weights=use_dynamic_weights,
    )

    # 8. Gráfica comparativa consolidada
    if save_final_plot:
        _plot_tunnel_results(
            model_pinn, model_nn, x_eval,
            t_snap_vals, prob_cn_snaps,
            x_barrier_left, x_barrier_right,
            epoch=epochs, loss=hist_pinn["total_loss"][-1],
            save_dir="img/tunnel",
            filename="comparativa_final.png",
        )

    # 9. Cálculo del Coeficiente de Transmisión (probabilidad más allá de la barrera)
    dx      = x_cn[1] - x_cn[0]
    t_final = torch.full((len(x_eval), 1), t_max, device=device)
    with torch.no_grad():
        u_p, v_p = model_pinn(x_eval, t_final)
        u_n, v_n = model_nn(x_eval,   t_final)

    prob_pinn_final = (u_p ** 2 + v_p ** 2).cpu().numpy().flatten()
    prob_nn_final   = (u_n ** 2 + v_n ** 2).cpu().numpy().flatten()
    prob_cn_final   = prob_cn_full[-1]

    T_cn   = float(np.sum(prob_cn_final[x_cn   > x_barrier_right]) * dx)
    T_pinn = float(np.sum(prob_pinn_final[x_cn > x_barrier_right]) * dx)
    T_nn   = float(np.sum(prob_nn_final[x_cn   > x_barrier_right]) * dx)

    # 10. Estructuración y guardado de resultados
    final_results = {
        "pinn": {
            "error_L2_mean": error_pinn,
            "time_s":        time_pinn,
            "T_pinn":        T_pinn,
        },
        "nn": {
            "error_L2_mean": error_nn,
            "time_s":        time_nn,
            "T_nn":          T_nn,
        },
        "crank_nicolson": {
            "time_s": time_cn,
            "T_cn":   T_cn,
            "method": ref_cn["method"],
        },
    }

    historial_completo = {
        "pinn": hist_pinn,
        "nn":   hist_nn,
    }

    print(f"\n{'=' * 60}")
    print("RESULTADOS FINALES (Efecto Túnel)")
    print(f"{'Método':<15} {'Error L2':>12} {'T (transmisión)':>16} {'Tiempo (s)':>12}")
    print(f"{'Crank-Nicolson':<15} {'—':>12} {T_cn:>16.4f} {time_cn:>12.4f}")
    print(f"{'NN pura':<15} {error_nn:>12.4e} {T_nn:>16.4f} {time_nn:>12.2f}")
    print(f"{'PINN':<15} {error_pinn:>12.4e} {T_pinn:>16.4f} {time_pinn:>12.2f}")
    print("=" * 60)

    save_experiment_results(config_exp, final_results, historial_completo)


if __name__ == "__main__":

    SEEDS = [42, 123, 7, 99, 2024, 314, 17, 56, 88, 200]

    # ----------------------------------------------------------------
    # Configuración BASE del experimento
    # ----------------------------------------------------------------
    BASE = dict(
        x_min=-10.0, x_max=10.0, t_max=3.0,
        V0=3.0, x_barrier_left=0.5, x_barrier_right=1.5,
        x0=-4.0, sigma=0.75, k0=2.0,
        mass=1.0, hbar=1.0,
        epochs=20000,
        lr=1e-3,
        num_collocation=5000,
        num_ic_points=500,
        num_bc_points=200,
        num_train_points=100,
        sampler="lhs",
        warmup_epochs=5000,
        optimizer_name="adam",
        hidden_layers=[64, 64, 64, 64],
        log_freq=20000,
        save_plots=False,
        save_final_plot=True,
        use_data_pinn=True,   
        lambda_data=1,
        use_dynamic_weights=False,
    )

    # ----------------------------------------------------------------
    # Matriz de configuraciones para el estudio de sensibilidad
    # ----------------------------------------------------------------
    variaciones = [
        # --- BASE (PINN con datos) ---
        {},

        # --- Eje 1: Puntos de colocación ---
        {"num_collocation": 1000},
        {"num_collocation": 2000},
        {"num_collocation": 10000},

        # --- Eje 2: Sampler ---
        {"sampler": "grid"},

        # --- Eje 3: Optimizador ---
        {"optimizer_name": "adam+lbfgs"},

        # --- Eje 4: Arquitectura ---
        {"hidden_layers": [128, 128, 128, 128]},

        # --- Eje 5: Learning rate ---
        {"lr": 0.01},

        # --- Eje 6: Warmup epochs ---
        {"warmup_epochs": 0},
        {"warmup_epochs": 2500},
        {"warmup_epochs": 7500},

        # --- Eje 7: Altura de barrera ---
        {"V0": 2.5},
        {"V0": 5.0},


        # --- Eje 8: Puntos de entrenamiento ---
        {"num_train_points": 20},
        {"num_train_points": 50},
        {"num_train_points": 200},

        # --- Eje 9: PINN sin datos ---
        {"use_data_pinn": False},
        
        # --- Eje 10: Pesos dinámicos ---
        {"use_dynamic_weights": True},
    ]

    total_configs = len(variaciones)
    total_runs    = total_configs * len(SEEDS)

    print(f"\n{'=' * 60}")
    print(f"ESTUDIO DE SENSIBILIDAD — Efecto Túnel")
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