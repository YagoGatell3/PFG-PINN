import os

import numpy as np
import torch
import torch.optim as optim

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
    save_experiment_results,
    set_seed,
    update_dynamic_weights,
    get_device,
)


def main(
    estado_n: int = 0,
    epochs: int = 5000,
    lr: float = 0.001,
    num_domain_points: int = 1000,
    num_train_points: int = 10,
    sampler: str = "lhs",
    log_freq: int = 1000,
    use_data: bool = True,
    use_dynamic_weights: bool = True,
    use_orthogonality: bool = False,
    use_physics: bool = True,
    optimizer_name: str = "adam",
):
    if not use_data and not use_physics:
        raise ValueError(
            "¡Error! Debes usar datos (use_data=True) o física (use_physics=True) para poder entrenar."
        )

    # 1. Configuración inicial
    set_seed(42)
    os.makedirs("img", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    config_exp = {
        "sistema": "qho",  # <- corregido typo y valor
        "estado_n": estado_n,
        "epochs": epochs,
        "lr": lr,
        "num_domain_points": num_domain_points,
        "num_train_points": num_train_points,
        "sampler": sampler,
        "use_data": use_data,
        "use_dynamic_weights": use_dynamic_weights,
        "use_orthogonality": use_orthogonality,
        "use_physics": use_physics,
    }

    print(f"--- Iniciando Entrenamiento (QHO | Estado n={estado_n}) ---")
    print(
        f"Config -> Datos: {use_data} | Física: {use_physics} | "
        f"Ortogonalidad: {use_orthogonality} | Pesos Din.: {use_dynamic_weights}\n"
    )

    # 2. Instancias del modelo y del optimizador
    device = get_device()
    pinn_model = PINN(hidden_layers=[32, 32, 32]).to(device)

    if optimizer_name == "adam":
        optimizer = optim.Adam(pinn_model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(pinn_model.parameters(), lr=lr)
    elif optimizer_name == "LBFGS":
        optimizer = optim.LBFGS(pinn_model.parameters(), lr=lr, max_iter=500)
    else:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no reconocido. Usa 'adam', 'sgd' o 'LBFGS'."
        )

    # 3. Generamos los datos fijos
    x_min, x_max = -10.0, 10.0
    x_left, x_right = generate_boundary_points(x_min, x_max)
    x_left = x_left.to(device)
    x_right = x_right.to(device)

    x_train = generate_grid_points(-5.0, 0.0, num_train_points, requires_grad=False).to(
        device
    )
    u_train = psi_QHO(x_train, n=estado_n).to(device)

    x_eval = generate_grid_points(x_min, x_max, 500, requires_grad=False).to(device)
    u_true = psi_QHO(x_eval, n=estado_n).to(device)

    lambda_ph = 1.0
    lambda_bound = 1.0
    historial = {
        "epoch": [],
        "total_loss": [],
        "data_loss": [],
        "ph_loss": [],
        "bound_loss": [],
        "ortho_loss": [],
        "epsilon": [],
        "lambda_ph": [],
        "lambda_bound": [],
    }

    if sampler == "lhs":
        x_domain = generate_lhs_points(x_min, x_max, num_domain_points).to(device)
    else:
        x_domain = generate_grid_points(x_min, x_max, num_domain_points).to(device)

    # 4. Bucle de Entrenamiento
    with Timer() as t_pinn:
        for epoch in range(1, epochs + 1):

            def closure():
                optimizer.zero_grad()

                if use_data:
                    pinn_pred_train = pinn_model(x_train)
                    data_loss = torch.mean((pinn_pred_train - u_train) ** 2)
                else:
                    data_loss = torch.tensor(0.0, device=device)  # <- device

                ph_loss_val = torch.tensor(0.0, device=device)  # <- device
                bound_loss_val = torch.tensor(0.0, device=device)  # <- device
                ortho_loss_val = torch.tensor(0.0, device=device)  # <- device
                norm_loss_val = torch.tensor(0.0, device=device)  # <- device

                if use_physics:
                    ph_loss_val = physics_loss_QHO(pinn_model, x_domain)
                    bound_loss_val = boundary_loss(pinn_model, x_left, x_right)

                    if use_orthogonality and estado_n > 0:
                        psi_prev = psi_QHO(x_domain, n=estado_n - 1).detach()
                        ortho_loss_val = orthogonality_loss(
                            pinn_model, x_domain, psi_prev, domain_length=20.0
                        )
                    if not use_data:
                        norm_loss_val = normalization_loss(
                            pinn_model, x_domain, domain_length=20.0
                        )

                if not use_physics:
                    total = data_loss
                elif use_data:
                    total = data_loss + ph_loss_val + bound_loss_val + ortho_loss_val
                else:
                    total = (
                        ph_loss_val
                        + bound_loss_val
                        + 10.0 * ortho_loss_val
                        + 10.0 * norm_loss_val
                    )

                total.backward()
                return total

            # LBFGS necesita closure, Adam y SGD no
            if optimizer_name == "LBFGS":
                total_loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()

                if use_data:
                    pinn_pred_train = pinn_model(x_train)
                    data_loss = torch.mean((pinn_pred_train - u_train) ** 2)
                else:
                    data_loss = torch.tensor(0.0, device=device)

                ph_loss_val = torch.tensor(0.0, device=device)
                bound_loss = torch.tensor(0.0, device=device)
                ortho_loss_val = torch.tensor(0.0, device=device)
                norm_loss_val = torch.tensor(0.0, device=device)

                if use_physics:
                    ph_loss_val = physics_loss_QHO(pinn_model, x_domain)
                    bound_loss = boundary_loss(pinn_model, x_left, x_right)

                    if use_orthogonality and estado_n > 0:
                        psi_prev = psi_QHO(x_domain, n=estado_n - 1).detach()
                        ortho_loss_val = orthogonality_loss(
                            pinn_model, x_domain, psi_prev, domain_length=20.0
                        )
                    if not use_data:
                        norm_loss_val = normalization_loss(
                            pinn_model, x_domain, domain_length=20.0
                        )

                if use_dynamic_weights and use_data and use_physics:
                    lambda_ph, lambda_bound = update_dynamic_weights(
                        data_loss,
                        ph_loss_val,
                        bound_loss,
                        pinn_model.net[-1].weight,
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
                        + lambda_bound * bound_loss
                        + ortho_loss_val
                    )
                else:
                    total_loss = (
                        lambda_ph * ph_loss_val
                        + lambda_bound * bound_loss
                        + 10.0 * ortho_loss_val
                        + 10.0 * norm_loss_val
                    )

                total_loss.backward()
                optimizer.step()

            # Monitorización
            if epoch % log_freq == 0 or epoch == epochs:
                print(
                    f"Época {epoch:05d} | Pérdida: {total_loss.item():.4e} "
                    f"| Epsilon: {pinn_model.epsilon.item():.4f}"
                )
                if use_dynamic_weights and use_data and use_physics:
                    print(
                        f"            | Pesos -> Física: {lambda_ph:.4f} "
                        f"| Frontera: {lambda_bound:.4f}"
                    )

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item() if use_data else 0.0)
                historial["ph_loss"].append(ph_loss_val.item() if use_physics else 0.0)
                historial["bound_loss"].append(
                    bound_loss.item() if use_physics else 0.0
                )
                historial["ortho_loss"].append(
                    ortho_loss_val.item() if use_physics else 0.0
                )
                historial["epsilon"].append(pinn_model.epsilon.item())
                historial["lambda_ph"].append(lambda_ph)
                historial["lambda_bound"].append(lambda_bound)

    # 5. Referencia numérica FDM
    x_np = np.linspace(x_min, x_max, 1000)
    ref = measure_numerical_reference(
        sistema="qho", x_or_t=x_np, mass=1.0, omega=1.0, hbar=1.0, k=estado_n + 1
    )

    # 6. Evaluación final
    pinn_pred_eval = pinn_model(x_eval).detach().cpu()
    error_l2 = calculate_l2_error(pinn_pred_eval, u_true.cpu())

    eigenvalues_fdm, eigenvectors_fdm = ref["solution"]
    epsilon_fdm = eigenvalues_fdm[estado_n]
    psi_fdm = eigenvectors_fdm[:, estado_n]

    from scipy.interpolate import interp1d

    fdm_interp = interp1d(x_np, psi_fdm, kind="cubic")
    x_eval_np = x_eval.detach().cpu().numpy().flatten()
    psi_fdm_eval = torch.tensor(fdm_interp(x_eval_np), dtype=torch.float32).unsqueeze(1)
    error_l2_fdm = calculate_l2_error(psi_fdm_eval, u_true.cpu())

    final_results = {
        "error_L2": error_l2,
        "error_L2_fdm": error_l2_fdm,
        "epsilon_final": pinn_model.epsilon.item(),
        "epsilon_exacto": estado_n + 0.5,
        "epsilon_fdm": float(epsilon_fdm),
        "pinn_time_s": t_pinn.elapsed,
        "numerical_time_s": ref["time_s"],
        "numerical_method": ref["method"],
        "speedup": ref["time_s"] / t_pinn.elapsed,
    }

    print(f"\n--- RESULTADOS FINALES (QHO | Estado n={estado_n}) ---")
    print(f"{'Método':<12} {'Autovalor':>12} {'Error L2':>12}")
    print(f"{'Exacto':<12} {estado_n + 0.5:>12.4f} {'—':>12}")
    print(f"{'FDM':<12} {float(epsilon_fdm):>12.4f} {error_l2_fdm:>12.4e}")
    print(f"{'PINN':<12} {pinn_model.epsilon.item():>12.4f} {error_l2:>12.4e}")
    print(f"\nTiempo PINN:        {t_pinn.elapsed:.2f}s")
    print(f"Tiempo FDM:         {ref['time_s']:.4f}s")
    print(f"Speedup (FDM/PINN): {ref['time_s'] / t_pinn.elapsed:.4f}x\n")

    save_experiment_results(config_exp, final_results, historial)


if __name__ == "__main__":
    from itertools import product

    # ============================================================
    # CAMBIA ESTE NÚMERO CADA VEZ QUE EJECUTES (0, 1, 2, ... 8)
    BATCH = 0
    BATCH_SIZE = 35
    # ============================================================

    estados = [0, 1, 2, 3]
    samplers = ["grid", "lhs"]
    domain_points = [500, 1000, 2000]
    data_physics_combos = [
        (True, True),
        (True, False),
        (False, True),
    ]
    dynamic_weights = [True, False]
    orthogonality_options = [True, False]
    optimizadores = ["adam", "sgd", "LBFGS"]

    configs = []
    for estado, sampler, n_pts, (
        use_data,
        use_physics,
    ), dyn_w, use_ortho, opt in product(
        estados,
        samplers,
        domain_points,
        data_physics_combos,
        dynamic_weights,
        orthogonality_options,
        optimizadores,
    ):
        if dyn_w and not (use_data and use_physics):
            continue
        if use_ortho and (not use_physics or estado == 0):
            continue
        if opt == "LBFGS" and dyn_w:
            continue

        configs.append(
            (estado, sampler, n_pts, use_data, use_physics, dyn_w, use_ortho, opt)
        )

    total = len(configs)
    inicio = BATCH * BATCH_SIZE
    fin = min(inicio + BATCH_SIZE, total)
    batch_configs = configs[inicio:fin]

    print(f"Total experimentos: {total}")
    print(f"Ejecutando batch {BATCH}: experimentos {inicio + 1} a {fin} de {total}\n")

    for (
        estado,
        sampler,
        n_pts,
        use_data,
        use_physics,
        dyn_w,
        use_ortho,
        opt,
    ) in batch_configs:
        print(
            f"\n>>> n={estado} | sampler={sampler} | pts={n_pts} | "
            f"data={use_data} | physics={use_physics} | "
            f"dynamic={dyn_w} | ortho={use_ortho} | opt={opt}"
        )
        try:
            main(
                estado_n=estado,
                epochs=10000,
                sampler=sampler,
                num_domain_points=n_pts,
                use_data=use_data,
                use_physics=use_physics,
                use_dynamic_weights=dyn_w,
                use_orthogonality=use_ortho,
                optimizer_name=opt,
                log_freq=99999,
            )
        except Exception as e:
            print(f"ERROR en config: {e}")
            continue
