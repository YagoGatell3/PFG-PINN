import math
import os

import torch
import torch.optim as optim

from src.exact_solutions import psi_infinite_well
from src.loss_functions import (
    boundary_loss,
    normalization_loss,
    orthogonality_loss,
    physics_loss_infinite_well,
)
from src.models import PINN
from src.samplers import (
    generate_boundary_points,
    generate_grid_points,
    generate_lhs_points,
)
from src.utils import (
    calculate_l2_error,
    plot_and_save_results,
    save_experiment_results,
    set_seed,
    update_dynamic_weights,
    Timer,
    measure_numerical_reference,
)


def main(
    estado_n: int = 1,
    L: float = 1.0,
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
        "sistema": "pozo_infinito",
        "estado_n": estado_n,
        "L": L,
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

    print(f"--- Iniciando Entrenamiento (Pozo Infinito | Estado n={estado_n}) ---")
    print(
        f"Config -> Datos: {use_data} | Física: {use_physics} | Ortogonalidad: {use_orthogonality} | Pesos Din.: {use_dynamic_weights}\n"
    )

    pinn_model = PINN(hidden_layers=[32, 32, 32])
    optimizer = optim.Adam(pinn_model.parameters(), lr=lr)

    # 3. Generamos los datos fijos adaptados al pozo [0, L]
    x_min, x_max = 0.0, L
    x_left, x_right = generate_boundary_points(x_min, x_max)

    # Damos datos solo de la primera mitad del pozo para evaluar la generalización
    x_train = generate_grid_points(0.05, L / 2.0, num_train_points, requires_grad=False)
    u_train = psi_infinite_well(x_train, n=estado_n, L=L)

    x_eval = generate_grid_points(x_min, x_max, 500, requires_grad=False)
    u_true = psi_infinite_well(x_eval, n=estado_n, L=L)

    # Fórmula matemática de la energía exacta para comparar (m=1, hbar=1)
    exact_epsilon = (estado_n**2 * math.pi**2) / (2.0 * L**2)

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
        x_domain = generate_lhs_points(x_min, x_max, num_domain_points)
    else:
        x_domain = generate_grid_points(x_min, x_max, num_domain_points)

    # 4. Bucle de Entrenamiento
    with Timer() as t_pinn:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # A) Pérdida de Datos
            if use_data:
                pinn_pred_train = pinn_model(x_train)
                data_loss = torch.mean((pinn_pred_train - u_train) ** 2)
            else:
                data_loss = torch.tensor(0.0, device=x_domain.device)

            # B) Pérdidas Físicas (PINN)
            ph_loss_val = torch.tensor(0.0, device=x_domain.device)
            bound_loss = torch.tensor(0.0, device=x_domain.device)
            ortho_loss_val = torch.tensor(0.0, device=x_domain.device)
            norm_loss_val = torch.tensor(0.0, device=x_domain.device)

            if use_physics:
                ph_loss_val = physics_loss_infinite_well(pinn_model, x_domain)
                bound_loss = boundary_loss(pinn_model, x_left, x_right)

                if (
                    use_orthogonality and estado_n > 1
                ):  # Para el pozo, los excitados son > 1
                    psi_prev = psi_infinite_well(x_domain, n=estado_n - 1, L=L).detach()
                    ortho_loss_val = orthogonality_loss(
                        pinn_model, x_domain, psi_prev, domain_length=L
                    )

                if not use_data:
                    norm_loss_val = normalization_loss(
                        pinn_model, x_domain, domain_length=L
                    )

            # C) Actualización de pesos dinámicos
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

            # D) Pérdida Total
            peso_norm = 10.0
            peso_ortho = 10.0

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
                    + peso_ortho * ortho_loss_val
                    + peso_norm * norm_loss_val
                )

            total_loss.backward()
            optimizer.step()

            # 5. Monitorización
            if epoch % log_freq == 0 or epoch == epochs:
                print(
                    f"Época {epoch:05d} | Pérdida: {total_loss.item():.4e} | Epsilon: {pinn_model.epsilon.item():.4f}"
                )
                if use_orthogonality and use_physics:
                    print(f"          | Ortho Loss: {ortho_loss_val.item():.4e}")
                if use_dynamic_weights and use_data and use_physics:
                    print(
                        f"            | Pesos -> Física: {lambda_ph:.4f} |  Frontera: {lambda_bound:.4f}"
                    )

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())
                historial["ph_loss"].append(ph_loss_val.item())
                historial["bound_loss"].append(bound_loss.item())
                historial["ortho_loss"].append(ortho_loss_val.item())
                historial["epsilon"].append(pinn_model.epsilon.item())
                historial["lambda_ph"].append(lambda_ph)
                historial["lambda_bound"].append(lambda_bound)

                plot_train_x = x_train if use_data else None
                plot_train_u = u_train if use_data else None

                plot_and_save_results(
                    pinn_model,
                    plot_train_x,
                    plot_train_u,
                    x_eval,
                    u_true,
                    epoch,
                    total_loss.item(),
                    n=estado_n,
                    save_dir="img",
                    sistema="pozo_infinito",
                )

    # 6. Evaluación final
    pinn_pred_eval = pinn_model(x_eval).detach()
    error_l2 = calculate_l2_error(pinn_pred_eval, u_true)

    # Referencia numérica para comparar
    import numpy as np
    x_np = np.linspace(x_min, x_max, 1000)
    ref  = measure_numerical_reference(
        sistema="pozo_infinito",
        x_or_t=x_np,
        mass=1.0, hbar=1.0, k=estado_n + 1,
    )

    final_results = {
        "error_L2":          error_l2,
        "epsilon_final":     pinn_model.epsilon.item(),
        "epsilon_exacto":    exact_epsilon,
        "pinn_time_s":       t_pinn.elapsed,
        "numerical_time_s":  ref["time_s"],
        "numerical_method":  ref["method"],
        "speedup":           ref["time_s"] / t_pinn.elapsed,
    }

    print(f"\n--- RESULTADOS FINALES (Pozo Infinito | Estado n={estado_n}) ---")
    print(f"Autovalor exacto:       {exact_epsilon:.4f}")
    print(f"Autovalor PINN:         {pinn_model.epsilon.item():.4f}")
    print(f"Error relativo L2:      {error_l2:.4e}")
    print(f"Tiempo PINN:            {t_pinn.elapsed:.2f}s")
    print(f"Tiempo FDM:             {ref['time_s']:.4f}s")
    print(f"Speedup (FDM/PINN):     {ref['time_s']/t_pinn.elapsed:.4f}x\n")


if __name__ == "__main__":
    main(
        estado_n=2,
        L=1.0,
        epochs=10000,
        lr=0.001,
        num_domain_points=1000,
        num_train_points=20,
        sampler="lhs",
        log_freq=1000,
        use_data=True,
        use_dynamic_weights=False,
        use_orthogonality=False,
        use_physics=True,
    )
