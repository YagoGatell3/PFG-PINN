import os

import torch
import torch.optim as optim

from src.exact_solutions import psi
from src.loss_functions import (
    boundary_loss,
    normalization_loss,
    orthogonality_loss,
    physics_loss,
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
):
    # 1. Configuración inicial
    set_seed(42)
    os.makedirs("img", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    config_exp = {
        "estado_n": estado_n,
        "epochs": epochs,
        "lr": lr,
        "num_domain_points": num_domain_points,
        "num_train_points": num_train_points,
        "sampler": sampler,
        "use_data": use_data,
        "use_dynamic_weights": use_dynamic_weights,
        "use_orthogonality": use_orthogonality,
    }

    print(f"--- Iniciando Entrenamiento PINN (Estado n={estado_n}) ---")
    print(
        f"Config -> Datos: {use_data} | Ortogonalidad: {use_orthogonality} | Pesos Din.: {use_dynamic_weights}\n"
    )

    # 2. Instancias del modelo y del optimizador
    pinn_model = PINN(hidden_layers=[32, 32, 32])
    optimizer = optim.Adam(pinn_model.parameters(), lr=lr)

    # 3. Generamos los datos fijos
    x_min, x_max = -10.0, 10.0
    x_left, x_right = generate_boundary_points(x_min, x_max)

    x_train = generate_grid_points(-5.0, 0.0, num_train_points, requires_grad=False)
    u_train = psi(x_train, n=estado_n)

    x_eval = generate_grid_points(x_min, x_max, 500, requires_grad=False)
    u_true = psi(x_eval, n=estado_n)

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

    # 4. Bucle de Entrenamiento
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        if sampler == "lhs":
            x_domain = generate_lhs_points(x_min, x_max, num_domain_points)
        else:
            x_domain = generate_grid_points(x_min, x_max, num_domain_points)

        # A) Pérdida de Datos (Opcional)
        if use_data:
            pinn_pred_train = pinn_model(x_train)
            data_loss = torch.mean((pinn_pred_train - u_train) ** 2)
        else:
            data_loss = torch.tensor(0.0, device=x_domain.device)

        # B) Pérdida de Física
        ph_loss_val = physics_loss(pinn_model, x_domain)

        # C) Pérdida de Frontera
        bound_loss = boundary_loss(pinn_model, x_left, x_right)

        # D) Pérdida de Ortogonalidad (Opcional)
        ortho_loss_val = torch.tensor(0.0, device=x_domain.device)
        if use_orthogonality and estado_n > 0:
            # Calculo de la solución exacta del estado anterior (n-1) en los mismos puntos
            psi_prev = psi(x_domain, n=estado_n - 1).detach()

            ortho_loss_val = orthogonality_loss(
                pinn_model, x_domain, psi_prev, domain_length=20.0
            )

        # E) Pérdida de Normalización (Opcional)
        norm_loss_val = torch.tensor(0.0, device=x_domain.device)
        if not use_data:
            norm_loss_val = normalization_loss(pinn_model, x_domain, domain_length=20.0)

        # F) Actualización de pesos dinámicos
        if use_dynamic_weights and use_data:
            lambda_ph, lambda_bound = update_dynamic_weights(
                data_loss,
                ph_loss_val,
                bound_loss,
                pinn_model.net[-1].weight,
                lambda_ph,
                lambda_bound,
            )
        else:
            # Si no hay datos, los pesos son 1 para evitar errores
            lambda_ph, lambda_bound = 1.0, 1.0

        # Pérdida Total
        peso_norm = 10.0
        peso_ortho = 10.0

        if use_data:
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
            if use_orthogonality:
                print(f"          | Ortho Loss: {ortho_loss_val.item():.4e}")
            if use_dynamic_weights and use_data:
                print(
                    f"            | Pesos -> Física: {lambda_ph:.4f} |  Frontera: {lambda_bound:.4f}"
                )

            # Guardamos historial
            historial["epoch"].append(epoch)
            historial["total_loss"].append(total_loss.item())
            historial["data_loss"].append(data_loss.item())
            historial["ph_loss"].append(ph_loss_val.item())
            historial["bound_loss"].append(bound_loss.item())
            historial["ortho_loss"].append(ortho_loss_val.item())
            historial["epsilon"].append(pinn_model.epsilon.item())
            historial["lambda_ph"].append(lambda_ph)
            historial["lambda_bound"].append(lambda_bound)

            # Para la visualización, si no usamos datos, no los pintamos
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
            )

    # 6. Evaluación final
    pinn_pred_eval = pinn_model(x_eval).detach()
    error_l2 = calculate_l2_error(pinn_pred_eval, u_true)

    final_results = {
        "error_L2": error_l2,
        "epsilon_final": pinn_model.epsilon.item(),
        "epsilon_exacto": estado_n + 0.5,
    }

    print(f"\n--- RESULTADOS FINALES (Estado n={estado_n}) ---")
    print(f"Autovalor exacto:  {estado_n + 0.5:.4f}")
    print(f"Autovalor PINN:    {pinn_model.epsilon.item():.4f}")
    print(f"Error relativo L2: {error_l2:.4e}\n")

    save_experiment_results(config_exp, final_results, historial)


if __name__ == "__main__":
    main(
        estado_n=1,
        epochs=10000,
        lr=0.001,
        num_domain_points=1000,
        num_train_points=10,
        sampler="lhs",
        log_freq=1000,
        use_data=True,
        use_dynamic_weights=True,
        use_orthogonality=False,
    )
