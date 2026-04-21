import os

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
    calculate_l2_error,
    plot_and_save_results,
    save_experiment_results,
    set_seed,
    update_dynamic_weights,
    Timer,
    measure_numerical_reference,
)


def main(
    t_max: float = 10.0,
    mass: float = 1.0,
    k: float = 2.0,
    u_0: float = 1.0,
    v_0: float = 0.0,
    epochs: int = 5000,
    lr: float = 0.001,
    num_domain_points: int = 1000,
    num_train_points: int = 15,
    sampler: str = "lhs",
    log_freq: int = 1000,
    use_data: bool = True,
    use_dynamic_weights: bool = True,
    use_physics: bool = True,
):
    if not use_data and not use_physics:
        raise ValueError(
            "¡Error! Debes usar datos (use_data=True) o física (use_physics=True)."
        )

    # 1. Configuración inicial
    set_seed(42)
    os.makedirs("img", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    config_exp = {
        "sistema": "oscilador_clasico",
        "t_max": t_max,
        "mass": mass,
        "k": k,
        "epochs": epochs,
        "lr": lr,
        "num_domain_points": num_domain_points,
        "num_train_points": num_train_points,
        "sampler": sampler,
        "use_data": use_data,
        "use_dynamic_weights": use_dynamic_weights,
        "use_physics": use_physics,
    }

    print("--- Iniciando Entrenamiento (Oscilador Clásico) ---")
    print(
        f"Config -> Datos: {use_data} | Física: {use_physics} | Pesos Din.: {use_dynamic_weights}\n"
    )

    pinn_model = PINNDynamic(hidden_layers=[32, 32, 32])
    optimizer = optim.Adam(pinn_model.parameters(), lr=lr)

    # 3. Generamos los datos
    # El instante t=0 explícito para la condición inicial
    t_zero = torch.tensor([[0.0]], requires_grad=True)

    # Datos empíricos: Solo le enseñamos el primer 20% del tiempo (de 0 a t_max*0.2)
    # Esto obligará a la red a "predecir el futuro" (extrapolación)
    t_train = generate_grid_points(
        0.0, t_max * 0.2, num_train_points, requires_grad=False
    )
    u_train = classical_oscillator(t_train, mass=mass, k=k, u_0=u_0, v_0=v_0)

    # Puntos de evaluación (toda la línea temporal)
    t_eval = generate_grid_points(0.0, t_max, 500, requires_grad=False)
    u_true = classical_oscillator(t_eval, mass=mass, k=k, u_0=u_0, v_0=v_0)

    lambda_ph = 1.0
    lambda_ic = 1.0  # Sustituye a lambda_bound

    historial = {
        "epoch": [],
        "total_loss": [],
        "data_loss": [],
        "ph_loss": [],
        "ic_loss": [],
        "lambda_ph": [],
        "lambda_ic": [],
    }
    
    if sampler == "lhs":
        t_domain = generate_lhs_points(0.0, t_max, num_domain_points)
    else:
        t_domain = generate_grid_points(0.0, t_max, num_domain_points)

    # 4. Bucle de Entrenamiento
    with Timer() as t_pinn:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # A) Pérdida de Datos
            if use_data:
                pinn_pred_train = pinn_model(t_train)
                data_loss = torch.mean((pinn_pred_train - u_train) ** 2)
            else:
                data_loss = torch.tensor(0.0, device=t_domain.device)

            # B) Pérdidas Físicas (PINN)
            ph_loss_val = torch.tensor(0.0, device=t_domain.device)
            ic_loss_val = torch.tensor(0.0, device=t_domain.device)

            if use_physics:
                # 1. Ecuación de Newton en todo el dominio temporal
                ph_loss_val = physics_loss_classical_oscillator(
                    pinn_model, t_domain, mass=mass, k=k
                )
                # 2. Condiciones iniciales en t=0
                ic_loss_val = initial_condition_loss(pinn_model, t_zero, u_0=u_0, v_0=v_0)

            # C) Actualización de pesos dinámicos
            if use_dynamic_weights and use_data and use_physics:
                # Usamos ic_loss_val como si fuera bound_loss para actualizar los pesos
                lambda_ph, lambda_ic = update_dynamic_weights(
                    data_loss,
                    ph_loss_val,
                    ic_loss_val,
                    pinn_model.net[-1].weight,
                    lambda_ph,
                    lambda_ic,
                )
            else:
                lambda_ph, lambda_ic = 1.0, 1.0

            # D) Pérdida Total
            peso_ic = 10.0  # Si no usamos datos, damos más peso a la condición inicial para que la red "arranque" bien

            if not use_physics:
                total_loss = data_loss
            elif use_data:
                total_loss = data_loss + lambda_ph * ph_loss_val + lambda_ic * ic_loss_val
            else:
                total_loss = lambda_ph * ph_loss_val + peso_ic * ic_loss_val

            total_loss.backward()
            optimizer.step()

            # 5. Monitorización
            if epoch % log_freq == 0 or epoch == epochs:
                print(f"Época {epoch:05d} | Pérdida: {total_loss.item():.4e}")
                if use_dynamic_weights and use_data and use_physics:
                    print(
                        f"            | Pesos -> Física: {lambda_ph:.4f} |  Condición Inicial: {lambda_ic:.4f}"
                    )

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())
                historial["ph_loss"].append(ph_loss_val.item())
                historial["ic_loss"].append(ic_loss_val.item())
                historial["lambda_ph"].append(lambda_ph)
                historial["lambda_ic"].append(lambda_ic)

                plot_train_t = t_train if use_data else None
                plot_train_u = u_train if use_data else None

                # Usamos n=0 simplemente para que la función de utils no falle si lo requiere
                plot_and_save_results(
                    pinn_model,
                    plot_train_t,
                    plot_train_u,
                    t_eval,
                    u_true,
                    epoch,
                    total_loss.item(),
                    n=0,
                    save_dir="img",
                    sistema="oscilador_clasico",
                )

    # 6. Evaluación final
    pinn_pred_eval = pinn_model(t_eval).detach()
    error_l2 = calculate_l2_error(pinn_pred_eval, u_true)

    import numpy as np
    t_np = np.linspace(0.0, t_max, 500)
    ref  = measure_numerical_reference(
        sistema="oscilador_clasico",
        x_or_t=t_np,
        mass=mass, k=k, u0=u_0, v0=v_0,
    )

    final_results = {
        "error_L2":           error_l2,
        "pinn_time_s":        t_pinn.elapsed,
        "numerical_time_s":   ref["time_s"],
        "numerical_method":   ref["method"],
        "speedup":            ref["time_s"] / t_pinn.elapsed,
    }
    

    print("\n--- RESULTADOS FINALES (Oscilador Clásico) ---")
    print(f"Tiempo PINN:            {t_pinn.elapsed:.2f}s")
    print(f"Tiempo RK4:             {ref['time_s']:.4f}s")
    print(f"Speedup (RK4/PINN):     {ref['time_s']/t_pinn.elapsed:.4f}x\n")

    save_experiment_results(config_exp, final_results, historial)


if __name__ == "__main__":
    main(
        epochs=10000,
        num_train_points=15,
        use_data=True,
        use_physics=True,
    )
