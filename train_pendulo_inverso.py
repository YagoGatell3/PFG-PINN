import os
import math
import numpy as np

import torch
import torch.optim as optim

# Importamos el generador numérico en lugar de soluciones exactas
from src.numerical_methods import solve_damped_pendulum_rk4
from src.loss_functions import physics_loss_damped_pendulum
from src.models import PINNDampedPendulum
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
    t_max: float = 20.0,
    L: float = 1.0,
    g_true: float = 9.81,  # Valor real que la red debe descubrir
    mu_true: float = 0.5,  # Valor real que la red debe descubrir
    theta0: float = math.pi / 4.0, # Posición inicial (45 grados)
    omega0: float = 0.0,           # Velocidad inicial
    noise_std: float = 0.05,       # Nivel de ruido en las observaciones
    epochs: int = 15000,
    lr: float = 0.001,
    num_domain_points: int = 2000,
    num_train_points: int = 80,
    sampler: str = "lhs",
    log_freq: int = 1000,
    use_dynamic_weights: bool = False,
    optimizer_name: str = "adam",
):
    # En un problema inverso, SIEMPRE necesitamos datos y física
    use_data = True
    use_physics = True

    # 1. Configuración inicial
    set_seed(42)
    os.makedirs("img", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    config_exp = {
        "sistema": "pendulo_inverso",
        "t_max": t_max,
        "L": L,
        "g_true": g_true,
        "mu_true": mu_true,
        "noise_std": noise_std,
        "epochs": epochs,
        "lr": lr,
        "num_domain_points": num_domain_points,
        "num_train_points": num_train_points,
        "sampler": sampler,
        "use_dynamic_weights": use_dynamic_weights,
    }

    print("--- Iniciando Entrenamiento (Péndulo Inverso / Descubrimiento de Parámetros) ---")
    print(f"Buscando g={g_true} y mu={mu_true} a partir de datos ruidosos (std={noise_std})\n")

    # 2. Instancias del modelo y del optimizador
    pinn_model = PINNDampedPendulum(hidden_layers=[32, 32, 32, 32, 32]) 
    # Selector de optimizadores con grupos de parámetros (Learning Rates separados)
    if optimizer_name == "adam":
        optimizer = optim.Adam([
            {'params': pinn_model.net.parameters(), 'lr': lr},
            {'params': [pinn_model.g, pinn_model.mu], 'lr': lr * 20} 
        ])
    elif optimizer_name == "sgd":
        optimizer = optim.SGD([
            {'params': pinn_model.net.parameters(), 'lr': lr},
            {'params': [pinn_model.g, pinn_model.mu], 'lr': lr * 20}
        ])
    elif optimizer_name == "LBFGS":
        optimizer = optim.LBFGS([
            {'params': pinn_model.net.parameters(), 'lr': lr},
            {'params': [pinn_model.g, pinn_model.mu], 'lr': lr * 20}
        ], max_iter=500)
    else:
        raise ValueError("Optimizador no reconocido. Usa 'adam', 'sgd' o 'LBFGS'.")

    # 3. Generamos los datos fijos
    
    # 3.1 Puntos de colocación físicos (Fuera del bucle for, como optimización)
    if sampler == "lhs":
        t_domain = generate_lhs_points(0.0, t_max, num_domain_points, requires_grad=True)
    else:
        t_domain = generate_grid_points(0.0, t_max, num_domain_points, requires_grad=True)

    # 3.2 Generar Ground Truth con RK4 (numpy)
    t_rk4 = np.linspace(0.0, t_max, 1000)
    import numpy as np
    ref_rk4 = measure_numerical_reference(
        sistema="pendulo_inverso",
        x_or_t=t_rk4,
        g=g_true, mu=mu_true, L=L, theta0=theta0, omega0=omega0,
    )
    theta_rk4 = ref_rk4["solution"][0]  # devuelve (theta, omega)

    # Convertimos igual que antes
    t_eval = torch.tensor(t_rk4, dtype=torch.float32).unsqueeze(1)
    u_true = torch.tensor(theta_rk4, dtype=torch.float32).unsqueeze(1)


    # 3.3 Datos de entrenamiento empíricos (simulando sensores reales)
    # Seleccionamos índices aleatorios o equidistantes del array rk4
    idx_train = np.linspace(0, len(t_rk4) - 1, num_train_points).astype(int)
    t_train_np = t_rk4[idx_train]
    theta_train_np = theta_rk4[idx_train]

    # Añadimos ruido gaussiano a las observaciones
    np.random.seed(42)
    noise = np.random.normal(0, noise_std, size=theta_train_np.shape)
    theta_train_noisy_np = theta_train_np + noise

    t_train = torch.tensor(t_train_np, dtype=torch.float32).unsqueeze(1)
    u_train = torch.tensor(theta_train_noisy_np, dtype=torch.float32).unsqueeze(1)

    # Variables para pesos dinámicos y métricas
    lambda_ph = 1.0
    lambda_data = 1.0 # En inversos a veces se ponderan los datos respecto a la física
    
    historial = {
        "epoch": [],
        "total_loss": [],
        "data_loss": [],
        "ph_loss": [],
        "g_pred": [],
        "mu_pred": [],
        "lambda_ph": [],
        "lambda_data": [],
    }

    # 4. Bucle de Entrenamiento
    with Timer() as t_pinn:
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # A) Pérdida de Datos (usando los datos con ruido)
            pinn_pred_train = pinn_model(t_train)
            data_loss = torch.mean((pinn_pred_train - u_train) ** 2)

            # B) Pérdidas Físicas (PINN) - Descubriendo la física
            ph_loss_val = physics_loss_damped_pendulum(pinn_model, t_domain, L=L)

            # C) Actualización de pesos dinámicos
            if use_dynamic_weights:
                dummy_bound = torch.tensor(0.0, device=t_domain.device, requires_grad=True)
                lambda_ph, _ = update_dynamic_weights(
                    data_loss,
                    ph_loss_val,
                    dummy_bound,
                    pinn_model.net[-1].weight,
                    lambda_ph,
                    1.0, # dummy lambda
                )

            # D) Pérdida Total
            warmup_epochs = 3000  # primeras épocas: solo datos
            ph_weight = lambda_ph if use_dynamic_weights else 1.0

            if epoch < warmup_epochs:
                # Fase 1: La red aprende la trayectoria sin restricción física
                total_loss = data_loss
            else:
                # Fase 2: Introducimos la física suavemente
                ramp = min(1.0, (epoch - warmup_epochs) / 2000.0)
                total_loss = data_loss + ramp * ph_weight * ph_loss_val
            
            total_loss = data_loss + lambda_ph * ph_loss_val

            total_loss.backward()
            optimizer.step()

            # 5. Monitorización
            if epoch % log_freq == 0 or epoch == epochs:
                g_pred = pinn_model.g.item()
                mu_pred = pinn_model.mu.item()
                
                print(f"Época {epoch:05d} | Pérdida: {total_loss.item():.4e}")
                print(f"            | g real: {g_true:.4f}  -> g pred: {g_pred:.4f} | Error: {abs(g_true-g_pred)/g_true*100:.2f}%")
                print(f"            | mu real: {mu_true:.4f} -> mu pred: {mu_pred:.4f} | Error: {abs(mu_true-mu_pred)/mu_true*100:.2f}%")
                
                if use_dynamic_weights:
                    print(f"            | Peso Dinámico Física: {lambda_ph:.4f}")

                historial["epoch"].append(epoch)
                historial["total_loss"].append(total_loss.item())
                historial["data_loss"].append(data_loss.item())
                historial["ph_loss"].append(ph_loss_val.item())
                historial["g_pred"].append(g_pred)
                historial["mu_pred"].append(mu_pred)
                historial["lambda_ph"].append(lambda_ph)

                plot_and_save_results(
                    pinn_model,
                    t_train,
                    u_train,
                    t_eval,
                    u_true,
                    epoch,
                    total_loss.item(),
                    n=0,
                    save_dir="img",
                    sistema="pendulo_inverso", 
                )

    # 6. Evaluación final
    pinn_model.eval() # Buenas prácticas: ponemos el modelo en modo evaluación
    with torch.no_grad():
        pinn_pred_eval = pinn_model(t_eval)
        
    error_l2 = calculate_l2_error(pinn_pred_eval, u_true)

    final_results = {
        "error_L2":              error_l2,
        "g_final":               pinn_model.g.item(),
        "mu_final":              pinn_model.mu.item(),
        "error_relativo_g":      abs(g_true - pinn_model.g.item()) / g_true,
        "error_relativo_mu":     abs(mu_true - pinn_model.mu.item()) / mu_true,
        "pinn_time_s":           t_pinn.elapsed,
        "numerical_time_s":      ref_rk4["time_s"],
        "numerical_method":      ref_rk4["method"],
        "speedup":               ref_rk4["time_s"] / t_pinn.elapsed,
    }

    print("\n--- RESULTADOS FINALES (Péndulo Inverso) ---")
    print(f"Gravedad (g):           Real = {g_true:.4f} | Predicha = {pinn_model.g.item():.4f}")
    print(f"Amortig. (mu):          Real = {mu_true:.4f} | Predicho = {pinn_model.mu.item():.4f}")
    print(f"Error relativo L2:      {error_l2:.4e}")
    print(f"Tiempo PINN:            {t_pinn.elapsed:.2f}s")
    print(f"Tiempo RK4:             {ref_rk4['time_s']:.4f}s")
    print(f"Speedup (RK4/PINN):     {ref_rk4['time_s']/t_pinn.elapsed:.4f}x\n")



if __name__ == "__main__":
    main(
        epochs=20000,         
        num_train_points=80,  
        noise_std=0,        # 10% de ruido para demostrar la robustez
        use_dynamic_weights=False,
    )