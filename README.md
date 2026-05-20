# Physics-Informed Neural Networks (PINNs) aplicadas a Sistemas Dinámicos

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-fast%20environment%20manager-purple)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Trabajo de Fin de Grado** · Grado en Ingeniería Matemática · Universidad Francisco de Vitoria (UFV)

---

## Descripción

Este repositorio contiene el código fuente desarrollado para el TFG del Grado en Ingeniería Matemática en la UFV. El proyecto explora la implementación y el rendimiento de las *Physics-Informed Neural Networks* (PINNs) para la resolución de ecuaciones diferenciales parciales (EDPs) y ordinarias (EDOs), abordando tanto **problemas directos** como **problemas inversos** (descubrimiento de parámetros).

Las PINNs integran las leyes físicas directamente en la función de pérdida de la red neuronal, permitiendo resolver EDPs/EDOs sin necesidad de grandes conjuntos de datos etiquetados. Este trabajo evalúa su aplicabilidad en seis sistemas de distinta naturaleza y complejidad.

---

## Sistemas modelados

| # | Sistema | Tipo de problema | Ecuación |
|---|---------|-----------------|----------|
| 1 | **Oscilador Armónico Clásico (1D)** | Directo | EDO lineal de 2.º orden |
| 2 | **Péndulo Inverso Amortiguado** | Inverso | Identificación de $g$ y $\mu$ bajo ruido |
| 3 | **Ecuación del Calor (1D)** | Inverso | Identificación de difusividad térmica $\alpha$ |
| 4 | **Pozo de Potencial Infinito** | Directo | Ecuación de Schrödinger estacionaria |
| 5 | **Oscilador Armónico Cuántico (QHO)** | Directo | Estados estacionarios y cuantización de energía |
| 6 | **Efecto Túnel Cuántico** | Directo (dependiente del tiempo) | Propagación de paquete de ondas a través de barrera |

---

## Estructura del repositorio

```text
pfg-pinn/
├── src/
│   ├── models.py              # Arquitecturas de redes neuronales (FCNs)
│   ├── loss_functions.py      # Residuos de EDPs/EDOs integrados en la función de pérdida
│   ├── samplers.py            # Estrategias de muestreo de puntos de colocación (LHS, Grid)
│   ├── numerical_methods.py   # Métodos numéricos clásicos (RK4, FDM, Crank-Nicolson) para validación
│   ├── exact_solutions.py     # Soluciones analíticas (ground truth)
│   └── utils.py               # Funciones auxiliares
├── train_*.py                 # Scripts de entrenamiento por sistema
├── Analisis_*.ipynb           # Notebooks de análisis y visualización
├── pyproject.toml             # Configuración del proyecto y dependencias
├── uv.lock                    # Lockfile de entorno (uv)
└── README.md
```

---

## Instalación

Este proyecto utiliza [uv](https://github.com/astral-sh/uv) para la gestión del entorno. Asegúrate de tenerlo instalado antes de continuar.

```bash
# Clonar el repositorio
git clone https://github.com/yagogatell3/pfg-pinn.git
cd pfg-pinn

# Crear entorno virtual e instalar dependencias
uv sync
```

---

## Uso

Cada sistema dispone de su propio script de entrenamiento. Por ejemplo, para entrenar la PINN del oscilador armónico:

```bash
uv run train_ho.py
```

Para el análisis y la generación de gráficas, abre el notebook correspondiente:

```bash
uv run jupyter notebook Analisis_HO.ipynb
```

---

## Metodología

El flujo general seguido para cada sistema es el siguiente:

1. **Formulación física** — definición de la EDP/EDO, condiciones de contorno e iniciales.
2. **Arquitectura de la red** — red neuronal totalmente conectada (FCN) con activaciones `tanh`.
3. **Función de pérdida compuesta** — residuo de la ecuación física + pérdida en condiciones de contorno/iniciales (+ pérdida en datos observados para problemas inversos).
4. **Muestreo de puntos de colocación** — mediante *Latin Hypercube Sampling* (LHS) o muestreo en rejilla.
5. **Entrenamiento** — optimizadores Adam y L-BFGS en combinación (*two-phase training*).
6. **Validación** — comparación con solución analítica o métodos numéricos clásicos (RK4, FDM, Crank-Nicolson).

---

## Dependencias principales

| Librería | Uso |
|----------|-----|
| `torch` | Definición y entrenamiento de las redes neuronales |
| `numpy` | Operaciones numéricas y muestreo |
| `matplotlib` | Visualización de resultados |
| `scipy` | Métodos numéricos de referencia |
| `jupyter` | Notebooks de análisis |

Véase `pyproject.toml` para la lista completa y versiones exactas.

---

## Resultados

Los resultados (gráficas, métricas y JSONs) no están incluidos en el repositorio y se generan localmente al ejecutar los notebooks `Analisis_*.ipynb`. Las comparativas con métodos numéricos clásicos y el análisis de sensibilidad se documentan en la memoria del TFG.

---

## Autor

**Santiago Gatell** — Grado en Ingeniería Matemática, Universidad Francisco de Vitoria (UFV)

---

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
