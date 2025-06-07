# Proyecto: Caracterización de Patrones de Speckle mediante FBM

Este proyecto implementa la metodología descrita en Guyot et al. (2004) para caracterizar patrones de speckle sintéticos utilizando el formalismo del Movimiento Browniano Fraccional (FBM).

## Configuración del Entorno

1. Clona este repositorio (si aplica).
2. Asegúrate de tener Conda instalado.
3. Crea el entorno Conda desde el archivo `environment.yml` (ubicado en el directorio raíz del proyecto `project/`):
   ```bash
   conda env create -f environment.yml
   ```
4. Activa el entorno:
   ```bash
   conda activate speckle_fbmsim
   ```

## Uso

El análisis principal se realiza ejecutando el script `src/main_analysis.py`.

1.  Asegúrate de tener el entorno Conda `speckle_fbmsim` activado.
2.  Navega al directorio `src/` dentro de tu proyecto:
    ```bash
    cd project/src
    ```
3.  Ejecuta el script de análisis:
    ```bash
    python main_analysis.py
    ```
    El script generará imágenes de speckle, calculará la función de difusión, realizará el ajuste del modelo y guardará las figuras resultantes en la carpeta `project/figures/` y los datos en `project/data/`.
    Puedes modificar los parámetros del análisis (tamaño de imagen, FWHM) directamente en la sección `if __name__ == '__main__':` del archivo `main_analysis.py`.

## Estructura del Proyecto

- `data/`: Almacena datos generados, como imágenes de speckle sintéticas (ej. `synthetic_speckle.npy`).
- `notebooks/`: Puede contener cuadernos Jupyter para análisis exploratorios adicionales o visualizaciones (opcional).
- `src/`: Código fuente Python.
  - `speckle.py`: Funciones para generación de speckle, cálculo de función de difusión y modelo.
  - `utils.py`: Funciones auxiliares (ej. ploteo).
  - `main_analysis.py`: Script principal para ejecutar el flujo de análisis completo.
- `figures/`: Guarda las figuras generadas por el análisis.
- `environment.yml`: Definición del entorno Conda.
- `README.md`: Este archivo.

## Flujo de Trabajo del Script (`src/main_analysis.py`)

El script `src/main_analysis.py` no solo realiza un análisis individual, sino que también está configurado para ejecutar una serie de experimentos, por ejemplo, variando el parámetro FWHM. Los resultados detallados de cada experimento (imágenes, datos de FD, ajustes) se guardan en subcarpetas dentro de `project/figures/experiments/` y `project/data/`. 

Al finalizar la serie de experimentos, se genera un archivo CSV con el resumen de los parámetros ajustados (`project/data/experiment_results/`) y gráficos de resumen (`project/figures/experiment_summary_plots/`) que muestran cómo varían los parámetros estimados (H, S, G) en función del parámetro modificado (ej. FWHM).

Para una descripción detallada de los experimentos implementados, los parámetros analizados y la interpretación de los resultados, consulta el archivo <mcfile name="ANALYSIS_DETAILS.md" path="c:\Users\santi\OneDrive - Universidad de Antioquia\Maestría\2025-1\Física Estadística\Speckle\project\ANALYSIS_DETAILS.md"></mcfile>.

A continuación, se describe el flujo para un **único análisis** (que es la base de cada experimento en la serie):

1.  **Importaciones:**
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from src.speckle import generate_speckle, diffusion_function, fd_model, radial_fd
    from src.utils import plot_loglog_fd
    from scipy.optimize import curve_fit
    import os
    ```
2.  **Crear directorios si no existen (opcional, buena práctica):**
    ```python
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('figures'):
        os.makedirs('figures')
    ```
3.  **Generar Speckle:**

    El patrón de speckle se origina cuando una luz coherente, como la de un láser, se dispersa al incidir sobre una superficie rugosa o un medio turbio. Este fenómeno resulta de la interferencia de múltiples ondas de luz que, al recombinarse, crean un patrón granular de puntos brillantes y oscuros. En este proyecto, generamos patrones de speckle sintéticos utilizando el método de fase aleatoria mediante la Transformada Rápida de Fourier (FFT). Se crea una matriz de fases aleatorias y se aplica una envolvente gaussiana en el espacio de frecuencias para controlar el tamaño de grano del speckle. La transformada inversa de Fourier de este campo complejo nos da el patrón de campo, y su intensidad al cuadrado (\$I = |u|^2\$) es la imagen de speckle que analizamos.
    ```python
    N_size = 512  # Tamaño de la imagen
    fwhm_val = 30 # FWHM para el tamaño de grano
    speckle_image = generate_speckle(N=N_size, fwhm=fwhm_val)
    np.save('data/synthetic_speckle.npy', speckle_image)
    
    plt.imshow(speckle_image, cmap='gray')
    plt.title(f'Speckle Sintético (N={N_size}, FWHM={fwhm_val})')
    plt.colorbar()
    plt.savefig('figures/synthetic_speckle_example.png')
    plt.show()
    ```
4.  **Calcular Función de Difusión (ej. a lo largo del eje x):**

    La función de difusión espacial, \$F_D(\Delta x)\$, cuantifica cómo varía la intensidad del speckle entre dos puntos separados por una distancia \$\Delta x\$. Se define como el valor esperado del cuadrado de la diferencia de intensidades: \$F_D(\Delta x) = E[(I(x + \Delta x) - I(x))^2]\$. Para escalas pequeñas, esta función sigue una ley de potencia \$F_D \approx C |\Delta x|^{2H}\$, donde \$H\$ es el exponente de Hurst que caracteriza la rugosidad o persistencia del patrón. A escalas grandes, la función satura debido al tamaño finito de los granos de speckle. El modelo empírico que se ajusta bien a los datos experimentales y que usamos en este proyecto es \$F_D(\Delta x) = 2 \sigma^2 [1 - \exp(-|\Delta x|^{2H} / \ell)]\$, donde \$\sigma^2\$ es la varianza de saturación y \$S = \sqrt{\ell}\$ es el tamaño característico del grano de speckle.
    ```python
    shifts, fd_values = diffusion_function(speckle_image, axis=1, max_shift=speckle_image.shape[1]//4)
    ```
5.  **Análisis Log-Log y Estimación Preliminar de H:**
    ```python
    # Visualizar para estimar H preliminar de la pendiente en la región lineal
    # (ej. tomar los primeros ~10-20% de los puntos para la pendiente)
    num_points_for_slope = len(shifts) // 5 
    log_shifts = np.log(shifts[:num_points_for_slope])
    log_fd = np.log(fd_values[:num_points_for_slope])
    m, c = np.polyfit(log_shifts, log_fd, 1) # Ajuste lineal en log-log
    H_prelim = m / 2
    print(f"Pendiente m ≈ {m:.2f}  => H preliminar ≈ {H_prelim:.2f}")
    
    plot_loglog_fd(shifts, fd_values, H_prelim=H_prelim) # Usar la función de utils
    plt.savefig('figures/loglog_fd_prelim.png')
    # plt.show() # plot_loglog_fd ya no llama a show(), así que se puede llamar aquí si es necesario.
    ```
6.  **Ajuste No Lineal:**
    ```python
    # Estimaciones iniciales (p0)
    # sigma2_guess: fd_values[-1]/2 (la mitad de la saturación observada)
    # ell_guess: (shifts[len(shifts)//2])**(2*H_prelim) # Algo intermedio, o S_guess**2
    # H_guess: H_prelim
    
    sigma2_guess = np.max(fd_values) / 2.0 
    # Para ell, si S es el tamaño de grano, S ~ fwhm. ell = S^2.
    # O, de F_D ≈ C |Δx|^{2H} y F_D ≈ 2σ² |Δx|^{2H} / ℓ para Δx pequeños,
    # C ≈ 2σ²/ℓ.  ℓ ≈ 2σ²/C. C = fd[0]/(shifts[0]**(2H_prelim))
    # ell_guess = (shifts[len(shifts)//10])**(2*H_prelim) # Un valor de Δx donde aún es lineal
    # ell_guess = (fwhm_val)**2 # Aproximación S ~ FWHM
    
    # Intentar con valores razonables basados en H_prelim y la forma de la curva
    # Si H_prelim es ~0.35, S (tamaño de grano) podría ser del orden de fwhm_val
    S_guess = fwhm_val 
    ell_guess = S_guess**2

    p0 = [sigma2_guess, ell_guess, H_prelim]
    print(f"Valores iniciales para el ajuste (sigma2, ell, H): {p0}")

    try:
        params, covariance = curve_fit(fd_model, shifts, fd_values, p0=p0, maxfev=5000)
        sigma2_fit, ell_fit, H_fit = params
        G_fit = 2 * sigma2_fit
        S_fit = np.sqrt(ell_fit)
        print(f"Parámetros ajustados: G={G_fit:.2e}, S={S_fit:.2f}, H={H_fit:.2f}")
        
        # Graficar con el ajuste
        plot_loglog_fd(shifts, fd_values, params_fit=(sigma2_fit, ell_fit, H_fit), H_prelim=H_prelim)
        plt.title(f'Ajuste Modelo F_D (H={H_fit:.2f}, S={S_fit:.1f})')
        plt.savefig('figures/fd_fit_final.png')
        plt.show()
    except RuntimeError:
        print("No se pudo realizar el ajuste. Revisa los valores iniciales (p0) o los datos.")
    except Exception as e:
        print(f"Ocurrió un error durante el ajuste: {e}")
    ```
7.  **Experimentos Paramétricos:**
    El script `main_analysis.py` está ahora configurado por defecto para ejecutar una serie de experimentos variando el FWHM. Los resultados se guardan automáticamente y se generan gráficos de resumen. Consulta <mcfile name="ANALYSIS_DETAILS.md" path="c:\Users\santi\OneDrive - Universidad de Antioquia\Maestría\2025-1\Física Estadística\Speckle\project\ANALYSIS_DETAILS.md"></mcfile> para más detalles sobre la configuración y cómo interpretar los resultados de estos experimentos.

Este README provee una guía para configurar el entorno y ejecutar el análisis del proyecto. Para una comprensión profunda de los análisis realizados y los experimentos, por favor revisa <mcfile name="ANALYSIS_DETAILS.md" path="c:\Users\santi\OneDrive - Universidad de Antioquia\Maestría\2025-1\Física Estadística\Speckle\project\ANALYSIS_DETAILS.md"></mcfile>.