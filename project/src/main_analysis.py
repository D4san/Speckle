# main_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# Import functions from local modules (speckle.py and utils.py)
# Assuming main_analysis.py is in the src/ directory, and speckle.py and utils.py are also in src/
import pandas as pd
import seaborn as sns
from speckle import generate_speckle, diffusion_function, fd_model # radial_fd is also available
from utils import plot_loglog_fd

def run_analysis(N_size=512, fwhm_val=30, experiment_label="single", save_figures=True, show_plots=True):
    """
    Runs the full speckle analysis: generation, F_D calculation, H estimation, and model fitting.

    Args:
        N_size (int): Size of the synthetic speckle image (N x N).
        fwhm_val (int): FWHM of the Gaussian envelope in k-space, controls speckle grain size.
        experiment_label (str): A label for the experiment, used in filenames.
        save_figures (bool): If True, saves generated plots to the 'figures' directory.
        show_plots (bool): If True, displays plots using plt.show().
    
    Returns:
        dict: A dictionary containing the results (N_size, fwhm_val, G_fit, S_fit, H_fit)
              Returns None if analysis fails.
    """
    print(f"--- Iniciando Análisis de Speckle (N={N_size}, FWHM={fwhm_val}) ---")

    # --- 1. Crear directorios si no existen ---
    base_data_dir = '../data'
    base_figures_dir = '../figures'
    experiment_figures_dir = os.path.join(base_figures_dir, 'experiments', experiment_label)
    
    output_dirs = [base_data_dir, base_figures_dir, os.path.join(base_figures_dir, 'experiments'), experiment_figures_dir]
    for D_dir in output_dirs:
        if not os.path.exists(D_dir):
            print(f"Creando directorio: {D_dir}")
            os.makedirs(D_dir)

    # --- 2. Generar Patrón de Speckle ---
    print("Generando patrón de speckle...")
    speckle_image = generate_speckle(N=N_size, fwhm=fwhm_val)
    speckle_data_filename = os.path.join(base_data_dir, f'synthetic_speckle_N{N_size}_FWHM{fwhm_val}_{experiment_label}.npy')
    np.save(speckle_data_filename, speckle_image)
    print(f"Patrón de speckle guardado en '{speckle_data_filename}'")

    if show_plots or save_figures:
        plt.figure(figsize=(7, 6))
        plt.imshow(speckle_image, cmap='gray')
        plt.title(f'Speckle: N={N_size}, FWHM={fwhm_val} ({experiment_label})')
        plt.colorbar()
        if save_figures:
            fig_path = os.path.join(experiment_figures_dir, f'speckle_N{N_size}_FWHM{fwhm_val}.png')
            plt.savefig(fig_path)
            print(f"Figura del speckle guardada en '{fig_path}'")
        if show_plots:
            plt.show()
        plt.close()

    # --- 3. Calcular Función de Difusión (ej. a lo largo del eje x) ---
    print("Calculando función de difusión F_D(Δx)...")
    # Usamos axis=1 para el eje x (columnas), axis=0 para el eje y (filas)
    max_s = speckle_image.shape[1] // 4 # Un cuarto del tamaño de la imagen
    shifts, fd_values = diffusion_function(speckle_image, axis=1, max_shift=max_s)
    
    # Filtrar valores de fd no válidos si los hubiera (ej. NaN o inf)
    valid_indices = np.isfinite(fd_values) & np.isfinite(shifts) & (shifts > 0) & (fd_values > 0)
    shifts = shifts[valid_indices]
    fd_values = fd_values[valid_indices]

    if len(shifts) == 0:
        print("Error: No se pudieron obtener valores válidos para la función de difusión.")
        return {"N_size": N_size, "fwhm_val": fwhm_val, "G_fit": np.nan, "S_fit": np.nan, "H_fit": np.nan, "error": "No valid FD values"}

    # --- 4. Análisis Log-Log y Estimación Preliminar de H ---
    print("Realizando análisis log-log para H preliminar...")
    # Tomar una porción inicial para el ajuste lineal (e.g., primeros 10-20% de los puntos)
    # Asegurarse de que haya suficientes puntos para el ajuste
    num_points_for_slope = max(2, len(shifts) // 5)
    if len(shifts) < num_points_for_slope:
        num_points_for_slope = len(shifts)

    if num_points_for_slope >= 2: # np.polyfit necesita al menos 2 puntos
        log_shifts_prelim = np.log(shifts[:num_points_for_slope])
        log_fd_prelim = np.log(fd_values[:num_points_for_slope])
        
        # Evitar problemas si hay NaNs o Infs después del log
        valid_log_indices = np.isfinite(log_shifts_prelim) & np.isfinite(log_fd_prelim)
        log_shifts_prelim = log_shifts_prelim[valid_log_indices]
        log_fd_prelim = log_fd_prelim[valid_log_indices]

        if len(log_shifts_prelim) >= 2:
            m, c = np.polyfit(log_shifts_prelim, log_fd_prelim, 1)
            H_prelim = m / 2
            print(f"  Pendiente log-log (m) ≈ {m:.3f} => H preliminar ≈ {H_prelim:.3f}")
        else:
            print("  No hay suficientes puntos válidos para el ajuste log-log preliminar. Usando H_prelim=0.5 por defecto.")
            H_prelim = 0.5 # Valor por defecto si no se puede estimar
    else:
        print("  No hay suficientes puntos para el ajuste log-log preliminar. Usando H_prelim=0.5 por defecto.")
        H_prelim = 0.5 # Valor por defecto

    if show_plots or save_figures:
        plot_loglog_fd(shifts, fd_values, H_prelim=H_prelim)
        plt.title(f'Log-Log F_D: N={N_size}, FWHM={fwhm_val} (H_prelim ≈ {H_prelim:.2f}) ({experiment_label})')
        if save_figures:
            fig_path = os.path.join(experiment_figures_dir, f'loglog_fd_prelim_N{N_size}_FWHM{fwhm_val}.png')
            plt.savefig(fig_path)
            print(f"Figura log-log preliminar guardada en '{fig_path}'")
        if show_plots:
            plt.show()
        plt.close()

    # --- 5. Ajuste No Lineal del Modelo F_D ---
    print("Realizando ajuste no lineal del modelo F_D(Δx)...")
    sigma2_guess = np.max(fd_values) / 2.0 
    S_guess = fwhm_val # Aproximación: tamaño de grano S ~ FWHM
    ell_guess = S_guess**2
    p0 = [sigma2_guess, ell_guess, H_prelim if H_prelim > 0 and H_prelim < 1 else 0.5] # Asegurar H_prelim válido
    
    print(f"  Valores iniciales para el ajuste (sigma2, ell, H): [{p0[0]:.2e}, {p0[1]:.2f}, {p0[2]:.2f}]")

    try:
        params, covariance = curve_fit(fd_model, shifts, fd_values, p0=p0, maxfev=10000, bounds=([0, 0, 0.01], [np.inf, np.inf, 0.99]))
        sigma2_fit, ell_fit, H_fit = params
        G_fit = 2 * sigma2_fit
        S_fit = np.sqrt(ell_fit)
        print(f"  Parámetros ajustados: G={G_fit:.3e}, S={S_fit:.2f}, H={H_fit:.3f}")

        if show_plots or save_figures:
            plot_loglog_fd(shifts, fd_values, params_fit=(sigma2_fit, ell_fit, H_fit), H_prelim=H_prelim)
            plt.title(f'Ajuste F_D: N={N_size}, FWHM={fwhm_val} (H={H_fit:.2f}) ({experiment_label})')
            if save_figures:
                fig_path = os.path.join(experiment_figures_dir, f'fd_fit_N{N_size}_FWHM{fwhm_val}.png')
                plt.savefig(fig_path)
                print(f"Figura del ajuste final guardada en '{fig_path}'")
            if show_plots:
                plt.show()
            plt.close()
        
        print("--- Análisis Completado ---")
        return {"N_size": N_size, "fwhm_val": fwhm_val, "G_fit": G_fit, "S_fit": S_fit, "H_fit": H_fit, "error": None}

    except RuntimeError as e:
        print(f"  Error en el ajuste no lineal: {e}. Revisa los valores iniciales o los datos.")
        return {"N_size": N_size, "fwhm_val": fwhm_val, "G_fit": np.nan, "S_fit": np.nan, "H_fit": np.nan, "error": str(e)}
    except Exception as e:
        print(f"  Ocurrió una excepción durante el ajuste: {e}")
        return {"N_size": N_size, "fwhm_val": fwhm_val, "G_fit": np.nan, "S_fit": np.nan, "H_fit": np.nan, "error": str(e)}
    
    # Fallback if something unexpected happens before try-except for fitting
    print("--- Análisis Incompleto (Error antes del ajuste) ---")
    return {"N_size": N_size, "fwhm_val": fwhm_val, "G_fit": np.nan, "S_fit": np.nan, "H_fit": np.nan, "error": "Analysis incomplete before fit"}

if __name__ == '__main__':
    # --- Configuración General para Experimentos ---
    image_size_default = 512
    # Lista de valores FWHM para experimentar
    fwhm_values_experiment = [2, 3, 5, 10, 15, 25, 35]
    # Lista de tamaños de imagen para experimentar (opcional, podríamos fijarlo por ahora)
    N_size_values_experiment = [128, 256, 512]

    all_results = []

    print("\n--- Iniciando Serie de Experimentos Variando FWHM ---")
    experiment_set_label = "fwhm_variation"
    for fwhm_exp in fwhm_values_experiment:
        print(f"\n*** Experimento: N={image_size_default}, FWHM={fwhm_exp} ***")
        # Etiqueta única para este conjunto de parámetros
        current_exp_label = f"N{image_size_default}_FWHM{fwhm_exp}"
        results = run_analysis(N_size=image_size_default, 
                                 fwhm_val=fwhm_exp, 
                                 experiment_label=current_exp_label, 
                                 save_figures=True, 
                                 show_plots=False) # No mostrar plots individuales durante la serie
        if results:
            all_results.append(results)
    
    # Convertir resultados a DataFrame y guardar
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Crear directorio para resultados CSV si no existe
        results_csv_dir = '../data/experiment_results'
        if not os.path.exists(results_csv_dir):
            os.makedirs(results_csv_dir)
        csv_path = os.path.join(results_csv_dir, f'summary_results_{experiment_set_label}_N{image_size_default}.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\nResultados de los experimentos guardados en: {csv_path}")

        # --- Generar Gráficos de Resumen de Experimentos ---
        print("\nGenerando gráficos de resumen de experimentos...")
        summary_fig_dir = '../figures/experiment_summary_plots'
        if not os.path.exists(summary_fig_dir):
            os.makedirs(summary_fig_dir)

        # Gráfico 1: H_fit vs fwhm_val
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=results_df, x='fwhm_val', y='H_fit', marker='o')
        plt.title(f'H Estimado vs. FWHM (N={image_size_default})')
        plt.xlabel('FWHM del Filtro (píxeles)')
        plt.ylabel('H Estimado (Ajuste)')
        plt.ylim(0,1)
        plt.grid(True)
        h_vs_fwhm_path = os.path.join(summary_fig_dir, f'H_vs_FWHM_{experiment_set_label}_N{image_size_default}.png')
        plt.savefig(h_vs_fwhm_path)
        print(f"Gráfico H vs FWHM guardado en: {h_vs_fwhm_path}")
        plt.show()
        plt.close()

        # Gráfico 2: S_fit vs fwhm_val
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=results_df, x='fwhm_val', y='S_fit', marker='o', color='green')
        plt.plot(results_df['fwhm_val'], results_df['fwhm_val'], linestyle='--', color='gray', label='Ideal S = FWHM') # Línea de referencia
        plt.title(f'Tamaño de Grano Estimado (S) vs. FWHM (N={image_size_default})')
        plt.xlabel('FWHM del Filtro (píxeles)')
        plt.ylabel('S Estimado (Ajuste, píxeles)')
        plt.ylim(0, max(results_df['fwhm_val'])*1.3)
        plt.legend()
        plt.grid(True)
        s_vs_fwhm_path = os.path.join(summary_fig_dir, f'S_vs_FWHM_{experiment_set_label}_N{image_size_default}.png')
        plt.savefig(s_vs_fwhm_path)
        print(f"Gráfico S vs FWHM guardado en: {s_vs_fwhm_path}")
        plt.show()
        plt.close()

        # Gráfico 3: G_fit vs fwhm_val
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=results_df, x='fwhm_val', y='G_fit', marker='o', color='purple')
        plt.title(f'Parámetro G Estimado vs. FWHM (N={image_size_default})')
        plt.xlabel('FWHM del Filtro (píxeles)')
        plt.ylabel('G Estimado (Ajuste)')
        plt.ylim(0, max(results_df['G_fit'])*1.3)
        plt.grid(True)
        g_vs_fwhm_path = os.path.join(summary_fig_dir, f'G_vs_FWHM_{experiment_set_label}_N{image_size_default}.png')
        plt.savefig(g_vs_fwhm_path)
        print(f"Gráfico G vs FWHM guardado en: {g_vs_fwhm_path}")
        plt.show()
        plt.close()

    print("\n--- Serie de Experimentos Completada ---")

    # Para ejecutar un solo análisis con plots visibles (como antes):
    # print("\n--- Ejecutando un solo análisis de prueba ---")
    # run_analysis(N_size=256, fwhm_val=30, experiment_label="test_single", save_figures=True, show_plots=True)

# Ejemplo de cómo podrías correr múltiples experimentos (descomentar para probar)
    # print("\n--- Iniciando Serie de Experimentos ---")
    # fwhm_values = [10, 20, 40]
    # for fwhm_exp in fwhm_values:
    #     print(f"\n*** Experimento con FWHM = {fwhm_exp} ***")
    #     run_analysis(N_size=256, fwhm_val=fwhm_exp, experiment_label=f"FWHM{fwhm_exp}_legacy", save_figures=True, show_plots=False) # No mostrar plots intermedios
    # print("\n--- Serie de Experimentos Completada ---")

if __name__ == '__main__':
    # --- Parámetros para el análisis ---
    image_size = 256      # Tamaño de la imagen (NxN)
    fwhm_parameter = 20   # Parámetro FWHM para el tamaño de grano
    
    # Ejecutar el análisis
    run_analysis(N_size=image_size, fwhm_val=fwhm_parameter, save_figures=True, show_plots=True)
    
    # Ejemplo de cómo podrías correr múltiples experimentos (descomentar para probar)
    # print("\n--- Iniciando Serie de Experimentos ---")
    # fwhm_values = [10, 20, 40]
    # for fwhm_exp in fwhm_values:
    #     print(f"\n*** Experimento con FWHM = {fwhm_exp} ***")
    #     run_analysis(N_size=256, fwhm_val=fwhm_exp, save_figures=True, show_plots=False) # No mostrar plots intermedios
    # print("\n--- Serie de Experimentos Completada ---")