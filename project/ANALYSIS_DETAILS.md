# Detalles del Análisis Experimental de Patrones de Speckle

Este documento describe los experimentos realizados para caracterizar patrones de speckle sintéticos, los parámetros variados, y cómo interpretar los resultados generados por el script `src/main_analysis.py`.

## Objetivos del Análisis

El objetivo principal es entender cómo los parámetros de entrada utilizados para generar el speckle (principalmente el FWHM del filtro en el espacio k, que controla el tamaño de grano) se relacionan con los parámetros físicos estimados a partir de la función de difusión (FD) utilizando el modelo de Movimiento Browniano Fraccional (FBM).

Los parámetros clave estimados son:

*   **H (Exponente de Hurst):** Relacionado con la rugosidad o persistencia del patrón.
*   **S (Tamaño de Grano Efectivo):** Estimación del tamaño característico de los speckles.
*   **G (Factor de Amplitud):** Relacionado con la varianza de la intensidad del speckle (G = 2σ²).

## Serie de Experimentos: Variación del FWHM

El script `src/main_analysis.py` está configurado para ejecutar una serie de experimentos donde se varía el parámetro `fwhm_val` (Full Width at Half Maximum) del filtro Gaussiano aplicado en el espacio k durante la generación del speckle. El tamaño de la imagen (`N_size`) se mantiene constante para esta serie de experimentos (actualmente N=256).

Los valores de `fwhm_val` utilizados en la serie actual son: `[10, 15, 20, 25, 30, 35, 40, 50]` píxeles.

### Proceso por Experimento

Para cada valor de `fwhm_val` en la serie:

1.  **Generación de Speckle:** Se genera una imagen de speckle de `N_size x N_size` utilizando el `fwhm_val` actual.
    *   La imagen de speckle se guarda en: `data/synthetic_speckle_N<N_size>_FWHM<fwhm_val>_<experiment_label>.npy`
    *   Una visualización de la imagen se guarda en: `figures/experiments/<experiment_label>/speckle_N<N_size>_FWHM<fwhm_val>.png`
2.  **Cálculo de la Función de Difusión (FD):** Se calcula la FD a lo largo del eje x.
3.  **Estimación Preliminar de H:** Se realiza un ajuste log-log en la porción inicial de la FD para obtener una `H_prelim`.
    *   El gráfico log-log con la línea de `H_prelim` se guarda en: `figures/experiments/<experiment_label>/loglog_fd_prelim_N<N_size>_FWHM<fwhm_val>.png`
4.  **Ajuste del Modelo FBM:** Se ajusta el modelo teórico de la FD (basado en FBM) a los datos de la FD calculada para estimar `sigma2_fit`, `ell_fit` (S²), y `H_fit`.
    *   El gráfico log-log con el ajuste del modelo se guarda en: `figures/experiments/<experiment_label>/fd_fit_N<N_size>_FWHM<fwhm_val>.png`
5.  **Almacenamiento de Resultados:** Los parámetros de entrada (`N_size`, `fwhm_val`) y los resultados del ajuste (`G_fit`, `S_fit`, `H_fit`) se almacenan.

### Resultados Agrupados

Una vez completados todos los experimentos de la serie:

1.  **Archivo CSV:** Todos los resultados individuales se compilan en un archivo CSV.
    *   Ubicación: `data/experiment_results/summary_results_fwhm_variation_N<N_size>.csv`
    *   Columnas: `N_size`, `fwhm_val`, `G_fit`, `S_fit`, `H_fit`, `error` (si lo hubo).
2.  **Gráficos de Resumen:** Se generan gráficos para visualizar las tendencias de los parámetros ajustados en función del `fwhm_val`.
    *   Ubicación: `figures/experiment_summary_plots/`
    *   **H vs. FWHM:** `H_vs_FWHM_fwhm_variation_N<N_size>.png`
    *   **S vs. FWHM:** `S_vs_FWHM_fwhm_variation_N<N_size>.png` (incluye una línea S=FWHM como referencia)
    *   **G vs. FWHM:** `G_vs_FWHM_fwhm_variation_N<N_size>.png`

## Interpretación de los Resultados

*   **H vs. FWHM:** Se espera que el exponente de Hurst (H) pueda mostrar alguna dependencia con el FWHM, aunque teóricamente para un speckle completamente desarrollado y un modelo FBM simple, H podría ser relativamente constante o cercano a 0.5 (para movimientos Brownianos estándar) o valores menores si hay anti-persistencia. Observar cómo varía H puede dar indicios sobre la aplicabilidad del modelo o la naturaleza del patrón de speckle generado bajo diferentes condiciones de filtrado.
*   **S vs. FWHM:** El tamaño de grano estimado (S) debería, idealmente, correlacionarse directamente con el `fwhm_val` de entrada, ya que el FWHM controla el tamaño de los granos de speckle. El gráfico `S_vs_FWHM` incluye una línea `S = FWHM` para una comparación directa. Desviaciones de esta línea pueden indicar limitaciones del modelo de ajuste, la definición de S, o efectos del tamaño finito de la imagen.
*   **G vs. FWHM:** El parámetro G (relacionado con la varianza) podría cambiar si el FWHM afecta la distribución general de intensidades o el contraste del patrón de speckle. 

El archivo CSV (`summary_results_...csv`) permite un análisis cuantitativo más detallado de estas relaciones.

## Futuros Experimentos Sugeridos

1.  **Variación de `N_size`:** Mantener `fwhm_val` constante y variar el tamaño de la imagen (`N_size`) para evaluar efectos de tamaño finito en los parámetros estimados.
2.  **Análisis Bidimensional:** Utilizar la función `radial_fd` (disponible en `src.speckle`) para calcular la función de difusión promediada radialmente y comparar los resultados con el análisis unidireccional.
3.  **Diferentes Ejes de Análisis:** Comparar los resultados de la FD calculada a lo largo del eje x vs. el eje y para verificar isotropía.
4.  **Introducción de Ruido:** Añadir diferentes niveles de ruido a las imágenes de speckle sintéticas y observar cómo afecta la estimación de los parámetros.
5.  **Análisis de Speckle Experimental:** Aplicar la metodología a imágenes de speckle reales obtenidas experimentalmente.

Estos experimentos permitirían una caracterización más robusta y una comprensión más profunda de la física de los patrones de speckle y la aplicabilidad del modelo FBM.