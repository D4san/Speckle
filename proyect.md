# Proyecto numérico: Caracterización de patrones de *speckle* mediante movimiento browniano fraccional

## 1. Introducción

El patrón de *speckle* surge cuando luz coherente se dispersa sobre medios rugosos o turbios, generando interferencias aleatorias. El artículo de Guyot *et al.* (2004) propone describir cuantitativamente estos patrones con la teoría del movimiento browniano fraccional (FBM) y una función de difusión espacial que satura a gran escala fileciteturn0file0.

Este proyecto reproduce y extiende numéricamente los resultados del artículo empleando **Python**. Trabajaremos con imágenes sintéticas de *speckle*, estimaremos parámetros fractales y exploraremos conexiones con la física estadística fuera del equilibrio.

---

## 2. Objetivos

1. **Replicar** la metodología del artículo para una imagen sintética: estimar coeficiente de Hurst \$H\$, varianza de saturación \$G\$ y tamaño característico \$S\$.
2. **Validar** el ajuste de la función de difusión fractal–exponencial.
3. **Explorar experimentos paramétricos** (p.,ej. variación de tamaño de grano) y evaluar cómo cambian \$(H,G,S)\$.
4. **Conectar** la caracterización espacial con fenómenos de difusión anómala en sistemas fuera del equilibrio.

---

## 3. Marco teórico detallado

### 3.1 Estadísticas fundamentales del patrón de *speckle*

Cuando un haz láser coherente se dispersa en un medio rugoso se superponen muchos frentes de onda con fases aleatorias. El campo complejo total puede escribirse como una suma de N fasores. Por el teorema del límite central, la parte real e imaginaria del campo se aproximan por dos gausianas independientes (campo gaussiano circular). Al registrar la intensidad I = |U|²:

* **Speckle totalmente desarrollado (n₀ = 1)** → distribución exponencial P(I) = (1/⟨I⟩) exp( − I/⟨I⟩ ).
* **Promedio multi‑look (n₀ > 1)** → distribución gamma. El parámetro n₀ es ≈ (contraste)⁻².

Esto justifica que el contraste (σᵢ / ⟨I⟩) mida indirectamente cuántos granos de speckle “independientes” contribuyen a cada píxel.

### 3.2 Difusión normal: movimiento browniano clásico

En difusión normal la varianza del desplazamiento crece linealmente con la distancia (o con el tiempo si se tratara de dinámica):

Var\[ B(x + Δx) − B(x) ] = 2 D |Δx|.

Aquí D es el coeficiente de difusión y los incrementos son independientes (sin memoria).

### 3.3 Movimiento browniano fraccional (FBM) y rugosidad

El FBM generaliza el proceso anterior introduciendo memoria larga mediante el exponente de Hurst H ∈ (0, 1):

Var\[ X\_H(x + Δx) − X\_H(x) ] ∝ |Δx|^{2H}.

* H = 0.5 → proceso normal.
* H > 0.5 → incrementos persistentes (superdifusión).
* H < 0.5 → incrementos anticorrelacionados (subdifusión).

Para perfiles 1‑D la dimensión fractal D = 2 − H. En una superficie 2‑D D\_s = 3 − H. En Fourier la densidad espectral sigue S(k) ∝ k^{ − (2H + 1)}.

### 3.4 Función de difusión espacial con saturación

Definimos la función de difusión de la intensidad

F\_D(Δx) = E \[ (I(x + Δx) − I(x))² ].

* En régimen de escalas pequeñas:  F\_D ≈ C |Δx|^{2H}.
* A escalas grandes la varianza no puede crecer sin límite: el tamaño finito del grano introduce un corte (longitud S).

Una forma empírica que ajusta muy bien los datos reales es

F\_D(Δx) = 2 σ² \[ 1 − exp( − |Δx|^{2H} ⁄ ℓ ) ],

donde  σ²  es la varianza de saturación y  S = √ℓ  es el diámetro efectivo de un grano de *speckle*.

### 3.5 Esbozo de un modelo Fokker–Planck efectivo

Si se interpreta el desplazamiento Δx como una “evolución” y la diferencia de intensidad ΔI como variable estocástica, se puede construir una ecuación de Fokker–Planck estacionaria con un coeficiente de difusión que depende de |Δx|^{2H − 1}. La solución estacionaria reproduce la forma exponencial anterior y enlaza el problema con procesos de Ornstein–Uhlenbeck generalizados.

### 3.6 Estimación de parámetros y fuentes de error

* **Pendiente log–log (punto de partida):** m ≈ 2H.
* **Ajuste global:** minimiza la diferencia entre datos y modelo para extraer (σ², ℓ, H).
* **Ruido instrumental:** añade un piso a F\_D; conviene medirlo con obturador cerrado y restarlo.
* **Ventanas de ajuste:** excluir desplazamientos muy pequeños (dominio de correlación de detector) y muy grandes (pocos pares de píxeles, mala estadística).

### 3.7 Conexión con difusión anómala fuera del equilibrio

La relación F\_D ∝ |Δx|^{2H} es formalmente idéntica a la variación temporal de la media cuadrática del desplazamiento en transporte anómalo ⟨r²(t)⟩ ∝ t^{α} con α = 2H. Por tanto:

* H > 0.5 ↔ superdifusión (caminata con saltos largos o memoria persistente).
* H < 0.5 ↔ subdifusión (atrapamiento, trampas energéticas).

Este paralelismo permite aplicar ideas de la física estadística fuera del equilibrio —por ejemplo, funciones de respuesta generalizadas o relaciones de fluctuación‑disipación modificadas— a la microscopia de patrones de *speckle*.

### 3.8 Límites de validez del modelo

1. **Speckle totalmente desarrollado**; de lo contrario la estadística se aleja de la gamma.
2. **Ergodicidad:** se asume que promediar en el espacio equivale a promediar en un ensamble.
3. **Isotropía local:** el medio no introduce ejes preferentes; de estar presentes, habría que analizar por separado las direcciones.
4. **Dispersión óptica lineal:** la teoría ignora efectos de múltiple dispersión de orden alto.
5. **Ruido aditivo gaussiano:** no se modela ruido multiplicativo.

## 4. Metodología numérica Metodología numérica

### 4.1 Generación de patrones sintéticos de *speckle*

1. **Método de fase aleatoria** (FFT):

   1. Crear matriz de fases \$\phi\$ unif. en $\[0,2\pi)\$.
   2. \$U(k\_x,k\_y)=\exp(j\phi)\cdot A(k\_x,k\_y)\$ donde \$A\$ es una envolvente gaussiana (controla tamaño de grano).
   3. Aplicar transformada inversa de Fourier \$u(x,y)=\mathcal F^{-1}{U}\$.
   4. Intensidad \$I=|u|^2\$.
2. **Control de tamaño de grano**: varía la desviación estándar de la envolvente gaussiana.

### 4.2 Cálculo de la función de difusión \$\widehat F\_D(\Delta x)\$

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def diffusion_function(img, axis=0, max_shift=None):
    if max_shift is None:
        max_shift = img.shape[axis]//4
    shifts = np.arange(1, max_shift)
    fd = np.zeros_like(shifts, dtype=float)
    for idx, s in enumerate(shifts):
        if axis==0:
            diff = img[s:, :] - img[:-s, :]
        else:
            diff = img[:, s:] - img[:, :-s]
        fd[idx] = np.mean(diff**2)
    return shifts, fd
```

### 4.3 Ajuste no lineal

```python
from scipy.optimize import curve_fit

def fd_model(delta, sigma2, ell, H):
    return 2*sigma2*(1.0-np.exp(-(delta**(2*H))/ell))

params, _ = curve_fit(fd_model, shifts, fd, p0=[fd[-1]/2, 1000, 0.4])
G = 2*params[0]
S = np.sqrt(params[1])
H = params[2]
```

### 4.4 Análisis log–log para extraer \$H\$ preliminar

Grafique \$\log F\_D\$ vs. \$\log \Delta x\$ en la región lineal. La pendiente \$m\approx 2H\$.

---

## 5. Entorno Python y estructura de proyecto


**Dependencias mínimas**:

* numpy ≥ 1.26
* scipy ≥ 1.12
* matplotlib ≥ 3.8
* jupyterlab
* scikit‑image (opcional para filtros)

```yaml
name: speckle_fbmsim
channels: [conda-forge]
dependencies:
  - python=3.12
  - numpy
  - scipy
  - matplotlib
  - jupyterlab
  - scikit-image
```


## 11. Bibliografía mínima

* S. Guyot, M.‑C. Péron & E. Deléchelle, **Spatial speckle characterization by Brownian motion analysis**, *Phys. Rev. E* **70**, 046618 (2004). fileciteturn0file0
* B. Mandelbrot & J. van Ness, **Fractional Brownian Motions, Fractional Noises and Applications**, *SIAM Rev.* **10**, 422 (1968).
* R. Metzler & J. Klafter, **The restaurant at the end of the random walk**, *J. Phys. A* **37**, R161 (2004).

---

### Apéndice A — Generador rápido de *speckle*

```python
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def generate_speckle(N=1024, fwhm=40):
    # malla de frecuencias
    kx = np.fft.fftfreq(N)
    ky = np.fft.fftfreq(N)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k2 = kx**2 + ky**2
    sigma = fwhm/ (2*np.sqrt(2*np.log(2))*N)
    envelope = np.exp(-k2/(2*sigma**2))

    phase = np.random.uniform(0, 2*np.pi, size=(N, N))
    U = envelope * np.exp(1j*phase)
    u = ifft2(U)
    I = np.abs(fftshift(u))**2
    return I/np.mean(I)
```

### Apéndice B — Cálculo rápido 2D de \$F\_D(r)\$ (isótropo)

```python
from scipy.ndimage import uniform_filter

def radial_fd(img, max_r=None):
    if max_r is None:
        max_r = min(img.shape)//4
    y, x = np.indices(img.shape)
    cy, cx = np.array(img.shape)//2
    r = np.sqrt((x-cx)**2 + (y-cy)**2).astype(int)
    bins = np.arange(1, max_r)
    fd = np.zeros_like(bins, dtype=float)
    for i, rr in enumerate(bins):
        mask = r==rr
        diff = img[mask] - img[cy, cx]
        fd[i] = np.mean(diff**2)
    return bins, fd
```

---

**¡Listo!** Este documento sirve como guía integral. Adapte y expanda según las necesidades del curso y la profundidad de su presentación.
