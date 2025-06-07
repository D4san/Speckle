import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import curve_fit
from numpy.fft import fft2, ifft2, fftshift, fftfreq
from scipy.ndimage import uniform_filter # Necesario para radial_fd si se usa

def generate_speckle(N=1024, fwhm=40):
    """
    Genera un patrón sintético de speckle con tamaño de grano controlado por FWHM.

    Parámetros:
    - N (int): tamaño de la imagen (N x N píxeles).
    - fwhm (float): ancho completo a mitad de altura del grano de speckle en píxeles.

    Retorna:
    - I_norm (ndarray): imagen de intensidad de speckle normalizada (media = 1).

    Descripción del algoritmo:
    1. Se crea una rejilla de frecuencias (kx, ky) en unidades de ciclos/píxel.
    2. Se calcula el desvío estándar espacial sigma_x a partir de la FWHM:  
       sigma_x = fwhm / (2*sqrt(2*ln2))  # píxeles
    3. Se obtiene el desvío en frecuencia sigma_k usando la relación inversa:
       sigma_k = 1 / (2*pi*sigma_x)      # ciclos/píxel
    4. Se define la envolvente gaussiana en k-espacio:
       E(k) = exp(- (kx^2 + ky^2) / (2*sigma_k^2) )
    5. Se genera fase aleatoria uniforme en [0,2π).  
    6. Se construye el campo en k-espacio U = sqrt(E) * exp(i*phase).  
    7. Se aplica IFFT para obtener el campo u(x) en espacio real.  
    8. Se calcula la intensidad I = |u|^2 y se normaliza para que su media sea 1.
    """
    # 1. Malla de frecuencias (ciclos/píxel)
    kx = fftfreq(N).reshape(N, 1)
    ky = fftfreq(N).reshape(1, N)
    k2 = kx**2 + ky**2

    # 2. Ancho espacial deseado (sigma_x)
    sigma_x = fwhm / (2 * np.sqrt(2 * np.log(2)))
    # 3. Ancho en frecuencia (sigma_k), inversamente proporcional a sigma_x
    sigma_k = 1.0 / (2 * np.pi * sigma_x)

    # 4. Envolvente gaussiana en k-espacio
    envelope = np.exp(-k2 / (2 * sigma_k**2))

    # 5. Fase aleatoria
    phase = np.random.uniform(0, 2 * np.pi, size=(N, N))

    # 6. Campo complejo en k-espacio
    U = envelope * np.exp(1j * phase)

    # 7. Transformada inversa para obtener el campo en espacio real
    u = ifft2(U)

    # 8. Intensidad y normalización
    I = np.abs(u)**2
    I_norm = I / np.mean(I)
    return I_norm

def diffusion_function(img, axis=0, max_shift=None):
    if max_shift is None:
        max_shift = img.shape[axis] // 4
    shifts = np.arange(1, max_shift)
    fd = np.zeros_like(shifts, dtype=float)
    for idx, s in enumerate(shifts):
        if axis == 0: # Desplazamiento vertical
            diff = img[s:, :] - img[:-s, :]
        else: # Desplazamiento horizontal
            diff = img[:, s:] - img[:, :-s]
        fd[idx] = np.mean(diff**2)
    return shifts, fd

def fd_model(delta, sigma2, ell, H):
    # delta: retardo espacial (Δx)
    # sigma2: varianza de saturación (σ²)
    # ell: parámetro relacionado con el tamaño del grano (ℓ = S²)
    # H: exponente de Hurst
    return 2 * sigma2 * (1.0 - np.exp(-(delta**(2 * H)) / ell))

def radial_fd(img, max_r=None):
    """Calcula la función de difusión radial F_D(r) para una imagen isotrópica."""
    if max_r is None:
        max_r = min(img.shape) // 4
    
    h, w = img.shape
    cy, cx = h // 2, w // 2
    
    y_indices, x_indices = np.indices(img.shape)
    
    # Calcular la distancia r desde el centro para cada píxel
    r_map = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2).astype(int)
    
    bins = np.arange(1, max_r + 1) # +1 para incluir max_r
    fd = np.zeros(len(bins), dtype=float)
    counts = np.zeros(len(bins), dtype=int)

    # Esta implementación de F_D(r) es E[(I(pixel_a) - I(pixel_b))^2] donde r = |pixel_a - pixel_b|
    # La versión en el markdown parece calcular E[(I(pixel_en_mascara_r) - I(centro))^2]
    # Lo cual es diferente. Implementaré la versión más común para F_D(r) isotrópica
    # que promedia sobre todos los pares de píxeles separados por una distancia r.
    # Sin embargo, para ser fiel al apéndice B, usaré la referencia al centro.
    # Nota: La implementación del apéndice B es una simplificación y puede no ser la F_D estándar.
    
    # Implementación según Apéndice B:
    # diff = img[mask] - img[cy, cx] -> diferencia con el píxel central
    # Esto no es la F_D(r) = E[(I(x+r) - I(x))^2] promediada sobre x y orientaciones de r.
    # Es más bien una medida de varianza radial respecto al centro.
    # Para una F_D(r) isotrópica más general, se usaría autocorrelación o promediar pares.
    
    # Siguiendo el Apéndice B:
    img_mean_center_pix = img[cy, cx] # Valor del píxel central
    for i, rr_bin_val in enumerate(bins):
        mask = (r_map == rr_bin_val)
        if np.any(mask):
            # diff = img[mask] - img_mean_center_pix # Diferencia con el píxel central
            # fd[i] = np.mean(diff**2)
            # La fórmula F_D(Δx) = E[(I(x + Δx) − I(x))²] sugiere diferencias entre píxeles desplazados.
            # La implementación de radial_fd en el apéndice es peculiar.
            # Una F_D(r) más estándar implicaría promediar (I(p1) - I(p2))^2 para todos los pares (p1, p2) con dist(p1,p2) = r.
            # Por simplicidad y para seguir el apéndice, se mantiene la referencia al centro, 
            # pero esto debe tenerse en cuenta al interpretar los resultados.
            pixel_values_at_r = img[mask]
            if pixel_values_at_r.size > 0:
                 # Para cada pixel en el anillo r, tomamos su diferencia con el pixel central
                diff_sq = (pixel_values_at_r - img_mean_center_pix)**2
                fd[i] = np.mean(diff_sq)
            else:
                fd[i] = np.nan # No hay píxeles a esta distancia exacta
        else:
            fd[i] = np.nan
            
    # Filtrar NaNs si es necesario y los bins correspondientes
    valid_indices = ~np.isnan(fd)
    return bins[valid_indices], fd[valid_indices]