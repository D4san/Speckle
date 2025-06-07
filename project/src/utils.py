# Funciones utilitarias (plotting, data i/o, etc.)
import matplotlib.pyplot as plt
import numpy as np

def plot_loglog_fd(shifts, fd, params_fit=None, H_prelim=None):
    """Grafica F_D vs Δx en escala log-log y el ajuste si se provee."""
    plt.figure(figsize=(8, 6))
    plt.loglog(shifts, fd, 'o', label='Datos experimentales F_D(Δx)')

    if params_fit is not None:
        sigma2_fit, ell_fit, H_fit = params_fit
        # Asegurarse que fd_model está disponible, usualmente importada de speckle.py
        # from ..src.speckle import fd_model # Si utils.py está en un subdirectorio y se ejecuta como módulo
        # O pasar fd_model como argumento si es más limpio
        # Para este script, asumimos que fd_model será importada en el notebook donde se use plot_loglog_fd
        # from speckle import fd_model # Si src está en PYTHONPATH o el notebook está en project/
        
        # Para que este módulo sea autocontenido o más robusto, fd_model debería ser importada aquí
        # o pasada como argumento. Por ahora, se asume que el notebook se encargará.
        # Si se quiere que utils.py sea independiente:
        # def fd_model_local(delta, sigma2, ell, H): # Copia local o importación
        #     return 2 * sigma2 * (1.0 - np.exp(-(delta**(2 * H)) / ell))
        # fd_fitted = fd_model_local(shifts, sigma2_fit, ell_fit, H_fit)
        
        # Suponiendo que fd_model será importada en el contexto de uso (notebook)
        # Esta es una dependencia implícita que debe manejarse en el notebook.
        # Para evitar esto, la función fd_model podría ser importada directamente aquí:
        try:
            from .speckle import fd_model # Intento de importación relativa
        except ImportError:
            # Fallback si se ejecuta utils.py directamente o speckle.py no está en el path relativo esperado
            # Esto es común si src/ no es tratado como un paquete.
            # En un notebook en project/notebooks/, se usaría from src.speckle import fd_model
            # Por ahora, dejaremos que el error ocurra si fd_model no está en el scope del llamador.
            # Una solución más robusta es pasar la función modelo como argumento.
            print("Advertencia: fd_model no se pudo importar en utils.py. Asegúrate de que esté disponible en el scope de llamada.")
            # Alternativamente, definirla aquí si es simple y no cambia:
            def fd_model(delta, sigma2, ell, H): # Definición local como fallback
                 return 2 * sigma2 * (1.0 - np.exp(-(delta**(2 * H)) / ell))
            fd_fitted = fd_model(shifts, sigma2_fit, ell_fit, H_fit)
        else:
            fd_fitted = fd_model(shifts, sigma2_fit, ell_fit, H_fit)
            
        plt.loglog(shifts, fd_fitted, '-', label=f'Ajuste: H={H_fit:.2f}, S={np.sqrt(ell_fit):.1f}, G={2*sigma2_fit:.2e}')

    if H_prelim is not None:
        if len(shifts) > 0 and len(fd) > 0 and shifts[0] > 0: # Evitar log(0) o shifts[0]**(...) si es 0
            C_guideline = fd[0] / (shifts[0]**(2*H_prelim))
            guideline = C_guideline * shifts**(2*H_prelim)
            plt.loglog(shifts, guideline, '--', label=f'Guía pendiente 2H (H={H_prelim:.2f})')

    plt.xlabel('log(Δx)')
    plt.ylabel('log(F_D)')
    plt.title('Función de Difusión vs. Desplazamiento (Log-Log)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    # plt.show() # Es mejor llamar a plt.show() en el notebook, no en la función de ploteo

# Ejemplo de cómo podría usarse fd_model desde speckle.py en este archivo si fuera necesario
# try:
#     from .speckle import fd_model
# except ImportError:
#     # Este es un apaño común si los módulos no están instalados como paquete
#     # y se ejecutan scripts individualmente. Mejor manejar imports en el notebook.
#     pass