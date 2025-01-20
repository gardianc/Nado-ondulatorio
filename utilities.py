import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.morphology import remove_small_holes, area_opening, skeletonize
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, savgol_filter
from pyfcd.fcd import calculate_carriers, fcd
from tqdm import tqdm

THRESHOLD = 120
MAX_PIXEL_VALUE = 255
FRAME_PER_SECOND = 60

def cortar(frame, limites):
    min_x, max_x, min_y, max_y = limites
    corte = frame[min_y:max_y, min_x:max_x]
    return corte

def gris(frame):
    im_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return im_gray

def binarize_gray(frame, threshold = THRESHOLD):
    _, im_binary = cv.threshold(frame, threshold, MAX_PIXEL_VALUE, cv.THRESH_BINARY)
    return im_binary

def binarize(frame, threshold = THRESHOLD):
    im_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, im_binary = cv.threshold(im_gray, threshold, MAX_PIXEL_VALUE, cv.THRESH_BINARY)
    return im_binary

def find(frame):
    y, x = np.where(frame==0)
    x_head = np.min(x)
    filas = np.unique(y)
    y_head = int(np.mean(filas))
    return x, y, x_head, y_head

def track_head(vs, limites, head_limit_th=100, SHOW=False):
    # Crea las listas vacias que van a ser nuestras mediciones
    pos_x = []
    pos_y = []
    
    while(vs.isOpened()):
        ret, frame = vs.read()

        if ret:
            frame = cortar(frame, limites)
            first_column = frame[:, 0:1]
            last_row = frame[-2:-1, :]
            is_not_touching_left = np.sum(cv.bitwise_not(binarize(first_column, threshold=head_limit_th))) == 0
            is_not_touching_bottom = np.sum(cv.bitwise_not(binarize(last_row, threshold=head_limit_th))) == 0
            
            if is_not_touching_left and is_not_touching_bottom:
                frame_bn = binarize(frame)
                mask = frame_bn>0
                frame_bn = remove_small_holes(mask,10)*255
                x, y, x_head, y_head = find(frame_bn)
                pos_x.append(x_head)
                pos_y.append(y_head)
                
                if SHOW:
                    frame[y, x] = [0, 255, 0]
                    frame = cv.circle(frame, (x_head, y_head), radius=1, color=(0, 0, 255), thickness=-1)
                    cv.imshow('frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        else:
            break
    cv.destroyAllWindows()
    return np.array(pos_x), np.array(pos_y)

def track_tail(vs, lim, SHOW=False):
    frame_count = int(vs.get(7))

    pos_sk = np.empty(frame_count, dtype=object)
    pos_tail = np.empty(frame_count, dtype=object)
    i=0
    while(vs.isOpened()):
        ret, frame = vs.read()
        if ret:            
            frame = cortar(frame, lim)
            last_column = frame[:, -2:-1]
            first_column = frame[:, 0:1]
            first_row = frame[0:1,:]
            last_row = frame[-2:-1, :]
            is_not_touching_left = (np.sum(cv.bitwise_not(binarize(first_column, threshold=170))) == 0)
            is_not_touching_right = (np.sum(cv.bitwise_not(binarize(last_column, threshold=160))) == 0)
            is_not_touching_top = np.sum(cv.bitwise_not(binarize(first_row, threshold=120))) == 0
            is_not_touching_bottom = np.sum(cv.bitwise_not(binarize(last_row, threshold=120))) == 0
                        
            if is_not_touching_right and is_not_touching_bottom and is_not_touching_top:
                if is_not_touching_left:
                    frame_bn = binarize_tail(guillotina(gris(frame)))
                    sk, tail = skeleton(frame_bn)
                    pos_sk[i] = sk
                    pos_tail[i] = tail
                    if SHOW:
                        frame[tail[0],tail[1]] = [0, 0, 255]
                        cv.imshow('frame', frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print(f'Termina en frame {i}')
                    break
            else:
                if SHOW:
                        # frame[tail[0],tail[1]] = [0, 0, 255]
                    cv.imshow('frame', frame)
                continue
            i+=1
        else:
            break
        
    cv.destroyAllWindows()
    return np.array(pos_sk[:i]), np.array(pos_tail[:i]), i

def binarize_tail(frame, th=THRESHOLD):
    _, im_binary = cv.threshold(
        src=frame,
        thresh=th-20,
        maxval=MAX_PIXEL_VALUE,
        type=cv.THRESH_BINARY)
    
    bool_mask = im_binary > 0 #im_binary contains only 0's and 1's, so this is a mask where all the 0's are false and all the 1's are true.
    # remove_small_holes rellena manchas negras en el fondo
    im_binary = remove_small_holes(bool_mask)*255
    # este proceso rellena manchas blancas dentro del cuerpo del filamento
    y, x = np.where(im_binary==0)
    im_bin = im_binary[min(y)-1:max(y)+2, min(x)-1:max(x)+2]
    im_bin = area_opening(im_bin, 400)
    im_binary[min(y)-1:max(y)+2, min(x)-1:max(x)+2] = im_bin
    return im_binary

def skeleton(frame):
    arr = frame < 255
    sk = skeletonize(arr)*255
    # si sobre una columna que ocupa el esqueleto de la cola hay más de un píxel blanco (debido a globos o ramas)
    # rellena esos puntos con negro y ubica un píxel blanco en la fila "promedio" de esos puntos.
    _, x = np.where(sk!=0)
    for c in np.unique(x):
        r = np.where(sk[:, c]!=0)[0]
        for p in r:
            sk[p, c] = 0
        sk[int(np.mean(r)), c] = 255
    tail = np.where(sk!=0)
    # devuelve la matriz y las coordenadas del esqueleto en la misma
    return sk, tail

def spline_skeleton(tail_x, tail_y):
    df = pd.DataFrame().assign(x=tail_x).assign(y=tail_y)
    df_grouped = df.groupby('x').aggregate('mean').round()
    sp = UnivariateSpline(df_grouped.index.values, df_grouped['y'], s=tail_x.size, k=3)
    xs = np.linspace(df_grouped.index.values.min(), df_grouped.index.values.max(), 100)
    ys = sp(xs)
    return xs, ys

def calibrate_with_head(vs, limites, head_area, head_limit_th=100):
    calibracion = []
    while(vs.isOpened()):
        ret, frame = vs.read()

        if ret:
            frame = cortar(frame, limites)
            first_column = frame[:, 0:1]
            last_row = frame[-2:-1, :]
            is_not_touching_left = np.sum(cv.bitwise_not(binarize(first_column, threshold=head_limit_th))) == 0
            is_not_touching_bottom = np.sum(cv.bitwise_not(binarize(last_row, threshold=head_limit_th))) == 0
            if is_not_touching_left and is_not_touching_bottom:
                frame_bn = binarize(frame)
                mask = frame_bn>0
                frame_bn = remove_small_holes(mask,10)*255
                number_of_pixels = frame_bn[frame_bn==0].size
                px_per_mm2 = number_of_pixels / head_area #number_of_pixels puede ser 0 entonces tomamos px_per_mm2
                calibracion.append(np.sqrt(px_per_mm2))
            else:
                break
    
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv.destroyAllWindows()
    delta_calibration = np.std(calibracion)/np.sqrt(len(calibracion))
    return calibracion, delta_calibration

def guillotina(frame):
    y, x = np.where(frame<=THRESHOLD)
    frame[y, x] = np.max(frame)
    frame -= np.min(frame)
    frame = (frame/np.max(frame))*255
    return frame

def rotate_point(x, y, angle, center_point):
    """ Rotate a point around the origin by a given angle. """
    angle_rad = np.radians(angle)
    x -= center_point[0]
    y -= center_point[1]

    x_new = x * np.cos(angle_rad) - y * np.sin(angle_rad) + center_point[0]
    y_new = x * np.sin(angle_rad) + y * np.cos(angle_rad) + center_point[1]
    return x_new, y_new

def rotate_curve(x, y, angle, center_point):    
    # Rotate points
    x_rotated, y_rotated = [], []
    for xi, yi in zip(x, y):
        x_rot, y_rot = rotate_point(xi, yi, -angle, center_point)
        x_rotated.append(x_rot)
        y_rotated.append(y_rot)
    
    return np.array(x_rotated), np.array(y_rotated)

##### SCHLIEREN IMAGING FUNCTIONS ####

def obtener_deformacion(vs, carriers, start = 0, finish = None, SHOW=False, mask = None):
    frame_count = int(vs.get(7))
    width  = int(vs.get(3)) - (int(vs.get(3)) - mask[3]) - mask[2]  # float `width`
    height = int(vs.get(4)) - (int(vs.get(4)) - mask[1]) - mask[0] # float `height`
    i=0
    if finish is None:
        finish = frame_count
        
    maps = np.zeros((finish - start + 1,height,width))
    while(vs.isOpened()):
        ret, frame = vs.read()
        if ret:
            frame_binarized = binarize(frame,threshold=150)

            if i>=start and i<=finish:
                height_map = fcd(frame_binarized[mask[0]:mask[1],mask[2]:mask[3]], carriers)
                i_frame = i - start
                maps[i_frame] = height_map
            if SHOW:
                cv.imshow('frame', frame_binarized)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            i+=1
        else:
            break
        
    cv.destroyAllWindows()
    
    # Obtengo las dimensiones de las imágenes
    x_len = maps[0].shape[1]
    y_len = maps[0].shape[0]

    #Recorto los bordes para eliminar artefactos por la no-periodicidad exacta del patrón
    maps = [maps[i,int(.1*y_len):int(.9*y_len),int(.1*x_len):int(.9*x_len)] for i in np.arange(len(maps))]

    return maps

def obtener_imagenes_crudas(vs, start = 0, finish = None, SHOW=False, mask = None):
    frame_count = int(vs.get(7))
    width  = int(vs.get(3)) - (int(vs.get(3)) - mask[3]) - mask[2]  # float `width`
    height = int(vs.get(4)) - (int(vs.get(4)) - mask[1]) - mask[0] # float `height`
    i=0
    if finish is None:
        finish = frame_count
        
    maps = np.zeros((finish - start + 1,height,width))
    while(vs.isOpened()):
        ret, frame = vs.read()
        if ret:
            if i>=start and i<=finish:
                i_frame = i - start
                maps[i_frame] = gris(frame[mask[0]:mask[1],mask[2]:mask[3],:])
            if SHOW:
                cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            i+=1
        else:
            break
        
    cv.destroyAllWindows()
    return maps

def make_cmap_norm(deformation_maps):
    max_height = max([abs(deformation_maps[i]).max() for i in np.arange(len(deformation_maps))])
    norm = plt.Normalize(-max_height, max_height)
    cmap = plt.colormaps.get_cmap('seismic')
    return cmap,norm

def load_ref_frame_and_try_mask(vs, mask = None):
    i=0
    ref_frame = None
    while(vs.isOpened()):
        ret, frame = vs.read()
        if ret:
            if mask:
                frame[:,:mask[2]]  = 0
                frame[:, mask[3]:] = 0
                frame[:mask[0], :] = 0
                frame[mask[1]:, :] = 0
            frame = frame[mask[0]:mask[1],mask[2]:mask[3]]
            frame = binarize(frame,threshold=150)
            if i==0:
                ref_frame = frame
            cv.imshow('frame', frame)
            i+=1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cv.destroyAllWindows()
    return ref_frame

def save_deformation_maps(PATH, maps, PX_PER_MM = 1):
    cmap, norm = make_cmap_norm(maps)
    for i_frame in tqdm(np.arange(len(maps))):
        fig = plt.figure()
        plt.imshow(maps[i_frame], aspect='equal', cmap=cmap, norm = norm)
        cbar_term = plt.colorbar(pad=5e-2, shrink=0.43)
        plt.xticks(ticks= plt.xticks()[0][1:-1], labels = [str(np.round(i/(PX_PER_MM),1)) for i in plt.xticks()[0][1:-1]])
        plt.yticks(ticks= plt.yticks()[0][1:-1], labels = [str(np.round(i/(PX_PER_MM),1)) for i in plt.yticks()[0][1:-1]])
        cbar_term.ax.ticklabel_format(axis='y',style='sci',scilimits=(0,2))
        cbar_term.ax.set_ylabel('Deformación [mm]',labelpad=10, fontsize=30)

        plt.ylabel('z [mm]')
        plt.xlabel('x [mm]')
        plt.grid()
        fig.tight_layout()
        plt.savefig(f'{PATH}/{i_frame:03}.tiff')
        plt.cla()
        plt.clf()
        plt.close()

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def propagar_error_omega(lambdas,delta_lambdas, profundidad = None):
    gravedad = 9810 #[mm/s²]
    # profundidad = 62 #[mm]
    gamma = 70e3 # [mN/mm]
    
    if not profundidad:
        w2_func = lambda l,a,b,c: (a*(2*np.pi/l) + b*(2*np.pi/l)**3)*np.tanh((2*np.pi/l)*c)
        
        dw2_dl_func = lambda l,a,b,c: (a*(2*np.pi/l) + b*(2*np.pi/l)**3)*(1/np.cosh(c*(2*np.pi/l))**2)*(c/l**2) \
                                            + (a*(2*np.pi/l)**2 + b*(2*np.pi/l)**4)*np.tanh(c*(2*np.pi/l))
        w2 = w2_func(lambdas, gravedad, gamma, profundidad)
        err_w2 = dw2_dl_func(lambdas, gravedad, gamma, profundidad)*delta_lambdas

    else:
        w2_func = lambda l,a,b: (a*(2*np.pi/l) + b*(2*np.pi/l)**3)
        
        dw2_dl_func = lambda l,a,b: (a*(2*np.pi/l)**2 + b*(2*np.pi/l)**4)


    err_omega = .5*w2**(-1/2)*err_w2
    return err_omega

def get_main_frequency(deformation_maps, y0, x0, DEBUG= False):
    wave_time = [frame[y0, x0] for frame in deformation_maps]
    fr = np.linspace(0, FRAME_PER_SECOND//2, len(wave_time)//2 + 1)
    wave_fft = np.abs(np.fft.rfft(wave_time))**2
    peak_fft = find_peaks(wave_fft)[0]
    max_peak_fft = np.argmax(wave_fft[peak_fft])
    main_freq = fr[peak_fft[max_peak_fft]]

    if DEBUG:
        plt.figure()
        plt.plot(fr,wave_fft, 'o-', color='blue')
        plt.axvline(fr[peak_fft[max_peak_fft]], color='r', ls='--', lw=3, label=f'$f_{{pk}} = {main_freq:.1f}$Hz')
        plt.xlabel('f [Hz]')
        plt.semilogy()
        plt.xlim([1,30])
        plt.legend()
        plt.show()
    
    return main_freq

def get_lambda_w_error(deformation_maps, main_freq, y0, min_x, max_x, n, DEBUG=False, **kwargs_find_peaks):
    xs = np.arange(min_x,max_x)
    int_period = int((main_freq**-1)*FRAME_PER_SECOND)
    number_of_periods = len(deformation_maps)//int_period + 1
    wave_length = np.zeros(number_of_periods)
    debug_wave = []
    debug_peaks = []

    for i,i_frame in enumerate(np.arange(len(deformation_maps) - 1)[20::int_period]):
        wave = deformation_maps[i_frame][y0, min_x:max_x]
        wave = savgol_filter(wave, n, polyorder=5)
        wave_peaks = find_peaks(wave, width=kwargs_find_peaks['width'], rel_height=kwargs_find_peaks['rel_height'])[0]
        
        debug_wave.append(wave)
        debug_peaks.append(wave_peaks)
        
        if len(wave_peaks)>1:
            wave_length[i] = np.diff(xs[wave_peaks]).mean()
            
    if DEBUG:
        plt.figure()
        plt.title(f'{main_freq:.1f}')
        for wave,peaks in zip(debug_wave,debug_peaks):
            plt.plot(xs,wave)
            plt.plot(xs[peaks],wave[peaks],'x',color='r')
        plt.show()
        
    wavelength = np.mean(wave_length[wave_length!=0])
    delta_wavelength = np.std(wave_length[wave_length!=0])/np.sqrt(len(wave_length[wave_length!=0]))
    return wavelength, delta_wavelength

def get_fwhm(fr,fft_vals):
    index_pico = np.argmax(fft_vals[1:]) + 1 # me da la ubicación del máximo y descarto el primer elemento porque es la componente de continua.
    altura_media = (np.max(fft_vals)-np.min(fft_vals))/2 # busco la mitad de la altura del gráfico, para calcular su ancho.
    
    intervalo_s = np.arange(index_pico-10,index_pico + 10) # defino el intervalo de interés, con un poco más de zoom que antes
    spline = UnivariateSpline(fr[intervalo_s], fft_vals[intervalo_s]-altura_media,s=0) # calculo la interpolación
    r1, r2 = spline.roots() # le pido las raíces (que son los puntos a altura media)
    
    frecuencia_spline = np.linspace(np.min(fr[intervalo_s]),np.max(fr[intervalo_s]),100*len(fr[intervalo_s]))
    index_max_spline = np.argmax(spline(frecuencia_spline)) # busco dónde está el máximo
    max_spline = frecuencia_spline[index_max_spline] # y evalúo
    
    print('Máximo interpolado en',np.round(max_spline),'Hz con un ancho a media altura de',r2-r1,'Hz')
    return r2-r1