import numpy as np
import cv2 as cv
import time
from matplotlib.image import imsave


def trackTemplate(frame, template, limites, GRAFICAR=False):
    # Leer frame
    if frame is None:
        return None, None
    
    # Cortar zona del tubo
    min_x, max_x, min_y, max_y = limites
    corte = frame[min_y:max_y, min_x:max_x, :]
    
    # Trackear el template
    res = cv.matchTemplate(corte, template, cv.TM_CCOEFF)
    top_left = cv.minMaxLoc(res)[3]
    
    if GRAFICAR:
        # Dimensiones del template (para dibujar el rectángulo)
        w, h = template.shape[:-1]
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv.rectangle(corte, top_left, bottom_right, 255, 2)
        cv.imshow("corte", corte)
    
    return top_left[0]


class imagenes:
    def __init__(self, vs):
        self.vs = vs
    
    def sin_cortar(self, nombre='pecera'):
        # Saca foto del tubo y guarda la foto en escala de grises
        _, img = self.vs.read()
        frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imsave(f'{nombre}.jpg', frame, cmap='gray')
        # Guarda el nombre de la foto a cortar
        self.a_cortar = nombre

    
    def cortar(self, limites, nombre='corte'):
        # Intenta leer imagen a cortar y si no saca una
        try:
            corte = cv.imread(f'{self.a_cortar}.jpg')
        except AttributeError:
            corte = self.vs
        # Define los limites
        min_x, max_x, min_y, max_y = limites
        # Corta la imagen
        corte = corte[min_y:max_y, min_x:max_x, :]
        # Guarda el corte
        corte = np.mean(corte, axis=2)
        imsave(f'{nombre}.jpg', corte, cmap='gray')
        self.a_filtrar = nombre

    def filtro(self, nombre='filtro'):
        img = cv.imread(f'{self.a_filtrar}.jpg')
        blur = cv.GaussianBlur(img,(5,5),0)
        imsave(f'{nombre}.jpg', blur)





class controlador:
    def __init__(self, vs, template, limites):
        # Constantes de camara
        self.vs = vs
        self.template = template
        self.limites = limites

    def autoTracker(self, ):
        # Crea las listas vacias que van a ser nuestras mediciones
        posiciones = []
        
        fps = self.vs.get(5)
        frame_count = self.vs.get(7)
        tiempo = [f for f in range(1, int(frame_count))]

        while(self.vs.isOpened()):
            ret, frame = self.vs.read()
            if ret:
                posiciones.append(trackTemplate(frame, self.template, self.limites, GRAFICAR=False))
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cv.destroyAllWindows()
        # video.release()
                
        return np.array(tiempo), np.array(posiciones)