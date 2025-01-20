import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.morphology import remove_small_holes, area_opening, skeletonize
import pandas as pd
from matplotlib import rcParams
from scipy.interpolate import UnivariateSpline

THRESHOLD = 120
MAX_PIXEL_VALUE = 255

def cortar(frame, limites):
    min_x, max_x, min_y, max_y = limites
    corte = frame[min_y:max_y, min_x:max_x]
    return corte

def gris(frame):
    im_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return im_gray


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
                    print('touches left')
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
    return np.array(pos_sk[:i]), np.array(pos_tail[:i])

def binarize_tail(frame):
    _, im_binary = cv.threshold(
        src=frame,
        thresh=100,
        maxval=MAX_PIXEL_VALUE,
        type=cv.THRESH_BINARY)
    
    bool_mask = im_binary > 0 #im_binary contains only 0's and 1's, so this is a mask where all the 0's are false and all the 1's are true.
    im_binary = remove_small_holes(bool_mask)*255
    return im_binary

def skeleton(frame):
    arr = frame < 255 #Convert frame to binary mask of 1's and 0's
    sk = skeletonize(arr)*255 #We apply skeletonize to boolean mask and upscale 1's to 255 (white)
    _, x = np.where(sk!=0)
    # esto cierra los globitos
    for c in np.unique(x):
        r = np.where(sk[:, c]!=0)[0]
        for p in r:
            sk[p, c] = 0
        sk[int(np.mean(r)), c] = 255
    tail = np.where(sk!=0)
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
