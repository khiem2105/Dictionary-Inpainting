import numpy as np
import sklearn.linear_model as lm
import pdb
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb
from time import sleep



### Dimension du patch x-h:x+h,y-h:y+h
H = 10
### Valeur fictive pour les pixels absents
DEAD = -100

def read_img(img):
    """ lecture d'une image """
    im = plt.imread(img)
    if im.shape[2]==4:
        im = im[:,:,:3]
    if im.max()>200:
        im = im/255.
    return rgb_to_hsv(im)-0.5


def show(im,fig= None):
    """ affiche une image """
    im = im.copy()
    im[im<=DEAD]=-0.5
    if fig is None:
        plt.figure()
        fig = plt.imshow(hsv_to_rgb(im+0.5))
    fig.set_data(hsv_to_rgb(im+0.5))
    plt.draw()
    plt.pause(0.001)
    return fig

def get_patch(i,j,im,h=H):
    """ retourne un patch centre en i,j """
    return im[(i-h):(i+h+1),(j-h):(j+h+1)]

def patch2vec(patch):
    """ transformation d'un patch en vecteur """
    return patch.reshape(-1)

def vec2patch(X):
    """ transformation d'un vecteur en patch image"""
    h = int(np.sqrt((X.shape[0]//3)))
    return X.reshape(h,h,3)


def noise_patch(patch,prc=0.2):
    npatch = patch.copy().reshape(-1,3)
    height,width = patch.shape[:2]
    nb=int(prc*height*width)
    npatch[np.random.randint(0,height*width,nb),:]=DEAD
    return npatch.reshape(height,width,3)


def remove_band(im,i,j,height=-1,width=-1):
    """ enleve une bande """
    im = im.copy()
    if width <0:
        width = im.shape[1]-j
    if height < 0:
        height = im.shape[0]-i
    im[i:(i+height),j:(j+width)]=DEAD
    return im




def inside(i,j,im,h=H):
    """ test si un patch est valide dans l'image """
    return i-h >=0 and j-h >=0 and i+h+1<=im.shape[0] and j+h+1<=im.shape[1]


def build_patches(im,step=H):
    """ construction du dictionnaire : tous les patchs sans pixels morts en parcourant step by step l'image """
    res=[]
    step = step
    for i in range(0,im.shape[0],step):
        for j in range(0,im.shape[1],step):
            if inside(i,j,im) and np.sum(get_patch(i,j,im)[:,:,0]<=DEAD)==0:
                res.append(patch2vec(get_patch(i,j,im)))
    return res


def remove_patch(i,j,im,h=H):
    imn= im.copy()
    imn[(i-h):(i+h+1),(j-h):(j+h+1)]=DEAD
    return imn, get_patch(i,j,im)



def inpainting(patch, dict):
    pass

