import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv,hsv_to_rgb

# patch dimension
H = 10

# Dead pixel value
DEAD = -100

def read_img(img: str):
    """ 
    Read an image
    Transform to numpy array with hsv color system 
    Normalize the pixel value

    :param img: image path
    """
    
    im = plt.imread(img)
    if im.shape[2] == 4:
        im = im[:, :, :3]
    if im.max() > 200:
        im = im / 255.
    
    return rgb_to_hsv(im) - 0.5

def show(im: np.ndarray, fig: bool=None):
    """
    Show an image

    :param im: image path
    :param fig: whether to create a new figure or not
    """
    
    im = im.copy()
    im[im <= DEAD] = -0.5
    
    if fig is None:
        plt.figure()
        fig = plt.imshow(hsv_to_rgb(im + 0.5))
    fig.set_data(hsv_to_rgb(im + 0.5))
    plt.draw()
    
    plt.pause(0.001)
    return fig

def get_patch(i: int, j: int, im: np.ndarray, h: int=H):
    """
    Return a patch with center at (i, j)

    :param i: y coordinate
    :param j: x coordinate
    :param im: numpy array represent the image
    :param h: patch dimension
    """

    return im[(i - h):(i + h + 1), (j - h):(j + h + 1)]

def patch2vec(patch: np.ndarray):
    """
    Flatten a 3D patch to a 1D vector

    :param patch: numpy array of the 3D patch
    """

    return patch.reshape(-1)

def vec2patch(X: np.ndarray):
    """
    Transform a 1D vector to a 3D patch

    :param X: numpy array of the 1D vector
    """

    h = int(np.sqrt(X.shape[0] // 3))

    return X.reshape(h, h, 3)

def make_noise(patch: np.ndarray, rate: float=0.2):
    """
    Randomly set some pixel to dead value in a patch or an image

    :param patch: numpy array of patch or image
    :param rate: noise rate
    """

    noisy_patch = patch.copy().reshape(-1, 3)
    h, w = patch.shape[:2]
    nb_noise = int(rate * h *w)

    noisy_patch[np.random.randint(0, h * w, nb_noise), :] = DEAD

    return noisy_patch.reshape(h, w, 3)

def remove(im: np.ndarray, i: int, j: int, height: int, width: int):
    """
    Remove a patch from the image

    :param im: numpy array of the image
    :param i: y coordiate of the patch center
    :param j: x coordinate of the patch center
    :param height: patch height
    :param width: patch width
    """

    im = im.copy()

    if width < 0:
        width = im.shape[1] - j

    if height < 0:
        height = im.shape[0] - i

    im[i:(i + height), j:(j + width)] = DEAD

    return im

def inside(i: int, j: int, im: np.ndarray, h: int=H):
    """
    Test if a patch is inside the image or not

    :param i, j: patch center coordinate
    :param im: numpy array of the image
    :param h: patch dimension
    """

    return i - h >= 0 and j - h >= 0 and i + h + 1 <= im.shape[0] and j + h + 1 <= im.shape[1]

def build_dict(im: np.ndarray, step: int=H):
    """
    Build the dictionary included all the patch without dead pixel from the img

    :param im: numpy array of the image
    :param step: step to move the patch
    """

    patch_dict = []

    for i in range(0, im.shape[0], step):
        for j in range(0, im.shape[1], step):
            if inside(i, j, im):
                patch = get_patch(i, j, im)

                if np.sum(patch[:, :, 0] <= DEAD) == 0:
                    patch_dict.append(patch2vec(patch))

    return np.array(patch_dict)