from sklearn.linear_model import Lasso
from patch_utils import *

from tqdm import tqdm

class Inpainting(object):
    def __init__(
        self,
        patch_size: int,
        step: int,
        lambda_: float,
        max_iterations: int,
        tolerance: float
    ):
        """
        Class for inpainting
        :param lambda, max_iteration, tolerance: for sklearn LASSO
        :param patch_size: patch dimension
        :param step: step to move the patch 
        """

        super(Inpainting, self).__init__()

        self.patch_size = patch_size
        self.step = step if step is not None else patch_size

        self.dead_pixels = np.ones(3,) * DEAD

        optimizer_kwargs = {
            "alpha": lambda_,
            "copy_X": True,
            "fit_intercept": True,
            "max_iter": max_iterations,
            "normalize": False,
            "positive": False,
            "precompute": False,
            "random_state": None,
            "selection": "cyclic",
            "tol": tolerance,
            "warm_start": False
        }

        self.optimizer = Lasso(**optimizer_kwargs)

    def _get_next_dead_pixel(
        self,
        im: np.ndarray
    ):
        """
        Method to get next dead pixel in the image
        """

        missing_x, missing_y, *_ = np.where(im <= self.dead_pixels)

        return zip(missing_x, missing_y).__next__()

    def inpaint(
        self,
        im: np.ndarray
    ):

        """
        Main method for inpainting
        """

        patch_dict = build_dict(im)

        nb_dead_pixels = np.sum(im == self) // 3

        progress_bar = tqdm(total=nb_dead_pixels)
        while self.dead_pixels in im:
            i, j = self._get_next_dead_pixel(im)
            next_patch = get_patch(i, j, im, h=self.patch_size)
            
            self.fit(patch_dict, next_patch)
            for x, y in iter_patch(im, i, j, self.patch_size):
                patch_coordinate_x = x - i + self.patch_size
                patch_coordinate_y = y - j + self.patch_size
                missing_value = self.predict(im, patch_dict, patch_coordinate_x, patch_coordinate_y)
                im[x, y] = missing_value
        progress_bar.close()
    
    def fit(
        self,
        patch_dict: np.ndarray,
        next_patch: np.ndarray
    ):

        """
        Method to compute the LASSO coefficient for a patch with missing
        value given the dictionary then predict the value of missing pixel
        using the sparse coefficient learned

        :param patch_dict: the dictionary
        :param next_patch: the patch with missing value
        """

        # fit
        next_patch_vec = patch2vec(next_patch)

        mask = (next_patch_vec > DEAD)
        mask_for_dict = mask[:, None].repeat(patch_dict.shape[-1], axis=-1)

        next_patch_vec_masked = next_patch_vec[mask]
        dict_masked = patch_dict[mask_for_dict]

        self.optimizer.fit(dict_masked, next_patch_vec_masked)

    def predict(
        self,
        im: np.ndarray,
        patch_dict: np.ndarray,
        patch_coordinate_x: int,
        patch_coordinate_y: int
    ):

        """
        Method for predicting the missing value

        :param im: numpy array of the image
        :param patch_dict: the dictionary
        :param patch_coordinate_x/y: coordinate of the missing pixel in the current patch
        """

        coordinate = [patch_coordinate_x * self.patch_size + patch_coordinate_y + n * self.patch_size ** 2 for n in range(3)]
        coordinates = np.array(coordinate)[:, None].repeat(patch_dict.shape[-1], axis=-1)

        patch_dict_missing = np.take_along_axis(patch_dict, coordinates, axis=0)
        missing_value = self.optimizer.predict(patch_dict_missing)

        return missing_value
