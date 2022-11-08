import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["main.py"],
    pythonpath=True,
    dotenv=True,
)
path = pyrootutils.find_root(search_from=__file__, indicator="main.py")

from sklearn.linear_model import Lasso
from src.patch_utils import *

from tqdm import tqdm

class Inpainting(object):
    def __init__(
        self,
        patch_size: int,
        step: int,
        max_missing_value: int,
        lambda_: float,
        max_iterations: int,
        tolerance: float
    ):
        """
        Class for inpainting
        :param lambda, max_iteration, tolerance: for sklearn LASSO
        :param patch_size: patch dimension
        :param step: step to move the patch 
        :param max_missing_value: max number of missing pixel in one dictionary patch
        """

        super(Inpainting, self).__init__()

        self.patch_size = patch_size
        self.step = step if step is not None else patch_size
        self.max_missing_value = max_missing_value

        self.dead_pixels = np.ones(3,) * DEAD

        optimizer_kwargs = {
            "alpha": lambda_,
            "copy_X": True,
            "fit_intercept": True,
            "max_iter": max_iterations,
            "positive": False,
            "precompute": False,
            "random_state": None,
            "selection": "cyclic",
            "tol": tolerance,
            "warm_start": False
        }

        self.hue_optimizer = Lasso(**optimizer_kwargs)
        self.saturation_optimizer = Lasso(**optimizer_kwargs)
        self.value_optimizer = Lasso(**optimizer_kwargs)

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

        patch_dict = build_dict(im, step=self.step, patch_size=self.patch_size, max_missing_value=self.max_missing_value)

        nb_dead_pixels = np.sum(im == self.dead_pixels) // 3

        progress_bar = tqdm(total=nb_dead_pixels)
        while self.dead_pixels in im:
            i, j = self._get_next_dead_pixel(im)
            print(f"Dead pixel: {i, j}")
            next_patch = get_patch(i, j, im, h=self.patch_size)
            print(f"Next patch shape: {next_patch.shape}")
            
            self.fit(patch_dict, next_patch)
            for x, y in iter_patch(im, i, j, self.patch_size):
                # print(x, y)
                patch_coordinate_x = x - i + self.patch_size
                patch_coordinate_y = y - j + self.patch_size
                missing_value = self.predict(im, patch_dict, patch_coordinate_x, patch_coordinate_y)
                im[x, y] = missing_value
            
                progress_bar.update(1) 
        progress_bar.close()

        return im
    
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
        mask = (next_patch > DEAD)

        data_y_hue = next_patch[mask[:, :, 0], 0]
        data_y_saturation = next_patch[mask[:, :, 1], 1]
        data_y_value = next_patch[mask[:, :, 2], 2]

        data_x_hue = patch_dict[:, mask[:, :, 0], 0].T
        data_x_saturation = patch_dict[:, mask[:, :, 1], 1].T
        data_x_value = patch_dict[:, mask[:, :, 2], 2].T
        
        self.hue_optimizer.fit(data_x_hue, data_y_hue)
        self.saturation_optimizer.fit(data_x_saturation, data_y_saturation)
        self.value_optimizer.fit(data_x_value, data_y_value)

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
        hue = self.hue_optimizer.predict(patch_dict[:, patch_coordinate_x, patch_coordinate_y, 0].reshape(1, -1))
        saturation = self.saturation_optimizer.predict(patch_dict[:, patch_coordinate_x, patch_coordinate_y, 1].reshape(1, -1))
        value = self.value_optimizer.predict(patch_dict[:, patch_coordinate_x, patch_coordinate_y, 2].reshape(1, -1))

        missing_value = np.hstack([hue, saturation, value])
        return missing_value
