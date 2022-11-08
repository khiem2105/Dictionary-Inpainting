from src.inpainting import *
from src.patch_utils import *

img = read_img("./image/outdoor.jpg")
noisy_img = remove(img, 288, 497, 190, 80)
show(noisy_img)

c = Inpainting(
    patch_size=50,
    step=None,
    max_missing_value=0,
    lambda_=0.0001,
    max_iterations=100000,
    tolerance=1e-4
)

new_im = c.inpaint(noisy_img)
show(new_im)