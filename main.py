import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["main.py"],
    pythonpath=True,
    dotenv=True,
)
path = pyrootutils.find_root(search_from=__file__, indicator="main.py")

from src.inpainting import *
from src.patch_utils import *

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate


@hydra.main(version_base="1.2.0", config_path=path / "config", config_name="inpainting")
def main(cfg: DictConfig):
    image_name = cfg.image_name.split(".")[0]

    img = read_img(path / f"image/{cfg.image_name}")
    show(img, path / f"image/{image_name}_origin.jpg")
    
    noisy_img_fn = instantiate(cfg.noise)
    noisy_img = noisy_img_fn(img)
    show(noisy_img, path / f"image/{image_name}_noisy.jpg")

    c = instantiate(cfg.lasso)

    new_im = c.inpaint(noisy_img)

    fig = show(new_im, path / f"image/{image_name}_inpainted.jpg")

if __name__ == "__main__":
    main()
