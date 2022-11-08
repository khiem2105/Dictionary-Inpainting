# Dictionary-Inpainting
Image inpainting using dictionary learning and sparse representation, inspired from the paper <a href="https://www.researchgate.net/publication/220736614_Image_inpainting_via_sparse_representation">Image Inpaiting via Sparse Representation</a>

# Some experimental results
<center>
  <img src="https://github.com/khiem2105/Dictionary-Inpainting/blob/main/image/outdoor_results1.png" alt="outdoor">
  <img src="https://github.com/khiem2105/Dictionary-Inpainting/blob/main/image/MU_results1.png" alt="MU">
  <img src="https://github.com/khiem2105/Dictionary-Inpainting/blob/main/image/castle_results1.png" alt="castle1">
  <img src="https://github.com/khiem2105/Dictionary-Inpainting/blob/main/image/castle_results2.png" alt="castle2">
 </center>

With an appropriate patch dimension, this method can recover an image that is 50% destroyed. However, it still has many disadvantages: computationally expensive, sensible to small change in patch size, etc. A better heuristic to reconstruct the missing patch still need to be considered.

# Running
The config files is organized with <a href="https://hydra.cc/">hydra</a>. The <a href="https://github.com/khiem2105/Dictionary-Inpainting/blob/main/config/lasso/default.yaml">lasso/default.yaml</a> file contains the default parameters for the LASSO regularizer. The <a href="">noise</a> folder contains 2 files for 2 method of destroying the image: add random noise and remove a whole rectangle patch, together with the corresponding hyperparameters (the noise ratio, the coordinate and the dimension of the patch to remove).

For example, to run an experiment to reconstruct an image in which a 190x80-pixel patch centered at (497, 288):

``` python main.py image_name=[path to your image]```

To run with a different set of parameters, hydra allows you to override the parameter right on the command line. For example:

```python main.py image_name=[...] lasso.lambda_=... lasso.max_iterations=... noise.rate=...```
