# Strawberry maturity classification

Research code and measurements for [*Classification of strawberry maturity using Swin Transformer V2: integrating near-infrared spectral, chemical, and optical properties*](https://doi.org/10.1016/j.jfutfo.2025.02.001), published in *Journal of Future Foods* (2027, 7(1), 81–89).

The study compares three image classifiers: Swin Transformer V2 Tiny, RegNetY 800MF, and RegNetY 8GF. Near-infrared (NIR) spectra, chemical measurements, and surface color measurements provide complementary evidence about maturity. **The image-classification notebooks train on photographs; they do not fuse NIR or chemical measurements into the neural-network inputs.**

## Data and repository contents

The [Strawberry maturity with spectral & chemical data dataset on Kaggle](https://www.kaggle.com/datasets/yihfeng/strawberry-maturity) contains the strawberry images and dataset explanations. Download it separately; the photographs are not stored here. The paper describes 270 images from 90 strawberry samples across three maturity stages.

| File | Purpose |
| --- | --- |
| `swin transformer V2.ipynb` | Swin Transformer V2 Tiny image classifier |
| `RegNetY800MF.ipynb` | RegNetY 800MF image classifier |
| `RegNetY8GF.ipynb` | RegNetY 8GF image classifier |
| `K-means constrained.ipynb` | Constrained K-means analysis |
| `349.csv` | Numeric spectral data |
| `strawberry raw NIR spectra (absorbance) and Titratable acidity (%) and Total dissolved solids (°Bx) and Brix-to-acidity ratio and surface optical L a b values.xlsx` | NIR, chemical, and surface color measurements |

## Run the image-classification notebooks

1. Download and extract the [Kaggle dataset](https://www.kaggle.com/datasets/yihfeng/strawberry-maturity). Locate its cropped strawberry JPEG images.
2. Open a model notebook in Google Colab or a local Jupyter environment with Python, PyTorch, torchvision, Pillow, NumPy, and scikit-learn. A CUDA GPU is helpful but optional. The constrained K-means notebook also calls for `k-means-constrained`.
3. Change the `path` variable near the beginning of each image notebook from `/content/drive/MyDrive/strawberry/berry_jpg_cropped` to the directory containing the cropped images. If using Colab and Google Drive, mount Drive first.
4. Run the notebook. The model notebooks use ImageNet pretrained weights, so the first run may download weights.

The image loader expects **exactly 270 JPEG files** named `strawberry001.jpg` through `strawberry270.jpg` in one directory. Its hard-coded label order is `C` for images 1–90, `B` for 91–180, and `A` for 181–270. Check the Kaggle dataset description before changing image order or interpreting those codes as maturity stages.

The Swin notebook keeps three images per strawberry together in its split, with 60 samples for training and 30 for testing. It applies resizing, normalization, flips during training, and randomly selected CutMix or MixUp. It uses AdamW with cosine annealing over 20 epochs and prints test accuracy after every epoch. It does not save a trained checkpoint. Results can vary across runs and environments.

## Reported findings

In the [published paper](https://doi.org/10.1016/j.jfutfo.2025.02.001), Swin Transformer V2 Tiny achieved **100% accuracy on the study's test set**, outperforming the two RegNet models. This is a result on the reported dataset and split, not a claim of general performance on unseen farms or imaging conditions. The paper also reports that the Brix-to-acidity ratio separated maturity stages and that optimized NIR preprocessing produced a best silhouette score of 0.5744. These are separate analyses from the image classifier.

## Citation

If you use this repository or dataset, please cite:

> Feng, Y., Wang, Y., Purohit, A., Liang, J., Wang, X., Qiao, M., Xiao, Z., & Luo, Y. (2027). Classification of strawberry maturity using Swin Transformer V2: integrating near-infrared spectral, chemical, and optical properties. *Journal of Future Foods, 7*(1), 81–89. https://doi.org/10.1016/j.jfutfo.2025.02.001

## License

The code in this repository is licensed under [Apache License 2.0](LICENSE). Consult the Kaggle dataset page for terms that apply to downloaded images and data.
