The findings of this study are summarized in the [report](report/main.pdf). Please note that the views and conclusions presented in this document reflect the opinions of the author and do not necessarily represent those of any other individual or organization. Additionally, some aspects of the work presented in the report may rely on earlier versions of this repository as per the tags:

1. v0.1: includes feature selection such as seasons, additional feature datasets such as SoilGrids and elevation, and Sentinel-2 bands.
2. v0.2: first hyperparameter tuning set of trials.
3. v0.3: second hyperparameter tuning set of trials.
4. v1.0: conclusion of project, includes code for, e.g. figures in the results section of the report.

The provided Python files serve as tools for the Jupyter notebooks and are best used in conjunction with the report.

To get started, it is recommended to create a new Python environment. Please refer to the guidelines for Conda or Pip for instructions on setting up a new environment. If using Conda, you can create the environment with the following command using the .yml file in this repository:

```
conda env create -f environment.yml -p your_env_path_here
```

However, it may be preferable to install only the libraries you need individually, as the environment file may include unnecessary dependencies.

Begin with explore_euforest.ipynb, which explores the EU-Forest dataset (link to dataset) through interactive visualizations.

Use the download_data.ipynb notebook to download data from Google Earth Engine (Google Earth Engine Python API). The downloaded data can then be explored using visualise_features.ipynb.

For hyperparameter tuning, refer to hypertune.ipynb, which is configured using hyper_model_creator.py. To visualize tuning results, simply activate Tensorboard code within hypertune.ipynb. Alternatively, plot_tuner.ipynb can be used as a basis for more tailored plots. 

The final notebook, climate.ipynb, requires ERA5 data. Instructions for obtaining this data can be found in download_data.ipynb. This notebook performs correlations between change maps in tree genus classification results and climate variables and implements a narrow neural network to predict tree change maps based on climate data.