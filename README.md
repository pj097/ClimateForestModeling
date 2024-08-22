Under development. To be considered completed once tag v1.0 is released. 

The findings of this study are summarized in the [report](report/main.pdf). Please note that the views and conclusions presented in this document reflect the opinions of the author and do not necessarily represent those of any other individual or organization. Additionally, some aspects of the work presented in the report may rely on an earlier version of this repository, tagged as v0.1.

The provided Python files serve as tools for the Jupyter notebooks and are best used in conjunction with the report.

To get started, it is recommended to create a new Python environment. Please refer to the guidelines for Conda or Pip for instructions on setting up a new environment. If using Conda, you can create the environment with the following command using the .yml file in this repository:

```
conda env create -f environment.yml -p your_env_path_here
```

However, it may be preferable to install only the libraries you need individually, as the environment file may include unnecessary dependencies.

Begin with explore_euforest.ipynb, which explores the EU-Forest dataset (link to dataset) through interactive visualizations.

Use the download_data.ipynb notebook to download data from Google Earth Engine (Google Earth Engine Python API). The downloaded data can then be explored using visualise_features.ipynb.

For hyperparameter tuning, refer to hypertune.ipynb, which is configured using hyper_model_creator.py. Alternatively, you can skip hyperparameter tuning and create the model directly with keras_model_creator.py, then train it using other notebooks such as climate.ipynb.

The final notebook, climate.ipynb, requires ERA5 data. Instructions for obtaining this data can be found in download_data.ipynb. This notebook performs correlations between change maps in tree genus classification results and climate variables and implements a narrow neural network to predict tree change maps based on climate data.




MIT License

Copyright (c) [2024] [Pedro Junio]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.