# Project Overview

This project is about "Machine Learning Based Classification of Myopic Maculopathy". Running main.py to run this project, settings such as different models and loss functions can also be adjusted.


## PyCharm Project Structure

`data_preprocessing` - Load data, data augmentation, and data preprocessing.  
`dataset` - Create dataset for training and testing, use data preprocessing methods here.  
`model` - Build different models here.  
`train_model` - Train and validate the model.  
`main` - Main process.  
`utils` - Useful tools for this project.  


## Project File Structure

The project code is developed using PyCharm, and for proper execution, the code and dataset must be structured as follows:

(Project file and dataset should in the same folder)  

  - `folder/classification_pycharm`


## Dataset Structure

Note: The dataset needs to be obtained from the organizer's website (https://codalab.lisn.upsaclay.fr/competitions/12441) and adjusted to the structure below.  

(Project file and dataset should in the same folder)  

## Suggested Environment for Project

If the computer already have python and machine learning environment, here are some new libraries that may need to be added:
 - `albumentations                1.4.0`
 - `autokeras                     1.1.0`
 - `deap                          1.4.1`
 - `efficientnet-pytorch          0.7.1`
 - `h5py                          3.10.0`
 - `huggingface-hub               0.20.3`
 - `imageio                       2.32.0`
 - `imbalanced-learn              0.11.0`
 - `joblib                        1.3.2`
 - `kagglehub                     0.1.9`
 - `keras                         2.10.0`
 - `keras-core                    0.1.7`
 - `keras-nlp                     0.7.0`
 - `Keras-Preprocessing           1.1.2`
 - `keras-tuner                   1.4.6`
 - `matplotlib                    3.8.0`
 - `matplotlib-inline             0.1.6`
 - `mlxtend                       0.23.0`
 - `numpy                         1.26.0`
 - `opencv-python                 4.9.0.80`
 - `opencv-python-headless        4.9.0.80`
 - `optuna                        3.6.1`
 - `pandas                        2.1.1`
 - `Pillow                        10.0.1`
 - `pip                           23.3.1`
 - `pretrainedmodels              0.7.4`
 - `PyWavelets                    1.4.1`
 - `scikit-image                  0.22.0`
 - `scikit-learn                  1.3.0`
 - `scipy                         1.11.3`
 - `seaborn                       0.13.0`
 - `segmentation-models-pytorch   0.3.3`
 - `SimpleITK                     2.3.1`
 - `skorch                        0.15.0`
 - `tensorboard                   2.10.1`
 - `tensorflow                    2.10.1`
 - `tf-keras                      2.15.0`
 - `timm                          0.9.2`
 - `torch                         2.1.0`
 - `torchvision                   0.16.0`
 - `TPOT                          0.12.1`
 - `tqdm                          4.66.1`
 - `transformers                  4.38.1`
 - `xgboost                       2.0.2`
  
Use pip to install :  
example: pip install albumentations  



Here is the list of libraries and their versions used when completing this project:
 - `absl-py                       2.1.0`
 - `albumentations                1.4.0`
 - `alembic                       1.13.1`
 - `anyio                         4.0.0`
 - `argon2-cffi                   23.1.0`
 - `argon2-cffi-bindings          21.2.0`
 - `arrow                         1.3.0`
 - `asttokens                     2.4.1`
 - `astunparse                    1.6.3`
 - `async-lru                     2.0.4`
 - `attrs                         23.1.0`
 - `autokeras                     1.1.0`
 - `Babel                         2.13.1`
 - `backports.functools-lru-cache 1.6.5`
 - `beautifulsoup4                4.12.2`
 - `bleach                        6.1.0`
 - `Brotli                        1.1.0`
 - `cached-property               1.5.2`
 - `cachetools                    5.3.2`
 - `certifi                       2023.11.17`
 - `cffi                          1.16.0`
 - `charset-normalizer            3.3.2`
 - `colorama                      0.4.6`
 - `colorlog                      6.8.2`
 - `comm                          0.1.4`
 - `contextlib2                   21.6.0`
 - `contourpy                     1.2.0`
 - `cycler                        0.12.1`
 - `deap                          1.4.1`
 - `debugpy                       1.8.0`
 - `decorator                     5.1.1`
 - `defusedxml                    0.7.1`
 - `dm-tree                       0.1.8`
 - `docopt                        0.6.2`
 - `efficientnet-pytorch          0.7.1`
 - `entrypoints                   0.4`
 - `et-xmlfile                    1.1.0`
 - `exceptiongroup                1.1.3`
 - `executing                     2.0.1`
 - `fastjsonschema                2.18.1`
 - `filelock                      3.13.1`
 - `flatbuffers                   23.5.26`
 - `fonttools                     4.44.0`
 - `fqdn                          1.5.1`
 - `fsspec                        2023.12.2`
 - `gast                          0.4.0`
 - `google-auth                   2.27.0`
 - `google-auth-oauthlib          0.4.6`
 - `google-pasta                  0.2.0`
 - `greenlet                      3.0.3`
 - `grpcio                        1.60.0`
 - `h5py                          3.10.0`
 - `higher                        0.2.1`
 - `huggingface-hub               0.20.3`
 - `idna                          3.4`
 - `imageio                       2.32.0`
 - `imbalanced-learn              0.11.0`
 - `imblearn                      0.0`
 - `importlib-metadata            6.8.0`
 - `importlib-resources           6.1.0`
 - `ipykernel                     6.26.0`
 - `ipython                       8.17.2`
 - `ipython-genutils              0.2.0`
 - `ipywidgets                    8.1.1`
 - `isoduration                   20.11.0`
 - `jedi                          0.19.1`
 - `Jinja2                        3.1.2`
 - `joblib                        1.3.2`
 - `json5                         0.9.14`
 - `jsonpointer                   2.4`
 - `jsonschema                    4.19.2`
 - `jsonschema-specifications     2023.7.1`
 - `jupyter                       1.0.0`
 - `jupyter_client                7.4.9`
 - `jupyter-console               6.6.3`
 - `jupyter_core                  5.5.0`
 - `jupyter-events                0.8.0`
 - `jupyter-lsp                   2.2.0`
 - `jupyter_server                2.9.1`
 - `jupyter_server_terminals      0.4.4`
 - `jupyterlab                    4.0.8`
 - `jupyterlab-pygments           0.2.2`
 - `jupyterlab_server             2.25.0`
 - `jupyterlab-widgets            3.0.9`
 - `kagglehub                     0.1.9`
 - `keras                         2.10.0`
 - `keras-core                    0.1.7`
 - `keras-nlp                     0.7.0`
 - `Keras-Preprocessing           1.1.2`
 - `keras-tuner                   1.4.6`
 - `kiwisolver                    1.4.5`
 - `kt-legacy                     1.0.5`
 - `lazy_loader                   0.3`
 - `libclang                      16.0.6`
 - `Mako                          1.3.3`
 - `Markdown                      3.5.2`
 - `markdown-it-py                3.0.0`
 - `MarkupSafe                    2.1.3`
 - `matplotlib                    3.8.0`
 - `matplotlib-inline             0.1.6`
 - `mdurl                         0.1.2`
 - `mistune                       3.0.2`
 - `ml-collections                0.1.1`
 - `ml-dtypes                     0.2.0`
 - `mlxtend                       0.23.0`
 - `mpmath                        1.3.0`
 - `munch                         4.0.0`
 - `munkres                       1.1.4`
 - `namex                         0.0.7`
 - `nbclient                      0.8.0`
 - `nbconvert                     7.10.0`
 - `nbformat                      5.9.2`
 - `nest-asyncio                  1.5.8`
 - `networkx                      3.2.1`
 - `nibabel                       5.1.0`
 - `notebook                      7.0.6`
 - `notebook_shim                 0.2.3`
 - `numpy                         1.26.0`
 - `oauthlib                      3.2.2`
 - `opencv-python                 4.9.0.80`
 - `opencv-python-headless        4.9.0.80`
 - `openpyxl                      3.1.2`
 - `opt-einsum                    3.3.0`
 - `optuna                        3.6.1`
 - `overrides                     7.4.0`
 - `packaging                     23.2`
 - `pandas                        2.1.1`
 - `pandocfilters                 1.5.0`
 - `parso                         0.8.3`
 - `patsy                         0.5.3`
 - `pickleshare                   0.7.5`
 - `Pillow                        10.0.1`
 - `pip                           23.3.1`
 - `pkgutil_resolve_name          1.3.10`
 - `platformdirs                  3.11.0`
 - `ply                           3.11`
 - `pretrainedmodels              0.7.4`
 - `prometheus-client             0.18.0`
 - `prompt-toolkit                3.0.39`
 - `protobuf                      3.19.6`
 - `psutil                        5.9.5`
 - `pure-eval                     0.2.2`
 - `pyasn1                        0.5.1`
 - `pyasn1-modules                0.3.0`
 - `pycparser                     2.21`
 - `Pygments                      2.16.1`
 - `pykwalify                     1.8.0`
 - `pyparsing                     3.1.1`
 - `PyQt5                         5.15.9`
 - `PyQt5-sip                     12.12.2`
 - `pyradiomics                   3.0.1`
 - `PySocks                       1.7.1`
 - `python-dateutil               2.8.2`
 - `python-json-logger            2.0.7`
 - `pytz                          2023.3.post1`
 - `PyWavelets                    1.4.1`
 - `pywin32                       306`
 - `pywinpty                      2.0.12`
 - `PyYAML                        6.0.1`
 - `pyzmq                         24.0.1`
 - `qtconsole                     5.4.4`
 - `QtPy                          2.4.1`
 - `qudida                        0.0.4`
 - `referencing                   0.30.2`
 - `regex                         2023.12.25`
 - `requests                      2.31.0`
 - `requests-oauthlib             1.3.1`
 - `rfc3339-validator             0.1.4`
 - `rfc3986-validator             0.1.1`
 - `rich                          13.7.0`
 - `rpds-py                       0.12.0`
 - `rsa                           4.9`
 - `ruamel.yaml                   0.18.5`
 - `ruamel.yaml.clib              0.2.8`
 - `safetensors                   0.4.2`
 - `scikit-image                  0.22.0`
 - `scikit-learn                  1.3.0`
 - `scipy                         1.11.3`
 - `seaborn                       0.13.0`
 - `segmentation-models-pytorch   0.3.3`
 - `Send2Trash                    1.8.2`
 - `setuptools                    68.2.2`
 - `SimpleITK                     2.3.1`
 - `sip                           6.7.12`
 - `six                           1.16.0`
 - `skorch                        0.15.0`
 - `sniffio                       1.3.0`
 - `soupsieve                     2.5`
 - `SQLAlchemy                    2.0.29`
 - `stack-data                    0.6.2`
 - `statsmodels                   0.14.0`
 - `stopit                        1.1.2`
 - `sympy                         1.12`
 - `tabulate                      0.9.0`
 - `tensorboard                   2.10.1`
 - `tensorboard-data-server       0.6.1`
 - `tensorboard-plugin-wit        1.8.1`
 - `tensorflow                    2.10.1`
 - `tensorflow-estimator          2.10.0`
 - `tensorflow-hub                0.16.1`
 - `tensorflow-intel              2.15.0`
 - `tensorflow-io-gcs-filesystem  0.31.0`
 - `tensorflow-text               2.10.0`
 - `termcolor                     2.4.0`
 - `terminado                     0.17.0`
 - `tf-keras                      2.15.0`
 - `threadpoolctl                 3.2.0`
 - `tifffile                      2023.9.26`
 - `timm                          0.9.2`
 - `tinycss2                      1.2.1`
 - `tokenizers                    0.15.2`
 - `toml                          0.10.2`
 - `tomli                         2.0.1`
 - `torch                         2.1.0`
 - `torch-lr-finder               0.2.1`
 - `torchaudio                    2.1.0`
 - `torchvision                   0.16.0`
 - `tornado                       6.3.3`
 - `TPOT                          0.12.1`
 - `tqdm                          4.66.1`
 - `traitlets                     5.13.0`
 - `transformers                  4.38.1`
 - `types-python-dateutil         2.8.19.14`
 - `typing_extensions             4.8.0`
 - `typing-utils                  0.1.0`
 - `tzdata                        2023.3`
 - `unicodedata2                  15.1.0`
 - `update-checker                0.18.0`
 - `uri-template                  1.3.0`
 - `urllib3                       2.0.7`
 - `wcwidth                       0.2.9`
 - `webcolors                     1.13`
 - `webencodings                  0.5.1`
 - `websocket-client              1.6.4`
 - `Werkzeug                      3.0.1`
 - `wheel                         0.41.3`
 - `widgetsnbextension            4.0.9`
 - `win-inet-pton                 1.1.0`
 - `wrapt                         1.14.1`
 - `xgboost                       2.0.2`
 - `xlrd                          2.0.1`
 - `zipp                          3.17.0`

