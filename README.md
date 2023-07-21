# Positron Binding Energy for Molecules

## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management - [article](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f)
* [hydra](https://hydra.cc/): Manage configuration files - [article](https://towardsdatascience.com/introduction-to-hydra-cc-a-powerful-framework-to-configure-your-data-science-projects-ed65713a53c6)
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting  - [article](https://towardsdatascience.com/4-pre-commit-plugins-to-automate-code-reviewing-and-formatting-in-python-c80c6d2e9f5?sk=2388804fb174d667ee5b680be22b8b1f)
* [DVC](https://dvc.org/): Data version control - [article](https://towardsdatascience.com/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-7cb49c229fe0)
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Project structure
```bash
.
├── config
│   ├── data                     # Configs related to the data paths
│   ├── eval                     # Configs related to the algorith evaluation paths
│   ├── feat_importance          # Configs related to path images of feature importance
│   ├── main.yaml                # Main Configuration File
│   ├── models                   # Configs related to optimized models path
│   ├── opt                      # Configs related to features to optimize and hyperparameter search space of models
│   └── process                  # Configs related to preprocessing the data
├── data
│   ├── final                    # Folder containing the molecules that have anisotropic polarizability
│   ├── processed                # Folder containing all molecules, but processed
│   └── raw                      # Raw data
├── docs
│   └── src
├── executar_fila.sh             # File to execute in terminal all the files to get opt models and eval performances.
├── general 
│   ├── creating_data.py         # File to get the processed and final data.
│   └── save_performances_img.py # File to save img of performances since the performances has been already calculated.
├── LICENSE
├── Makefile                     # Store useful commands to set up the environment
├── models                       # Folder cointaining the optimized models for each dataset and feature settings.
│   ├── apolar                          
│   ├── polar                           
│   └── polar_apolar                    
├── notebooks
│   └── Apolar Molecules         # Folder cointaining unstructured notebook with some random tests
├── performances                 # Folder with the performances in csv files for each dataset
│   ├── apolar                          
│   ├── polar                          
│   └── polar_apolar                    
├── performances_imgs            # Folder with the performances in .png graphs for each dataset
│   ├── apolar
│   ├── polar
│   └── polar_apolar
├── poetry.lock                  # Poetry folder for versioning the project
├── pyproject.toml               # Dependencies installed for this project.
├── README.md
├── src                          # Folder cointaining general utilized functions cointained in the documentation.
│   ├── __init__.py
│   ├── __pycache__
│   └── utils
├── src_apolar                   # Folder with the whole process including feature importance results with apolar molecules
├── src_polar                    # Folder with the whole process including feature importance results with polar molecules
├── src_polar_apolar             # Folder with the whole process including feature importance results with polar + apolar molecules
├── transformers.py              # File to replace in skopt library
└── txt_outputs                  # Folder with the principal outputs of .py files.
```

## Set up the environment
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make activate
make install
```

## Install new packages
To install new PyPI packages, run:
```bash
poetry add <package-name>
```
## Fixing some issues
### Optimization Library
The librarie used here to hyperparameter optimization (skopt) have some bugs that we have to fix mannually and it can be done just
by replacing the transformers.py file. One easy way to do this is get the directory returned by "poetry shell" command and do the following:
```bash
cp transformers.py {poetry_shell_path_returned}/lib/python3.9/site-packages/skopt/space/transformers.py
```
Where you just have do substitute the {poetry_shell_path_returned} with the path provided by poetry shell when activating the virtual environment.

### Running Jupyter Notebook
To run an Jupyter Notebook within the virtual environment created by poetry in this project, you could, of couse in the fold of the project when the environment is activated by "poetry shell" tip the following command:
```bash
poetry run ipython kernel install --user --name=pbe_for_molecules
```
If you want to change the name of the kernel created, you can just replace "pbe_for_molecules" by another name.

# Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs
```
