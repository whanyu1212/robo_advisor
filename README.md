# ML Powered Robo Advisor
A robo advisor that is trained on ML to predict user preference in order to assign and execute trading strategy accordingly

```
|-- cfg
|   |-- config.yml
|   |-- strategy_template.yaml
|-- data
|   |-- processed
|       |-- df_trimmed.csv
|   |-- raw
|       |-- synthetic_data.csv
|-- logs
|-- mlruns
|-- notebooks
|   |-- test_api.ipynb
|-- src
|   |-- util
|       |-- data_schema.py
|       |-- general_util_functions.py
|       |-- strategic_conditions.py
|       |-- synthetic_data_generator.py
|   |-- interface.py
|   |-- data_processing.py
|   |-- fastapi_serving.py
|   |-- gcp_deployment.py
|   |-- main.py
|   |-- mlflow_tuning.py
|   |-- model_pipeline.[y]
|-- tests
|-- .env
|-- .gitignore
|-- .pre-commit-config.yaml
|-- dockerfile
|-- poetry.lock
|-- pyproject.toml
|-- README.md
|-- requirements.txt
```

### Part 1: ML Prediction to find the suitable investment strategy based on user profile features

### Part 2: Recommend and execute trading algorithm based on the results from Part 1

#### Configuring environment:
We will be using Pyenv + Poetry for environment & dependency management in this project. Conda is a decent enough alternative but I personally don't find it lightweight enough.

(1) Managing Python versions using Pyenv:
  - Version manager that helps you to manage different versions of Python based on project needs
  - Unfortunately, it doesn't work with windows. You will need WSL (a subsystem of Ubuntu within Windows) if you want to use Pyenv

```
# if you are using mac
brew install pyenv 

# if you are using linux
curl https://pyenv.run | bash 

# Or you can do it through the git clone way
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```
Adding Pyenv's path to .bashrc (linux) or .zshrc (mac)
```
# Run the following in terminal to add the path to .bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# Run the following in terminal to add the path to .zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```


(2) Use Poetry to ensure that this codebase is reproducible:
  - poetry can be installed using the following command in terminal if you are using mac/linux
    ```curl -sSL https://install.python-poetry.org | python3 -```
  - For windows(powersell):
    ```(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -```
  - Assume you already have a root folder of the project created
  - Call ```poetry init``` in terminal to instantiate a pyproject.toml, running ```poetry add [package]``` will create another poetry.lock file. 
  - Having these 2 files together and run ```poetry install``` after cloning the repository will ensure that someone else would be able to reproduce the environment and dependency of this project.

(3) Activating the environment:
- run ```poetry shell``` in the terminal