# Integrating Static Program Analysis and Language Models for Co-occurrences Detection of Android Smells

Replication package for our paper, "Integrating Static Program Analysis and Language Models for Co-occurrences Detection of Android Smells", submitted to TOSEM. In this README, we provide comprehensive instructions on setting up the repository and running the experiments presented in our paper. The code is designed to be easily adapted for further exploration of parameter-efficient fine-tuning methods applied to Large Language Models (LLMs) for other classification tasks.

## Directory Structure of the Repo

- `train` folder

  contains 5 python files, including `prompt_tuning.py`, `prefix_tuning.py`, `lora.py`, `IA3.py`, and  `full_fine_tuning.py`. These files implement different PEFT methods as well as the full fine-tuning approach for training language models.

- `utils.py`

  includes utility functions that are commonly used across the training scripts.

- `test.py`

  evaluates the fine-tuned language models, either using PEFT methods or full fine-tuning.

- `dataset` folder

  contains the training, validation, and test sets for the occurrence of LPL and MIM smells
- `results` folder

  contains the results of our experiments

- `requirements.txt`

  contains a list of all the Python packages required for running the experiments in this repo.

​	

## Installation

1. Clone the repo using `git clone` command.

2. Using Conda to create a `Python 3.10` virtual environment and install the dependencies.

   ```
   conda create -n myenv python=3.10
   conda activate myenv
   pip install -r requirements.txt
   ```

   Note that we run all the experiments on a single a 24G NVIDIA RTX 4090D GPU.

   

