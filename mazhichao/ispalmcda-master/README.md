# Integrating Static Program Analysis and Language Models for Co-occurrences Detection of Android Smells

In this README, we provide comprehensive instructions on setting up the repository and running the experiments presented in our paper. The code is designed to be easily adapted for further exploration of parameter-efficient fine-tuning methods applied to Large Language Models (LLMs) for other classification tasks.

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
- `ASSD`
  We have developed a static program analyzer capable of detecting code smells.  To complete this experiment, we enhanced ASSD's functionality to enable the detection of co-occuring MIM and LPL code smells.
​ASSD – Android-Specific Smell Detection
ASSD is an open-source, Java-based tool for detecting Android-specific code smells (anti-patterns) in Android projects.
<img width="1072" height="653" alt="assd" src="https://github.com/user-attachments/assets/4af98aa4-78d1-47a9-a9e2-d6b8b60e0f87" />
✨ Features
Detects a wide range of Android code smells (e.g., Leaking Inner Class, Unused Resources, Overdraw, etc.)
User-friendly graphical interface (GUI)
Supports custom selection of smell types
Displays detailed results: class names, file paths, and total count
🚀 Quick Start
Prerequisites
JDK 1.8
Android Studio or IntelliJ IDEA
Setup
Clone or download this repository:
bash
git clone https://github.com/cmdzlw/415/edit/main/mazhichao/ispalmcda-master/ASSD.git
Open the project in Android Studio or IntelliJ IDEA.
Run ASD.java — the main GUI entry point.
Usage
Click “Browse” to select an Android project.
Tick the code smells you want to detect.
Click “Start Parsing”, then “Detect Selected Code Smell”.
View results in the console at the bottom, including:
Detected issues
Affected class names
Full file paths
Total count of detected smells
📄 License
This project is licensed under the MIT License.	

## Installation

1. Clone the repo using `git clone` command.

2. Using Conda to create a `Python 3.10` virtual environment and install the dependencies.

   ```
   conda create -n myenv python=3.10
   conda activate myenv
   pip install -r requirements.txt
   ```

   Note that we run all the experiments on a single a 24G NVIDIA RTX 4090D GPU.

   

