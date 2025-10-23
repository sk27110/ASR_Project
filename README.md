# Automatic Speech Recognition (ASR) with PyTorch

## Кочетков Александр, Падии 3

### Результаты
https://wandb.ai/kfloddroffff-national-research-university-higher-school-/SpeechModel_librispeech100/reports/ASR-Project--VmlldzoxNDgwODA0OQ?accessToken=jgvnfshyik4rxda9nn6buk3pz786a2yfbb5b3xjbo3bqnzdbnx0cmrwd66w9s5ns

Из проблем:

1) В коллабе не достаточно бесплатных вычислительных единиц для хорошего обучения.
2) Лосс функция на тесте и валидации ведет себя очень нестабильно.

Если есть какие-то комментарии, как можно решить проблемы -- можно написать в личку @Alexandr_Kochetkov или на почту sashakoch27@gmail.com

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
