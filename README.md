# PyTorch OCR

[[_TOC_]]

## 🚀 Description

Project for the Hauptseminar "Introduction to Neural Networks and Sequence-To-Sequence Learning" SoSe 2024.

The goal of this project is to implement a simple Optical Character Recognition Model in PyTorch. 
The trained model is tested on the captcha and IAM Handwriting Word dataset.

The raw experiment results can be found in [./results/](./results/).

Details of the implementation and the experiment results can be found in the project report.


## 📁 Project structure

TODO     


## 📚 How to run the code

The code is only tested on Python 3.10.

First the datasets need to be downloaded:
* Follow thes instructions to get the captcha dataset:
    * Download the dataset from this [website](https://www.kaggle.com/datasets/fournierp/captcha-version-2-images/data)
    * Unpack the downloaded directory into [./data/captcha/raw](./data/captcha/raw/), so that the captcha images are directly in this directory
    * Create empty train/test/val directories in [./data/captcha/](./data/captcha/)

* Follow these instructions to get the IAM dataset:
    * Register for free on this [website](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
    * Download words/words.tgz
    * Unpack this directory into [./data/iam/raw](./data/iam/raw/), so that the sub-directories a01, a02 etc. are in [./data/iam/raw](./data/iam/raw/)
    * Create empty train/test/val directories in [./data/iam/](./data/iam/)

To train/evaluate the model just run the corresponding Jupyter Notebook: [run_captcha.ipynb](./src/run_captcha.ipynb) or [run_iam.ipynb](./src/run_iam.ipynb).

If you only want to evaluate the model, run all the cells before training and then load one of the provided [models](./models/) with the code.

The rest of the code in the Jupyter Notebooks should be self explaining. For further help you can also look at the docstrings of all my implemented functions.

## 👩‍🚀 Author

* [Mirko Sommer](mailto:mirko.sommer@stud.uni-heidelberg.de)
