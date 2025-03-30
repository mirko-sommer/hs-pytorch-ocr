# PyTorch OCR

## 🚀 Description

Project for the seminar "Introduction to Neural Networks and Sequence-To-Sequence Learning" SoSe 2024 at University Heidelberg.

The goal of this project is to implement a simple Optical Character Recognition Model in PyTorch. 
The trained model is tested on the captcha and IAM Handwriting Word dataset.

The raw experiment results can be found in [./results/](./results/).

Details of the implementation and the experiment results can be found in the [project report](./report.pdf).


## 📁 Project structure

```
.
├── data 
│     ├── captcha               
│     │         ├── raw         # put raw data set here
│     │         ├── test        # split test data
│     │         ├── train       # split train data
│     │         └── val         # split val data
│     ├── iam
│     │         ├── raw         # put raw data set here
│     │         ├── test        # split test data
│     │         ├── train       # split train data
│     └──       └── val         # split val data          
├── models     
│     ├── captcha               # trained captcha models
│     └── iam                   # trained iam models           
├── results  
│     ├── captcha               
│     │         ├── not_pretrained_resnet18     # results of resnet18 model, that has not been pre-trained
│     │         ├── pretrained_resnet18         # results of pretrained resnet18 model
│     │         └── simple_cnn                  # results of my simple cnn implementation
│     ├── iam
│     │         ├── not_pretrained_resnet18     # results of resnet18 model, that has not been pre-trained
│     │         ├── pretrained_resnet18         # results of pretrained resnet18 model
│     └──       └── simple_cnn                  # results of my simple cnn implementation                                      
├── src
│    ├── crnn.py                # model architecture
│    ├── dataset.py             # dataset processing, CaptchaDataloader and IAMDataloader
│    ├── plotting.py            # functions to plot results
│    ├── test_eval.py           # CTCModelTesterEvaluator Class to test and evaluate model
│    ├── train.py               # CTCModelTrainer Class to train the model
│    ├── utils.py               # utilites to encode and decode model outputs and text
│    ├── run_captcha.ipynb      # run captcha experiments
│    └── run_iam.ipynb          # run iam experiments
├── .gitignore
├── report.pdf   # final report of this project
├── README.md    # this README               
└── requirements.txt
```  


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

* Mirko Sommer

## References
[1] Alex Graves, Santiago Fernandez, Faustino Gomez, and Jurgen Schmidhuber. 
Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural ’networks. 
ICML 2006 - Proceedings of the 23rd International Conference on Machine
Learning, 2006:369–376, 01 2006.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015.

[3] U-V Marti and Horst Bunke. The iamdatabase: an english sentence database for offline handwriting recognition.
International Journal on Document Analysis and Recognition, 5(1):39–46,2002. 
Available at https://fki.tic.heia-fr.ch/databases/iam-handwriting-database.

[4] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 
Bleu: a method for automatic evaluation of machine translation. 
In Pierre Isabelle, Eugene Charniak, and Dekang Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics.

[5] Baoguang Shi, Xiang Bai, and Cong Yao.
An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition, 2015.

[6] Rodrigo Wilhelmy and Horacio Rosa.
captcha dataset, July 2013. 
Available at https://www.kaggle.com/datasets/fournierp/captcha-version-2-images/data.

## Code References
These references inspired my code structure, model architecture, and output formatting in some way.
1. https://medium.com/analytics-vidhya/resnetunderstand-and-implement-from-scratchd0eb9725e0db
2. https://github.com/GabrielDornelles/pytorchocr/tree/main
3. https://github.com/carnotaur/crnntutorial/tree/master
4. https://www.kaggle.com/code/kowalskyyy999/captcharecognition-crnn-ctcloss-using-pytorch
5. https://deepayan137.github.io/blog/markdown/2020/08/29/buocr.html
