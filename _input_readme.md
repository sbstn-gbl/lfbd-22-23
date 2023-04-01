# GitHub Repository for Course "Learning from Big Data"

## README content

<!-- vim-markdown-toc GFM -->

* [Repository content](#repository-content)
* [Requirements](#requirements)
* [Setup](#setup)
  * [Makefile targets](#makefile-targets)
  * [Step-by-step instructions](#step-by-step-instructions)
* [Course preparation](#course-preparation)
* [Lecture notebooks](#lecture-notebooks)
  * [Module 2](#module-2)
  * [Module 3](#module-3)

<!-- vim-markdown-toc -->

## Repository content

```
.
├── Makefile                  # run `make help` to see make targets
├── README.md                 # this readme file
├── requirements.txt          # virtualenv requirements file
├── lectures                  # lecture notebooks
├── preparation               # course preparation notebooks
└── source                    # sources, e.g., images for notebooks
```

Please consider the following instructions and the material in this repository carefully. The repository content is designed to make participation in Learning from Big Data as easy and enjoyable for you as possible.

## Requirements

1. Python 3.8
1. `virtualenv`

Optional:
1. node (for `plotly`)
1. `graphviz` (install with `brew install graphviz`)

Please familiarize yourselves with `virtualenv` (or a similar tool such as `conda`). Some background information can be found in the [virtualenv docs](https://virtualenv.pypa.io/en/latest/) or [here](https://stackoverflow.com/questions/34398676/does-conda-replace-the-need-for-virtualenv).

In the lectures, we will use Jupyter notebooks to illustrate implementation-related key points. The notebooks will be published in this repository well ahead of the lecture. Please make sure that you can execute the notebooks before joining the class so you can easily follow the coding parts in the lectures.

For the homework assignments, use an [IDE](https://en.wikipedia.org/wiki/Integrated_development_environment) of your choice. IDE choice really depends on personal preferences. A very popular choice is PyCharm (JetBrains offers a [free pro license for students](https://www.jetbrains.com/community/education/#students)). If you are familiar with coding this should be easy to manage. Other people like [Spyder](https://www.spyder-ide.org), [JupyterLab](https://jupyter.org) or [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb?utm_source=scs-index). Do some research to figure out which IDE suits your background and preferences best.

## Setup

### Makefile targets

The [Makefile](./Makefile) included in this repository is purely for convenience (e.g., setting up the virtual environment, launching a notebook server). It should work on Linux and Mac OS X systems.

```
$ make help
Make targets:
  build          create virtualenv and install packages
  build-lab      `build` + lab extensions
  freeze         persist installed packaged to requirements.txt
  clean          remove *.pyc files and __pycache__ directory
  distclean      remove virtual environment
  run            run jupyter lab
Check the Makefile for more details
```

### Step-by-step instructions

1. Open a terminal and navigate to the path that you want to clone the repository to
1. Clone the repository
    ```
    $ git clone git@github.com:sbstn-gbl/learning-from-big-data.git
    ```
1. Navigate to repository path, create virtual environment and install required modules with
    ```
    $ cd learning-from-big-data && make build
    ```
    or `make build-lab` to include `jupyterlab` dependencies.
1. Start a notebook server with
    ```
    $ make run
    ```

If `make` does not work on your computer run the steps included in the [Makefile](./Makefile) manually. You only need to do this setup once.

## Course preparation

Please solve the following three pre-course assignments before the first lecture.

- [Notebook 1: Data Tasks](preparation/notebook-1-data.ipynb)
- [Notebook 2: ML Metrics](preparation/notebook-2-metrics.ipynb)
- [Notebook 3: Gradient Descent](preparation/notebook-3-gradient.ipynb)

Use textbooks or online resources to fill gaps in your skills. The pre-course assignments will prepare you for the materials covered in Learning from Big Data and help you assess whether you are ready for this course. The pre-course assignments are challenging, if you find them ___too___ challenging, you should consider enrolling in this course in the following year. If you are not sure, feel free to contact one of the teachers before starting this course.

Please also study the material covered in the following online courses:

- [Introduction To Python Programming](https://www.udemy.com/course/pythonforbeginnersintro/)
- [Master Data Analysis with Python - Intro to Pandas](https://www.udemy.com/course/master-data-analysis-with-python-intro-to-pandas/)

## Lecture notebooks

### Module 2
- [Lecture 05-1: Logistic Regression (Motivation)](lectures/l05-1-binary-classification-motivation.ipynb)
- [Lecture 05-2: Missing Data](lectures/l05-2-missing-data.ipynb)
- [Lecture 05-3: Logistic Regression w/ MBGD](lectures/l05-3-binary-classification-mbgd.ipynb)
- [Lecture 06-1: Decision Trees](lectures/l06-1-decision-trees.ipynb)
- [Lecture 06-2: Representations](lectures/l06-2-representations.ipynb)
- [Lecture 06-3: Dimensionality Reduction](lectures/l06-3-dimensionality-reduction.ipynb)
- [Lecture 06-4: (Extra) Entropy](lectures/l06-4-extra-entropy.ipynb)
- [Lecture 07-1: AdaBoost](lectures/l07-1-adaboost.ipynb)
- [Lecture 07-2: Gradient Boosting](lectures/l07-2-gradient-boosting.ipynb)
- [Lecture 07-3: Overfitting](lectures/l07-3-overfitting.ipynb)
- [Tutorial Assignment 2 (Part 1)](lectures/tutorial-assignment-2.ipynb)
- [Tutorial Assignment 2 (Part 2)](lectures/tutorial-assignment-2-category-features.ipynb)

### Module 3
- [Lecture 09-1: NN Activation Functions](lectures/l09-1-activation-functions.ipynb)
- [Lecture 09-2: NN and PyTorch Autograd](lectures/l09-2-gradient-descent-revisited.ipynb)
- [Lecture 09-3: NN Backpropagation](lectures/l09-3-back-propagation.ipynb)
- [Lecture 09-4: NN Example Implementation](lectures/l09-4-nn-spiral.ipynb)
- [Lecture 09-5: NN Generalization](lectures/l09-5-generalization.ipynb)
- [Lecture 11-1: NN TensorBoard](lectures/l11-1-tensorboard.ipynb)
- [Lecture 11-2: NN Optimizer](lectures/l11-2-optimizers.ipynb)
- [Lecture 11-3: NN Initialization](lectures/l11-3-weight-initialization.ipynb)
- [Lecture 13-1: Example for Endogeneity](lectures/l13-1-endogeneity.ipynb)
- [Tutorial Assignment 3](lectures/tutorial-assignment-3.ipynb)
