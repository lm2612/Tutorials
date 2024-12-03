# Introduction to AI Tutorials

This github repoitory contains code for Introduction to Supervised Learning and Introduction to Deep Learning courses. You can run the notebooks on Google colab. All code is in python and all neural networks are built in pytorch.


## Set up instructions
You will need to either login or sign up to a Google account to open and run notebooks on [Google colab](https://colab.research.google.com/). (Note, I have included details of my conda environment in `requirements.txt` file for running offline, but it is highly recommended to use Google colab during the tutorial to avoid any potential issues with package installations.)


## Introduction to Supervised Learning
### Part 1: Regression
We will use the [California house price dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

#### Tutorial
Use python packages to predict house prices starting with linear regression with one input feature, then adding more terms (polynomial regression), adding more variables (multivariate linear regression) and even regularisation.

<a target="_blank" href="https://colab.research.google.com/github/lm2612/Tutorials/blob/main/1_supervised_learning_regression/1-LinearRegression_HousePrice.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Advanced Tutorial (Code your own algorithms)
Code up your own linear regression, gradient descent, and stochastic gradient descent. Compare the results to the python packages scikit-learn and even Pytorch.

<a target="_blank" href="https://colab.research.google.com/github/lm2612/Tutorials/blob/main/1_supervised_learning_regression/1-AdvancedLinearRegression_HousePrice.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Part 2: Classification
We will use the [Titanic survival rate dataset](https://www.kaggle.com/c/titanic/overview).

#### Tutorial
Use python packages to compare different classification methods. We will build a logistic regression and classification tree and at the end we will compare their true negatives, true positives, false negatives and false positives. 

<a target="_blank" href="https://colab.research.google.com/github/lm2612/Tutorials/blob/main/2_supervised_learning_classification/2-Classification_Titanic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

#### Advanced Tutorial (Code your own algorithms)
Code up your own logistic regression and classification trees. For logistic regression, we will consider how to interpret the coefficients we have learned. For the classification trees, we will think about which variable to split on by considering which variable leads to the greatest reduction in entropy.

<a target="_blank" href="https://colab.research.google.com/github/lm2612/Tutorials/blob/main/2_supervised_learning_classification/2-Advanced_Classification_Titanic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Introduction to Deep Learning
### Part 3: Deep neural networks
We will use the same problem as part 1, the [California house price dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices), although we are using a larger dataset. We will build a simple neural network in Pytorch and you will learn how to code up the training loop, following the steps discussed in class. Look out for signs of overfitting and explore different neural network structures. Compare the results to the linear regression model we built on Tuesday.

<a target="_blank" href="https://colab.research.google.com/github/lm2612/Tutorials/blob/main/3_deeplearning/3-DeepLearning_HousePrice.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


### Part 4: Computer vision 
We will build a convolutional neural network (CNN) classifier to classify cats and dogs, using the [Oxford-IIIT-Pet dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). We will go through the steps needed to create a CNN to predict if an image is a dog (1) or a cat (0). Pay attention to the size of the output at each convolutional layer and check for any errors before running the training loop. At the end, you can look at what different layers of the network are doing, in other words, what kernels/filters have we learned?

<a target="_blank" href="https://colab.research.google.com/github/lm2612/Tutorials/blob/main/4_computervision/4-ComputerVision_Classification.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Resources

Python for R users:
* [Rebecca Barter Blog plot: From R to Python](https://rebeccabarter.com/blog/2023-09-11-from_r_to_python). Includes how to use Pandas dataframes, similar to R dataframes.
* [R Vignettes: Python Primer](https://cran.r-project.org/web/packages/reticulate/vignettes/python_primer.html). Includes an introduction to using classes, needed for deep learning tutorial.



