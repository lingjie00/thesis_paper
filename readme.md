# Thesis research on Deep Learning in Asset Pricing

This GitHub repo contains the research effort and code implementation for my economics undergraduate honours thesis.

The final research topic is on applying deep learning in asset pricing model, based on [Luyang Chen, Markus Pelger, Jason
Zhu's work](https://arxiv.org/abs/1904.00745)

We will re-implement the [original code base](https://github.com/jasonzy121/Deep_Learning_Asset_Pricing) and adopt the
model to a new dataset.

We will introduce the following

1. [Thesis objective](#Thesis%20objective)
2. [Code base structure](#Code%20base%20structure)

The respective sub folder trees will contain readme for their respective contents.

# Thesis objective

1. [Re-implementation](#Re-implementation)
2. [Testing result with personal computers](#Testing%20result%20result%20with%20personal%20computers)
3. [Testing performance on new dataset](#Testing%20performance%20on%20new%20dataset)
4. [Tweaking model to improve performance](#Tweaking%20model%20to%20improve%20performance)

## Re-implementation

Although author has released their code base and dataset
[here](https://github.com/jasonzy121/Deep_Learning_Asset_Pricing), the code base lacks proper documentation and is based
on dated TensorFlow version 1 framework.

Our objective here will be to provide proper documentation and upgrade the code base to TensorFlow version 2 framework.

We hope more practical usage could happen with the proper documentation and latest framework.


## Testing result with personal computers

Authors ran the models based on two GPU clusters. Each cluster contains two Intel Xeon E5-2698 v3 CPUs with 1TB memory.
And 8 Nvidia Titan V GPUs. This is clearly only possible with strong resources.

As a result of their extensive resources, they were able to fit the whole data in memory for training. This is clearly
not possible for common personal computers.

We will modify the original implementation to suit common personal computers. In my case we will be using 1x Intel i7
with 8 GB memory, and Nvidia GTX 1080 with 8GB memory.

We will test if modifying the training to suit personal computers will affect the model performance in any way.


## Testing performance on new dataset

The authors have implemented their model based on the US stock exchange. We will also test the model performance based
on other markets to validate the theoretical accuracy of the model.


## Tweaking model to improve performance

We will be experimenting with different techniques to improve the model performance.

1. Testing RNN with firm characters

Although authors have used RNN for macroeconomic features, no RNN network is used for firm characters. We
propose that past firm characters might also affect the model performance
 
2. Testing transfer learning to another dataset

Since the most crucial part of the authors' implementation is training the SDF and moments network. We wanted to test
the model performance on a new dataset when we apply transfer learning and random initiation.

# Code base structure

We included both thesis writing (LaTeX based) and code writing (Python based) in this repertory.

## Thesis writing

- [topic research](topic_research)
    - explains the thought process during my topic brainstorming
- [model explaination](model_explaination)
    - explains the different components of the model

## Code writing

- [code implementation](code_implementation)
    - detailed code base for the model implementation
