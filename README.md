# Imbalanced-Quickdraw

Winning an intern competition on imbalanced image classification. The competition took place between November, 15th 2021 and December, 15th 2021.

## Table of contents

- [Description and rules of the competition](#description-and-rules-of-the-competition)
- [Tasks list](#tasks-list)
- [Overview](#overview)
  - [Data augmentation](#data-augmentation)
  - [Architectures](#architectures)
  - [Regularization](#regularization)
  - [Mobiletnet as feature extractor](#mobilenet-as-a-feature-extractor)
  - [Few shot learning](#few-shot-image-classification)
  - [Semi-supervised learning with noisy student](#semi-supervised-learning-noisy-student-learning)
  - [Ensemble methods](#ensemble-method)
- [What I learnt](#what-i-learnt)

## Description and rules of the competition

The competition is about imbalanced image classification. The dataset is a subset of Google [_Quick, draw_ dataset](https://quickdraw.withgoogle.com/). Images are black and white drawings of 28x28 pixels.

![preview](images/preview.jpg)

For the competition, the dataset is restricted to 100 classes. The first class has 10 images, the second has 20, the third has 30, ..., the last class has only 1,000 images ! The test set is a balanced set of 100,000 images. The goal is to reach the best accuracy score on the testing set. The competition took place between November, 15th 2021 and December, 15th 2021.

The dataset for the competition can be found [here](https://drive.google.com/file/d/1AMHo0YKzlh7yMqqA8GDDvk2TFXnkz4Y0/view?usp=sharing).

If you want to test your model, the solution is [here](https://drive.google.com/file/d/1AcvXyTArv_MnJEXnCqLhVhaAduIRc_Cb/view?usp=sharing).

### Special rules

To encourage participants to make the best of their knowledge, some rules were added:

- People can group into teams of 3 people or less
- Pre-trained models for fine-tuning are forbidden
- Only the provided dataset can be used
- One team can only submit 10 times during all the competition

This competition was organized by Automatants - the AI student organization of CentraleSupélec in order to promote Deep learning and to compete with each other. Thanks to [Thomas Lemercier](https://github.com/ThomasLEMERCIER) for organizing this competition and to Loïc Mura for hosting the competition.

## Tasks list

- - [x] Data Augmentation
- - [x] Several architectures of CNNs:

  * - [x] Base CNN
  * - [x] Resnet
  * - [x] Mobilenet v2

- - [x] Regularization

  * - [x] Weight Decay
  * - [x] Label smoothing
  * - [x] Dropout
    * - [x] Classic dropout
    * - [x] Spatial dropout
    * - [x] Stochastic Depth
  * - [x] Early Stopping

- - [x] Reduce LR on plateau
- - [x] Feature extractor + Classifier
  * - [x] Mobilenet + LDA
  * - [x] Mobilenet + Gradient Boosting
- - [x] Semi-supervised training
- - [ ] Few shot learning
- - [x] Ensemble
  * - [x] Vote
  * - [x] Weighed vote
  * - [x] Meta Learner
  * - [ ] Distillation
- - [ ] Weighted loss

## Overview

### Data augmentation

I mainly used basic data augmentation to limit class imbalance influence on training. Advanced data augmentation techniques such as cutmix, random erasing or mixup seemed less adapted to this problem and harder to implement.

Reference:

- [The Effectiveness of Data Augmentation in Image Classification using Deep
  Learning (Dec 2017)](https://arxiv.org/pdf/1712.04621.pdf)
- [mixup: Beyond Empirical Risk Minimization (Oct 2017)](https://arxiv.org/pdf/1710.09412)
- [Random Erasing Data Augmentation (Aug 2017)](https://arxiv.org/pdf/1708.04896)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (Aug 2019)](https://arxiv.org/pdf/1905.04899)

### Architectures

As I did not have a very powerfull hardware, I had to use small networks. Thus, I mainly used architectures with Mobiletnet v2 modules.

Reference:

- [Deep Residual Learning for Image Recognition (Dec 2015)](https://arxiv.org/pdf/1512.03385v1)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks (Jan 2018)](https://arxiv.org/pdf/1801.04381)

### Regularization

All my models were overfitting so I tried many regularization techniques to limit that: Weight decay, Label smoothing and Dropout.

Reference:

- [When Does Label Smoothing Help? (Jun 2020)](https://arxiv.org/pdf/1906.02629.pdf)
- [Dropout: A Simple Way to Prevent Neural Networks from
  Overfitting (2014)](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [Understanding the Disharmony between Dropout and Batch Normalization by
  Variance Shift (Jan 2018)](https://arxiv.org/pdf/1801.05134.pdf)
- [Deep Networks with Stochastic Depth (Jul 2016)](https://arxiv.org/abs/1603.09382v3)

### Mobilenet as a feature extractor

As more basic Machine learning techniques are often more robust to overfitting and data imbalance, I tried to used features geenrated by a Mobilenet and classify them using other Machine learning algorithms. I tried to use Linear discriminant analysis and Gradient boosting but the accuracies were nowhere near the regular Mobilenet. An explanation might be that Mobilenet reached a state through a global optimization and thus by exploring the space of the solution in a continuous way. The same conclusions might be too hard to reach with transfer learning.

Reference:

- [How transferable are features in deep neural networks? (Nov 2014)](https://arxiv.org/pdf/1411.1792.pdf)

### Semi-supervised learning: Noisy student learning

Looking for more data is forbidden and the training set only contains 50,500 while the testing set has 100,000 samples. This happens a lot in the real world as labelizing data costs a lot. Semi-supervised learning takes advantage of this unlabeled data.

Noisy student learning consists in training a CNN with a noisy training (Dropout, data-augmentation, ...). This trained CNN then generates pseudo-labels on the unlabeled data. A threshold is applied to only keep predictions where the confidence is high. Another CNN is then trained with the additional pseudo-labeled data.

Reference:

- [Self-training with Noisy Student improves ImageNet classification (June 2020)](https://arxiv.org/pdf/1911.04252.pdf)

### Few shot image classification

Few shot learning corresponds to the challenge of classifying images with very few training data. I did not have enough time to make it work but it would definitly be worth to investigate FSL. A way to use FSL would have been to use top K predictions by mobilenet, feeding it to a FSL algorithm and do an ensemble method on both predictions.

Reference:

- [Learning to Compare: Relation Network for Few-Shot Learning
  (Mar 2018)](https://arxiv.org/pdf/1711.06025)

### Ensemble method

As a last way to improve my models, I used ensemble methods to combine models. I started with a simple vote among my best models and this constitued my best model. I also tried to combine predictions with a meta-model but that was no better.

### Weighted loss

To limit class imbalance, I could have used a weighted loss instead of augmenting my data until it is balanced. I would have saved some memory and some time but it was not really limiting.

### Best model

My best model was a ensemble of six regular Mobilenets trained with Noisy student learning.

## What I learnt

I started learning about Machine Learning one year ago. How far can I go now ? I wanted to put all my efforts in improving my understanding of CNNs, even if it meant taking more time to read paper than fine-tuning.

Surprisingly, the architecture did not have a great influence on the accuracy. However, using Mobilenet v2 was definitly a great choice to have a light weight model which could be trained much faster than any other models.

Regularization was the key of the competition. As my models were always overfitting, regularization techniques helped me a lot to increase accuracy without overfitting.

Ensemble methods are powerful tools as well. If I had the time I would have tried to use distillation on my ensemble model as it is much more satisfaying to have a good accuracy with smaller models.

I also had the occasion to try more original methods such as **Semi-supervised training with Noisy student learning**, **Few shot learning** and LDA/Gradient Boosting with a deep feature extractor. I was sort of mitigated between trying something new with low chances to improve my score and trying to find better hyperparameters. I am glad to have chosen to try as many different approches as possible.
