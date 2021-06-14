---
author:
- Lingjie
title: Thesis topic research
---

# Overview

[Computational Economics](https://www.springer.com/journal/10614):
Research on applying Machine Learning techniques on economics issues
$\rightarrow$ focusing on the HOW (framework and usage)

Proposed structure

1.  Research on existing framework and algorithms Explaining algorithms
    Comparing performance

2.  Applying framework and algorithms to illustrate the benefit it
    brings to economic field classic data (with known ground truth or
    baseline for comparison) new dataset to illustrate usefulness

Relevant keyword: computational economics, machine learning, economic +
\*topic keywords\*

Search words: [machine learning on
economics](https://scholar.google.com.sg/scholar?q=
    machine+learning+on+economics&hl=en&as_sdt=0&as_vis=1&oi=scholart)

## Questions on writing thesis paper

How in-depth should an undergraduate thesis be? $\rightarrow$ what is
the expectation? I saw some thesis simply compare off the shelve ML
models without hyper-parameter tuning.

# Possible topics

## ML + Causal inference

**Motivation:** I think social sciences' unique selling point is in its
causal inference study. The ability and focus on causal relationship
instead of correlation sets a social scientist apart from computer
scientist.\
Personally, I hope to further research on applying ML and other
techniques to estimate heterogeneous causal effects.\
However, I have no experience in causal inference and I'm uncertain
about how 'impactful' a causal paper might be, given that there is no
ground truth on counter factual in the real world to compare with.\
Having said that, research on causal inference is of my top interest
now.

### Relevant papers

Framework

-   [Machine Learning for Estimating Heterogeneous Causal
    Effects](https://ideas.repec.org/p/ecl/stabus/3350.html)

-   [Estimating treatment effect heterogeneity in randomized program
    evaluation](https://imai.fas.harvard.edu/research/files/svm.pdf)

-   [Modelling Heterogeneous Treatment Effects in Survey Experiments
    with Bayesian Additive Regression
    Trees](https://academic.oup.com/poq/article-abstract/76/3/491/1893905)

-   [Machine Learning for Causal Inference: On the Use of Cross-fit
    Estimators](https://oce-ovid-com.libproxy1.nus.edu.sg/article/00001648-202105000-00012/HTML)

-   [a list of research papers on causal
    ML](https://github.com/jvpoulos/causal-ml)

Sample use case

-   [Machine learning for causal inference in
    Biostatistics](https://academic-oup-com.libproxy1.nus.edu.sg/biostatistics/article/21/2/336/5631847)

Open-source library

-   [Uber causal ML](https://github.com/uber/causalml)

-   [Microsoft causal ML](https://github.com/microsoft/EconML) and
    [Microsoft Causality and Machine
    Learning](https://www.microsoft.com/en-us/research/group/causal-inference/)

-   [paid course on causal
    ML](https://www.altdeep.ai/p/causal-ml-minicourse)

## ML + Causal inference + Optimization

**Motivation:** This topic will be an extension of ML + Causal
inference. In a business use case, usually the causal effect is not the
final variable of interest. Business usually wants to maximise profit
and/or minimise cost. Therefore, the causal effect is used to construct
the optimization model which represents profit/cost function.\
However, currently business use correlation study to construct the
optimization model instead of causation. It would be interesting to
establish a framework from causal inference to optimization.

### Relevant papers

Framework

-   [Business Data Science: Combining Machine Learning and Economics to
    Optimize, Automate, and Accelerate Business
    Decisions](https://www.oreilly.com/library/view/business-data-science/9781260452785/)
    $\rightarrow$ 'Understand how use ML tools in real world business
    problems, where causation matters more that correlation'

## ML + Forecasting

**Motivation:** This will be most classic use case for Machine Learning
$\rightarrow$ prediction based problem. We can apply different ML
techniques to improve the economic forecasting ability.\
Furthermore, it will be the easiest to write about given the current
maturity in data science field and most of my internship experience is
prediction based.\
However, this topic is the least interesting to me as I have worked on
various projects related to prediction.\
In addition, I think there will be little value add to the academic
world given the massive research done on prediction problems.

### Relevant papers

Sample use case

-   [Machine Learning in Economics and
    Finance](https://link-springer-com.libproxy1.nus.edu.sg/article/10.1007/s10614-021-10094-w)

# Possible dataset

## Open data

Possible open sourced dataset (either through web scraping or direct
download)

-   Social media website (e.g. Twitter, Facebook, Linkedin)

-   [Google mobility data](https://www.google.com/covid19/mobility/)

-   Financial data Bitcoin/stock market Financial news (NLP)

-   Macro-economic indicators GDP/employment data

## Private data

1.  Singapore water consumption (domestic or non-domestic) $\rightarrow$
    require further communication with IWP and PUB I had a previous
    internship with IWP and my researcher said we could discuss further
    for data access
