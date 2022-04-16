# Generative Adversarial Network (GAN) model in Asset Pricing

Planning: 164 words / minute, total 20 minutes presentation
-> 3280 words.

## Slide 1: title slide

Good afternoon professors, my name is Lingjie and welcome to
my thesis presentation on Generative Adversarial Network
model, or GAN model, in Asset pricing. 

If you have any questions that might affect your
understanding of the model, please do let me know at the end
of each slide. However, if the question is not urgent, I
would appreciate it if you could save to the end of the
presentation.

## Slide 2: motivation

Let me first introduce the motivation for this thesis paper.

One important goal in the field of asset pricing is to
explain the variation in asset returns. To put it simply,
why are we able to make more money in some stocks but not
the others?

No-arbitrage pricing theory provides an elegant solution,
suggesting all assets are prices through a pricing kernel,
also known as stochastic discount factor. We will use the
term SDF to describe the pricing kernel in this
presentation, and we will revisit no-arbitrage pricing
theory in-depth later.

However, no-arbitrage pricing theory does not inform us what
exactly is the SDF. The empirical SDF estimation faces three
challenges.

Firstly, what are the right factors, or data features, to be
used in this estimation.

Secondly, what are the right assets, combined to be a
portfolio, to be used in this estimation.

Lastly, what is the functional form of the SDF?

We will first take a look at the current literature, before
introducing the GAN model used in this paper.

## Slide 3: literature review

The first class of empirical model proposed to estimate SDF
is the factor models. Starting from the 1960s, models such
as Capital Asset Pricing Model (CAPM), the Fama & French
three-factor model and four-factor model builds the
foundation for the empirical SDF today and is commonly used
as a benchmark model.

Generally, the factor models assumes that SDF is a linear
combination of a few constructed factors. We will explain
the factor models in-depth when we discuss the
methodology.

The literature direction after the factor models was to
search for even more factors to explain the variations in
asset returns. Hundreds of new factors were proposed in the
literature and Feng used the term "factor zoo" to describe
the variety and size of the factors.

However, Freyberger and Feng's did a systematic review of
all the factors and found only a few factors are
statistically significant in explaining the asset returns,
suggesting the search for factors remain a challenge to the
literature. Therefore, the literature direction expands on
the selected factors to a group of potentially infinite
factors.

## Slide 4: literature review (cont)

Previous factor models and factor search mainly focus on
linear estimation of SDF. Some literature has also expanded
the search for the SDF functional form and use non-linear,
non-parametric machine learning models for estimation.

Gu compared the various machine learning models proposed in
various literature and found that neural network
model has the highest out-of-sample R^2 in explaining the
asset returns. The empirical estimation of SDF has expanded
to searching for a general function of SDF using the
potentially infinite factors.

The most recent literature built on the work of previous
literature by including economic theory onto the machine
learning models. Chen proposed the GAN model used in this
paper and Gu proposed an Autoencoder network model. Both
models impose no-arbitrage pricing model to improve the
performance of a standard neural network.

Previously SDF is assumed to be only a function of firm
characteristic data, such as past returns and income
statement, firm fundamental data. However, Chen's GAN model
further includes macroeconomic data in the SDF estimation.
The current literature thus has expand the search for SDF
into a general function of potentially infinite factors and
macroeconomic data. And further impose economic theory in
the SDF estimation.

## Slide 5: research objective

In the spirit of Chen's GAN model give birth to the research
question we are discussing today. We want to examine the
external validity of Chens's GAN model using the UK LSE
data.

Why bother about the UK data you might ask. Chen, and many
other literature, focus mainly on the US data and resulted
in a "home bias", as Karolyi described. Therefore, we wanted
to know if GAN model works in markets outside the US, and
using UK market as an illustration.

We first summarise the results in this study. We have four
key findings:
Firstly, the GAN model indeed outperformed the benchmark
four-factor model in terms of Sharpe ratio.
Secondly, including macroeconomic data improves the SDF
estimation.
Thirdly, factors in the factor model proposed by Fama &
French are the most important covariates in SDF estimation.
Lastly, the factors proposed by Fama & French factor models
are nearly linear in the SDF, explaining why factor models
are commonly used as the benchmark model. However, we found
that there is interaction effects between the factors,
suggesting models such as neural network that is able to
take into consideration these interaction effects will be
able to outperform the benchmark models.

## Slide 6: methodology: no-arbitrage pricing model

We first explain the no-arbitrage asset pricing model.

The no-arbitrage asset pricing model assumes the existence
of an asset independent, time-dependent SDF M such that the
expected excess return, denoted by R^e, is 0.

We can also show that the SDF can be expressed as an affine
transformation of tangency portfolio F such that M = a - bF.
The tangency portfolio is any portfolio that achieves the
maximum sharpe ratio possible.

To construct the tangency portfolio, we construct a portfolio 
weighted by all the assets possible, and we assume the
weight of this portfolio, where we call SDF weight, is a
function of the firm characteristics data I_{t, i} and a
macroeconomic data I_t

We can further consider one of the many possible tangency
portfolio where a = 1 and b = 1, then the no-arbitrage asset
pricing model can be expressed as a function of the SDF
weight such that the following equation holds.

## Slide 7: methodology - no-arbitrage pricing loss

Now, if no-arbitrage pricing model alone is insufficient to
explain the variations in asset returns, then we can define
an alternative function g such that the variations in asset
returns are explained by both the no-arbitrage pricing
model and a g function we constructed.

This g function can be viewed as a combination of assets and
factors that no-arbitrage pricing model is least able to
explain.

This equation is what motives our no-arbitrage pricing loss
used in the GAN model. We define the loss function as the
following:


## Slide 8: methodology - empirical no-arbitrage pricing loss

Since not all stocks exists in the full duration of the
dataset, we are faced with an unbalanced panel data issue.
To overcome this, the empirical pricing loss is designed to
weighted the loss of the ith firm by the number of observation Ti

The GAN model will be optimized with respect to this
empirical pricing loss function.


## Slide 9: methodology - neural network model

Let me first explained a simple feed-forward neural network
before explaining the more complicated GAN model.

A standard feed-forward network consist of 3 layers. The
input layer, the hidden layer and the output layer. The
covariates first flow from the input layer onto the hidden
layer before combined in the output layer to produce a
numerical output. Note that Figure 1 illustrates a case
where the output is a numeric output, for a regression
problem.

Each of the units in the input layer correspond to a single
covariates. Therefore, in Figure 1 we have 3 data columns.

Each of the input units is connected to the individual
hidden units in the hidden layer. Each of the hidden units
perform a linear regression before outputting to a
non-linear function. As showed in the 'Hidden unit', X is
used as a linear regression while h is the non-linear
transformation.

A hidden layer will have multiple hidden units, where the
number of hidden units M is a hyperparameter set by the
user. A deep neural network will have multiple hidden
layers.

At the output unit, all the hidden layers will then be
combined to form a single numeric output in the case of an
output is quantitative or regression problem.

## Slide 10: methodology - LSTM

Now let me explain a Long Short-Term memory cell or LSTM.
