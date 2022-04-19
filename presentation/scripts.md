# Generative Adversarial Network (GAN) model in Asset Pricing

Planning: 164 words / minute, total 20 minutes (1200 sec)
presentation → max 3280 words.

Summary of time taken:

1. Motivation (90 sec)
2. Literature review (100 sec)
3. Research objective (54 sec)
4. Methodology (155 sec)
6. Evaluation metrics (30 sec)
7. Data (83 sec)
8. Results (500 sec)
9. Discussion (40 sec)
10. Conclusion (58 sec)

## Slide 1: title slide (25 sec)

Good afternoon professors, my name is Lingjie and welcome to
my thesis presentation.

I will be presenting on Generative Adversarial Network
model in Asset pricing. I will use the term "GAN model" to
describe the model from now on.

If you have any questions, please feel free to ask me at the
end of the presentation.

## Slide 2: overview (25 sec)

I will first explain the motivation that drives this paper,
and the current development in the literature before
introducing my research objective.

Thereafter, I will explain the methodology, evaluation
metrics, and data before showcasing the results.

Finally, I will end the presentation with some discussion
and conclusion.

## Slide 3: motivation (50 sec)

Our paper falls in the domain of asset pricing.
One important goal in asset pricing is to explain the
variation in asset returns.

No-arbitrage pricing theory explains the asset returns with
a pricing kernel, also known as stochastic discount factor.
We use the term "SDF" in this presentation.

However, a question remains:
what exactly is this SDF, and how do we estimate it?

There are three challenges in estimating the SDF.
Firstly, choosing the right factors, or data features, for
estimation.

Secondly, selecting the right combination of assets, or
portfolio, for estimation.

Lastly, estimating the functional form of the SDF.

Let's first take a look at how the current literature
answers these challenges.

## Slide 4: literature review (50 sec)

The first few models used in asset pricing were proposed in
the 60s and 90s and are still regarded as the benchmark
model today.

These models include Capital Asset Pricing Model, Fama and
French three factor and Carhart four factor model.
These models estimate the SDF with an OLS and some
constructed factors.

Following the literature, the factors in these classical
models will be address as "extended Fama French factors".

Literature after the 90s expanded the search for more
factors.
Feng used the term "factor zoo" to describe the
hundreds of new factors proposed in the literature.

However, the search for factors does not end here.
Freyberger and Feng reviewed all proposed factors and found
few of them are relevant in explaining the asset returns.

## Slide 5: literature review (cont) (55 sec)

Beyond the factor search, literature also explored non-linear,
and non-parametric SDF estimation.

Gu compared various machine learning models and found neural
network model as the best performing one.
However, the improvement in neural network compare to fama
french model is only marginal.

The latest research therefore expanded on the classical
machine learning to introduce economic theory in model
estimation.

Chen proposed the GAN model used in this paper and Gu
proposed an Autoencoder neural network.
Both models improve upon standard neural network with
no-arbitrage pricing theory.

Chen also found that including macroeconomic data further
improves the SDF estimation.

Summarizing the current literature, SDF seems to be a
general function of potentially infinite factors, while
including macroeconomic data and imposing economic theory
can further improve the empirical SDF estimation.

Next, let me formally introduce our research objective.

## Slide 6: research objective (54 sec)

Our research is built on Chen's work.
We aim to examine the GAN model's external validity with
United Kingdom London Stock Exchange data.

Our paper contributes to the limited literature done on
non-US market.

Most literature focus only on the US data, resulting in a
"home bias". Our research aims to investigate if the best
performing model in the US will continue to perform in other
regions.

There are four key findings in our research:
Firstly, the GAN model outperformed the benchmark
four-factor model even in the UK market.
Secondly, macroeconomic data is important in the SDF
estimation.
Thirdly, extended fama french factors are the most important
covariates in the SDF estimation and
Lastly, these factors are nearly linear in the SDF. However,
there are interaction effects between the factors.

We will now explain the methodology in depth to arrive at
these results.

## Slide 7: methodology: no-arbitrage pricing model (52 sec)

The first model we will be introducing is the no-arbitrage
pricing model, which motivates our loss function.

The model assumes an asset independent, time-dependent SDF
such that there is no excess return in expectation. Equation
1 shows this relationship formally.

We can also express the SDF as an affine transformation of a
tangency portfolio F.
The tangency portfolio is any portfolio that maximizes the
mean-variance efficiency, and is constructed as a weighted
portfolio of all possible assets. We assume this SDF weight
omega is a function of firm characteristics data I ti and
macroeconomic data I t

We consider one specific tangency portfolio where 
a = 1, b = 1, then the no-arbitrage asset
pricing model can be expressed as a function of the SDF
weight shown in equation 2

## Slide 8: methodology - no-arbitrage pricing loss (30 sec)

However, if no-arbitrage pricing model alone is insufficient
to explain the variations in asset returns, then we can
define a competing function g such that there is no excess
return in expectation only when we consider both the
no-arbitrage pricing model and the conditional moment
function g as shown in equation 3

This g function can be viewed as selecting portfolio and
factors that no-arbitrage pricing model is least able to
explain.

Equation 3 motives our pricing loss function used in the GAN
model estimation. The loss function is shown in equation 4.

## Slide 10: methodology - neural network model (37 sec)

The backbone of the GAN model is various kinds of neural
network model.

We will first explain a simple feed-forward neural network model
before explaining the GAN model.

A feed-forward network consist of one input layer
one or more hidden layers
and one output layer.

In essence, feed-forward neural network performs a linear
combination of covariates before passing the intermedia
output to a non-linear function h

The output from the non-linear function is then
linearly combined again, and the procedure repeats until the
output layer, producing the final output

Besides the feed-forward neural network, our GAN model also
uses a recurrent neural network LSTM which takes into
consideration the time dynamic nature of macroeconomic data.

## Slide 11: methodology - GAN model (35 sec)

To construct the GAN model, we use two competing neural
network models.

The first model, called the discriminator, estimates the SDF
weight omega while the second model, called the generator,
estimates the conditional moment function g.

The GAN training procedure can be viewed as a zero-sum game
where the discriminator minimizes the pricing loss while
generator maximize the pricing loss.

The objective here is to achieve a Nash-Equilibrium where the
discriminator best approximates the SDF function while
generator is able to construct portfolio and factors that
no-arbitrage pricing theory is least able to explain.

Next, we will take a look at how to train the GAN model

## Slide 12: methodology - GAN model structure (15 sec)

As shown in Figure 2, the GAN model training involves the
discriminator and generator linked by a single pricing loss
function.

Both models take the firm characteristics data and
macroeconomic data as input, but produce different outputs.

As mentioned earlier, the objective here is to sequentially
train the discriminator and generator through a min max
optimisation in order to achieve a Nash equilibrium

## Slide 13: methodology - benchmark four-factor model (16 sec)

The benchmark model used in this paper is the Carhart
four-factor model.

The four extended fama french factors in this model are
market risk, difference in returns between big and small
firms, difference in returns between high value and low
value firms, and the momentum factor.

The factor model is estimated through standard OLS
regression.

## Slide 14: evaluation metrics - Sharpe Ratio (30 sec)

We use Sharpe ratio as the evaluation metric

Sharpe ratio was proposed by Sharpe to evaluate the
performance of risk-adjusted assets relative to a risk-free
asset. 

The formulation of Sharpe ratio express the desire to
maximize excess return while minimizing risk.
The risk is approximated by the standard deviation of
asset returns.

A higher Sharpe ratio indicates a better performing
portfolio and tangency portfolio by definition achieve the
highest Sharpe ratio possible.

## Slide 15: Data - UK LSE (40 sec)

The data used in this paper consist of UK LSE monthly stock
prices from 1998 to 2017.

The stock prices are extracted from Yahoo! Finance and we
compute the log difference as the stock returns.

The risk-free rate and fama french factors were provided by
Gregory, while the firm characteristics data is retrieved
from paid data subscription service Finage.

The UK macroeconomic data is provided by Coulombe.

In total, our data contains 132 covariates, 942 stocks with
maximum of 20 years returns.

## Slide 16: Data - Data splitting (43 sec)

We divide the data into three periods, 12 years of training
data, 4 years of validation data and 4 years of
out-of-sample test data.

We constructed two different data splits to maximise the
available data.
The first split is called "Factor data" as it contains only
extended fama french factors and macroeconomic data. This dataset
contains 942 stocks with 116 covariates.

The second data split is called "Fundamental data", which
consist of firm income statement data on top of the
covariates available in the "Factor data".

The two data split is essential as we only keep the stocks
with full covariates in a particular month, and Finage
database only contains limited income statement data for a
subset of stocks.

## Slide 17: Results - Sharpe ratio

Now, let us analyze the empirical results.

First we compare the out-of-sample Sharpe ratio between GAN
models and benchmark factor model. We found that GAN models
generally achieve a higher Sharpe ratio than the four-factor
model.

We notice that the GAN model trained with only factor data
performed worse than the factor model. Our hypothesis here
is the limited covariates in this GAN model affect the
neural network estimation. 

Next, we compare the out-of-sample Sharpe ratio between GAN
models trained with macroeconomic data to those without.

We notice that GAN models trained with the macroeconomic
data achieve a higher Sharpe ratio than those without.
Suggesting that including macroeconomic data is important in
SDF estimation.

Furthermore, the best performing GAN model is trained on
factor data with macroeconomic data.

Sharpe ratio provide a summary statistics to compare the
models, but why are GAN models better performing, and what
are the key covariates required for SDF estimation?

These are the questions we will be exploring next

## Slide 18: Results - Variable Importance

To better understand how each variable affects the SDF
weight, we computed a sensitivity score
The sensitivity score measures the average absolute gradient
of the SDF weight, as a function of the individual
covariates.

Figure 3 shows the individual variable importance. We
normalise the sum to be 1 and a higher score indicates a
more important variable.

The most important variable belong to the
extended Fama French factors, followed by firm specific past
return data and macroeconomic indexes affecting consumers,
business and general stock market.

One interesting observation is that the UK SDF weight function is
also affected by the US S&P 500. 
This suggests that we could potentially improve the UK SDF
estimation with the US market data.

## Slide 19: Results - SDF structure

We analyse the SDF weight structure to better understand why
the GAN model is able to outperform the four-factor model.

We plot the SDF weight structure as a function of the
individual factors. With reference to the first image in
Figure 4, the x-axis is the value of the high minus low
while the y-axis is the SDF weight. All other factors were
kept at their mean values in this image.

As shown in the first image, there is a near linear
relationship between high minus low and the SDF weight.
This might be the reason why linear factor model is able to
perform well.

We can investigate the interaction effect between factors by
varying the value of other factors.
The different colored lines in the second image indicate
different values of market risk.
We notice that as we vary the market risk, the impact of
high minus low on the SDF weight changes. 
Image three and four shows similar observations when we
change small minus big and momentum values.

This observation suggests that there is an interaction effect
between the factors. Therefore, models that are able to
take into consideration these interaction effects, 
such as neural network model, will outperform the factor
model.

Lastly, the observation here also holds when we look at
factor other than high minus low and is consistent with the
result in Chen's US SDF estimation.

## Slide 20: Discussion - Limitation (40 sec)

No paper is perfect, so is ours.

One key limitation of this paper is the considerable gap
between Chen's dataset and our dataset.

Chen's data includes 50 years of historical data with over
10,000 stocks while our paper contains 20 years of data with
942 stocks.

The data-poor nature of the UK market could be one of the
crucial reason why most literature focus on the US data.
Even the paid subscription provider was not able to furnish
complete and rich data comparable to the US.

Nonetheless, the results in this paper are still relevant in
assessing the performance of the GAN model when subject to
data limitations.

# Slide 21: Conclusion (50 sec)

In conclusion, our paper set to examine the external
validity of Chen's GAN model using the UK LSE 1998 - 2017
data, and we deliver four key findings.

Firstly, we found the GAN model indeed outperformed
benchmark four-factor model in terms of Sharpe ratio.

Secondly, similar to Chen's finding, including macroeconomic
data improves the SDF estimation.

Thirdly, the extended fama french factors are the most
important covariates in SDF estimation, and they are nearly
linear in the SDF, explaining the wide popularity of fama
french model.

Lastly, there are interaction effects between factors and
models that is able to take into consideration these
interaction effects will outperform the fama french factor
model.

# Slide 22: Thank you and questions (8 sec)

With that we have come to the of our presentation.
Thank you, and please feel free to ask any questions.
