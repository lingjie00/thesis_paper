# Generative Adversarial Network (GAN) model in Asset Pricing

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

## Slide 6: methodology
