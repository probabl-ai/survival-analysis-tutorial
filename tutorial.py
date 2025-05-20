# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# # What is survival analysis, and when do you need it?
#
# Program for this talk:
#
# 1. What is right-censored time-to-event data and why naive regression models fail on
#    such data?
# 2. Modeling
#    - 2.1 Unconditional survival analysis with Kaplan Meier
#    - 2.2 Conditional survival analysis with Cox Proportional Hazards, a log-linear
#      estimator
#    - 2.3 Conditional survival analysis and competing risk with SurvivalBoost, a non
#      linear boosting tree method.
# 3. Use-cases and limitations
# ---
#
# %% [markdown]
#
# ## 1. What is right-censored time-to-event data?
#
# ### 1.1 Censoring
#
# Survival analysis is a **time-to-event regression** problem, with censored data. We
# call censored all individuals that didn't experience the event during the range of the
# observation window.
#
# In our setting, we're mostly interested in right-censored data, meaning we that the
# event of interest did not occur before the end of the observation period (typically
# the time of collection of the dataset):
#
# <figure>
# <img src="assets/censoring.png" style="width:80%">
# <figcaption align = "center"> <i>image credit: scikit-survival</i> </figcaption>
# </figure>
#
# Individuals can join the study at the same or different times, and the study may or
# may not be ended by the time of observation.
#
# Survival analysis techniques have wide applications:
#
# - In the **medical** landscape, events can consist in patients dying of cancer, or on
#   the contrary recovering from some disease.
# - In **predictive maintenance**, events can consist in machine failures.
# - In **insurance**, we are interesting in modeling the time to next claim for a
#   portfolio of insurance contracts.
# - In **marketing**, we can consider a user churning as an event, or we could focus on
#   users becoming premium (members that choose to pay a subscription after having used
#   the free version of a service for a while).
#
# Answering questions like:
# - What is the probability that a new user churn in 6 months? 1 year? 5 years?
# - How long does it take for this user to churn with a probability of 50%?
#
# As we will see, for all those applications, it is not possible to directly train a
# machine learning-based regression model on such a **right-censored** time-to-event
# target since we only have a lower bound on the true time to event for some data
# points. **Naively removing such points from the dataset would cause the model
# predictions to be biased**.
#
# %% [markdown]
#
### 1.2 Our Tasks
#
# Some notations:
#
# We denote the observed time-to-event $T = \min(T^*, C)$, where:
# - $T^* \in \mathbb{R}_+$ is the event time for an hypothetical unlimited observation
#   window.
# - $C \in \mathbb{R}_+$ is the censoring time
#
# In summary, we can observe $(\bold{X},T,\Delta) \sim D$, where:
# - $\bold{X} \sim \mathcal{X}$ are our covariates: information about the
#   individuals at the time of making the prediction.
# - $T = \min(T^*, C)$ with $T \in \mathbb{R}_+$, the censored time-to-event.
# - $\Delta = T^* < C$ with $ \Delta \in [0, 1]$, the event indicator, 0 is a
#   censored observation.
#
# We sometimes assemble all the observable information related to the target as
# the joint $Y = (T, \Delta)$.
#
# However, we are primarily interested in the conditional distribution $T^*|\bold{X}$.
#
# Our main quantities of interest to estimate are:
#
# - **The Survival Function** represents the probability that an event doesn't occur at
#   or before some given time $t$:
#
# $$S^*(t|\bold{x})=P(T^*>t|\bold{X=x})$$
#
# - **The Cumulative Incidence Function** is the inverse of the survival function, and
#   represents the probability that an event occur before some given time $t$:
#
# $$F^*(t|\bold{x}) = 1 - S^*(t|\bold{x}) = P(T^*\leq t|\bold{X=x})$$

# %% [markdown]
#
# ### 1.3 Our target `y`
#
# For each individual $i\in[1, N]$, our survival analysis target $y_i$ is comprised of
# two elements:
#
# - The event indicator $\delta_i\in\{0, 1\}$, where $0$ marks censoring and $1$ is
#   indicative that the event of interest has actually happened before reaching
#   the end of the observation window.
# - The censored time-to-event $t_i=\min(t^*_{i}, c_i) > 0$, that is the
#   minimum between the date of the hypothetical event $t^*_i$ and the censoring
#   date $c_i$. In a real-world setting, we don't have direct access to $t^*_i$
#   when $\delta_i=0$. We can only record $t_i$.
#
# We define the compound target $y_i = (t_i, \delta_i)$ as a tuple of the two
# observables for each individual.
#
# Here is how we represent our target for a synthetic predictive maintenance
# dataset of 10,000 observations collected by the operator of a fleet of
# trucks:

# %%
from sklearn.datasets import fetch_file
import pandas as pd

DATA_URL = "https://github.com/probabl-ai/survival-analysis-tutorial/releases/download/data-2025-05-19/"
file_url = DATA_URL + "truck_failure_10k_any_event.parquet"
y = pd.read_parquet(fetch_file(file_url, folder="truck_dataset"))
y.round(1)

# %% [markdown]
#
# ### 1.4 Why is it a problem to train time-to-event regression models?
#
# Without survival analysis, we have two naive options to deal with right-censored time
# to event data:
# - We ignore censored data points from the dataset, only keep events that happened and
#   perform naive regression on them.
# - We consider that all censored events happen at the end of our observation window.
#
# **Both approaches are wrong and lead to biased results.**
#
# Let's compute the average and median time to event using either of those naive
# approaches on our truck failure dataset:

# %%
y.loc[y["event"]]["duration"].median().round(1)

# %%
y_max_impute = y.copy()
y_max_impute.loc[~y["event"], "duration"] = y_max_impute["duration"].max()
y_max_impute["duration"].median().round(1)

# %% [markdown]
# We can compare them to the mean of the ground-truth event time $T^*$, that we
# would obtained with an infinite observation window.
#
# Note that we can access to the random variable $T^*$ because we generated
# this synthetic dataset. With real-world data, you only have access to $T =
# \min(T^*, C)$, where $C$ is a random variable representing the censoring
# time.

# %%
file_url = DATA_URL + "truck_failure_10k_any_event_uncensored.parquet"
y_uncensored = pd.read_parquet(fetch_file(file_url, folder="truck_dataset"))
y_uncensored["duration"].median().round(1)

# %% [markdown]
# We can see that either naive approach leads to a biased estimate of the
# median time. The ground-truth median time to event lies between the two naive
# estimates.

# %% [markdown]
#
# ## 2. Modeling
#
# Let's start with unconditional estimation of the any event survival curve.
#
# ### 2.1 Unconditional survival analysis with Kaplan-Meier
#
# We now introduce the survival analysis approach to the problem of estimating the
# time-to-event from censored data. For now, we ignore any information from $X$ and
# focus on $y$ only.
#
# Here our quantity of interest is the survival probability:
#
# $$S(t)=P(T > t)$$
#
# This represents the probability that an event doesn't occur at or before some given
# time $t$, i.e. that it happens at some time $T > t$.
#
# The most commonly used method to estimate this function is the **Kaplan-Meier**
# estimator. It gives us an **unbiased estimate of the survival probability**. It can be
# computed as follows:
#
# $$\hat{S}(t)=\prod_{i: t_i\leq t} (1 - \frac{d_i}{n_i})$$
#
# Where:
#
# - $t_i$ is the time of event for individual $i$ that experienced the event,
# - $d_i$ is the number of individuals having experienced the event at $t_i$,
# - $n_i$ are the remaining individuals at risk at $t_i$.
#
# Note that **individuals that were censored before $t_i$ are no longer considered at
# risk at $t_i$**.
#
# Contrary to machine learning regressors, this estimator is **unconditional**: it only
# extracts information from $y$ only, and cannot model information about each individual
# typically provided in a feature matrix $X$.
#
# In a real-world application, we aim at estimating $\mathbb{E}[T]$ or $Q_{50\%}[T]$.
# The latter quantity represents the median survival duration i.e. the duration before
# 50% of our population at risk experiment the event.
#
# We can also be interested in estimating the survival probability after some reference
# time $P(T > t_{ref})$, e.g. a random clinical trial estimating the capacity of a drug
# to improve the survival probability after 6 months.

# %%
from lifelines import KaplanMeierFitter

km = KaplanMeierFitter()
km.fit(
    durations=y["duration"],
    event_observed=y["event"],
)
ax = km.plot_survival_function()
ax.axhline(y=0.5, linestyle="--", color="r", label="median")
ax.set_ylabel("Survival Probability")
ax.set_ylim(0, 1)
ax.legend()
# %% [markdown]
#
# We can read the median time to event directly from this curve: it is the
# time at the intersection of the estimate of the survival curve with the horizontal
# line for a 50% probability of failure.
#
# Since we have censored data, $\hat{S}(t)$ doesn't reach 0 within our observation
# window. We would need to extend the observation window to estimate the survival
# function beyond this limit. **Kaplan-Meier does not attempt the extrapolate beyond the
# last observed event**.

# %% [markdown]
#
# ### 2.2 Kaplan Meier on Subgroup of `X`
#
# We can enrich our analysis by introducing covariates, that are statistically
# associated to the events and durations.

# %%
file_url = DATA_URL + "truck_failure_10k_features.parquet"
X = pd.read_parquet(fetch_file(file_url, folder="truck_dataset"))
df = X.join(y)
df

# %% [markdown]
# Let's stratify using the brand variable:

# %%
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
for brand_name, group in df.groupby("brand"):
    (
        KaplanMeierFitter()
        .fit(
            durations=group["duration"],
            event_observed=group["event"],
        )
        .plot_survival_function(ax=ax, label=brand_name)
    )
ax.set_ylabel("Survival Probability")
ax.set_ylim(0, 1)

# %% [markdown]
#
# ### 2.3 Model evaluation
#
# The Brier score and the C-index are measures that **assess the quality of a predicted
# survival curve** on a finite data sample.
#
# #### Integrated Brier Score (IBS)
#
# The **[Brier
# score](https://soda-inria.github.io/hazardous/generated/hazardous.metrics.brier_score_survival.html)
# is a proper scoring rule**, meaning that an estimate of the survival curve has minimal
# Brier score if and only if it matches the true survival probabilities induced by the
# underlying data generating process. In that respect the **Brier score** assesses both
# the **calibration** and the **ranking power** of a survival probability estimator.
#
# It is comprised between 0 and 1 (lower is better). It answers the question "how close
# to the real probabilities are our estimates?".
#
# <figure>
# <img src="assets/BrierScore.svg" style="width:80%">
# </figure>
#
# #### Concordance-Index (C-index)
#
# The **C-index** only assesses the **ranking power**: it is invariant to a monotonic
# transform of the survival probabilities. It only focus on the ability of a predictive
# survival model to identify which individual is likely to fail first out of any pair of
# two individuals. It answers the question "given two individuals, how likely are we to
# predict in the correct order that one has experienced the event before the other?"
#
# Conceptually, it is quite similar to the Kendall's Tau coefficient or the ROC AUC. It
# is also comprised between 0 and 1, where 1 means perfect ranking and 0.5 is equivalent
# to a random ranking.
#
# Let's put this in practice.
# %%
import numpy as np
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
km = KaplanMeierFitter().fit(
    durations=y_train["duration"],
    event_observed=y_train["event"],
)

observed_times = y_train.loc[y_train["event"]]["duration"]
time_grid = np.quantile(observed_times, np.linspace(0, 1, 30))
y_pred_km = km.survival_function_at_times(time_grid).values

y_pred_km = np.repeat(y_pred_km[None, :], y_test.shape[0], axis=0)
y_pred_km.shape

# %%
from hazardous.metrics import (
    concordance_index_incidence,
    brier_score_survival,
    integrated_brier_score_survival,
)


class Scorer:
    def __init__(self):
        self.brier_scores = dict()
        self.ibs = dict()
        self.c_index = dict()

    def __call__(self, model_name, y_train, y_test, y_pred, time_grid):
        self.brier_scores[model_name] = brier_score_survival(
            y_train, y_test, y_pred, time_grid
        )
        self.ibs[model_name] = float(
            integrated_brier_score_survival(y_train, y_test, y_pred, time_grid)
        )
        self.c_index[model_name] = float(
            concordance_index_incidence(
                y_test, 1 - y_pred, y_train, time_grid=time_grid
            )
        )

        _, ax = plt.subplots()
        for name, brier_score in self.brier_scores.items():
            label = (
                f"{name} - IBS: {self.ibs[name]:.4f}"
                f" - C-index: {self.c_index[name]:.4f}"
            )
            ax.plot(time_grid, brier_score, label=label)
        ax.set_ylabel("Time-dependent Brier score")
        ax.set_xlabel("Time horizon")
        ax.legend()


scorer = Scorer()
scorer("Kaplan Meier", y_train, y_test, y_pred_km, time_grid)

# %% [markdown]
#
# We observed that the "prediction error" is largest for time horizons between 200 and
# 1500 days after the beginning of the observation period.
#
# Additionally, we compute the Integrated Brier Score (IBS) which we will use to
# summarize the Brier score curve and compare the quality of different estimators of the
# survival curve on the same test set: $$IBS = \frac{1}{t_{max} -
# t_{min}}\int^{t_{max}}_{t_{min}} BS(t) dt$$
#
# This is equivalent to a random prediction. Indeed, as our Kaplan Meier is a
# unconditional estimator: it can't be used to rank individuals predictions as it
# predicts the same survival curve for any row in `X_test`.
#
# Next, we'll study how to fit survival models that make predictions conditional on `X`.
#
# ### 2.4 Cox Proportional Hazards
#
# The hazard rate $\lambda(t)$ represents the "speed of failure" or **the probability
# that an event occurs in the next $dt$, given that it hasn't occured yet**. This can be
# written as:
#
# $$\begin{align} \lambda(t) &=\lim_{dt\rightarrow 0}\frac{P(t \leq T < t + dt | P(T
# \geq t))}{dt} \\
# &= \lim_{dt\rightarrow 0}\frac{P(t \leq T < t + dt)}{dtS(t)} \\
# &= \frac{f(t)}{S(t)} \end{align} $$
#
# The Cox PH model is the most popular way of dealing with covariates $X$ in survival
# analysis. It computes a log linear regression on the target $Y = \min(T, C)$, and
# consists in a baseline term $\lambda_0(t)$ and a covariate term with weights $\beta$.
# $$\lambda(t, x_i) = \lambda_0(t) \exp(x_i^\top \beta)$$
#
# Note that only the baseline depends on the time $t$, but we can extend Cox PH to
# time-dependent covariate $x_i(t)$ and time-dependent weigths $\beta(t)$. We won't
# cover these extensions in this tutorial.
#
# This methods is called ***proportional*** hazards, since for two different covariate
# vectors $x_i$ and $x_j$, their ratio is: $$\frac{\lambda(t, x_i)}{\lambda(t, x_j)} =
# \frac{\lambda_0(t) e^{x_i^\top \beta}}{\lambda_0(t) e^{x_j^\top
# \beta}}=\frac{e^{x_i^\top \beta}}{e^{x_j^\top \beta}}$$
#
# This ratio is not dependent on time, and therefore the hazards are proportional.
#
# Let's run it on our truck-driver dataset using the implementation of `lifelines`. This
# models requires preprocessing of the categorical features using One-Hot encoding.
# Let's use the scikit-learn column-transformer to combine the various components of the
# model as a pipeline:

# %%
from skrub import TableVectorizer
from lifelines.fitters.coxph_fitter import CoxPHFitter

df_train = X_train.join(y_train)
df_test = X_test.join(y_test)

vectorizer = TableVectorizer()
df_train = vectorizer.fit_transform(df_train)
df_test = vectorizer.transform(df_test)

cox = CoxPHFitter(penalizer=1e-2).fit(
    df_train, duration_col="duration", event_col="event"
)
y_pred_cox = cox.predict_survival_function(df_test, times=time_grid).to_numpy().T
y_pred_cox.shape

# %%
scorer("Cox PH Fitter", y_train, y_test, y_pred_cox, time_grid)

# %% [markdown]
#
# So the Cox Proportional Hazard model from lifelines fitted as a simple pipeline
# with one-hot encoded categorical variables and raw numerical variables seems already
# significantly better than our unconditional baseline.
#
# Let's now display the survival curves of the first 5 trucks-driver pairs. %%


# %%
def plot_survival_curves(y_pred, time_grid, n_curves=5):
    _, ax = plt.subplots()
    for idx in range(n_curves):
        ax.plot(time_grid, y_pred[idx], label=str(idx))
    ax.set_ylim(0, 1)
    ax.set_ylabel("S(t)")
    ax.set_title(r"Survival Probabilities $\hat{S}(t)$ of Cox PH")
    ax.axhline(y=0.5, linestyle="--", color="r", alpha=0.5)
    ax.axvline(x=1000, linestyle="--", color="b", alpha=0.5)
    ax.legend()
    ax.grid()


plot_survival_curves(y_pred_cox, time_grid)
# %% [markdown]
#
# We see that predicted survival functions can vary significantly for different test
# samples.
#
# There are two ways to read this plot:
#
# First we could consider our **predictive survival analysis model as a probabilistic
# regressor**: if we want to **consider a specific probability of survival, say 50%**,
# we can mentally draw an horizontal line at 0.5, and see that:
#
# - test data point `#0` has an estimated median survival time around 300 days,
# - test data point `#1` has an estimated median survival time around 800 days,
# - test data point `#2` has an estimated median survival time around 450 days...
#
# Secondly we could also consider our **predictive survival analysis model as a
# probabilistic binary classifier**: if we **consider a specific time horizon, say 1000
# days**, we can see that:
#
# - test data point `#0` has less than a 20% chance to remain event-free at day 1000,
# - test date point `#3` has around a 50% chance to remain event-free at day 1000...
#
# Let's try to get some intuition about the features importance from the first 5
# truck-driver pairs and their survival probabilities.

# %%
import matplotlib as mpl

(
    np.log(cox.hazard_ratios_.sort_values(ascending=False)).plot.barh(
        facecolor=mpl.color_sequences["tab10"], grid=True
    )
)

# %% [markdown]
#
# ### 2.5 SurvivalBoost
#
# We now introduce a novel survival estimator named
# [`SurvivalBoost`](https://soda-inria.github.io/hazardous/generated/hazardous.SurvivalBoost.html),
# based on sklearn's
# [`HistGradientBoostingClassifer`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
# It can estimate cause-specific cumulative incidence functions in a competing risks
# setting, by minimizing a cause specific proper scoring rule objective.
#
# We just published a paper detailing the Survival Boost model at the AISTATS 2025
# conference.

# %%
from hazardous import SurvivalBoost
from sklearn.preprocessing import OrdinalEncoder


vectorizer = TableVectorizer(high_cardinality=OrdinalEncoder())
X_train_trans = vectorizer.fit_transform(X_train)
X_test_trans = vectorizer.transform(X_test)

surv_boost = SurvivalBoost(show_progressbar=False).fit(X_train_trans, y_train)
y_pred_survboost = surv_boost.predict_survival_function(X_test_trans, time_grid)

scorer("Survival Boost", y_train, y_test, y_pred_survboost, time_grid)

# %% [markdown]
#
# SurvivalBoost gives great performance on the Brier Score, however C-index is slightly
# under the log-linear model for this simplistic dataset.

# %%
from sklearn.inspection import permutation_importance


permutations = permutation_importance(surv_boost, X_test_trans, y_test)

(
    pd.Series(permutations["importances_mean"], index=X_test_trans.columns)
    .sort_values(ascending=True)
    .plot.barh(facecolor=mpl.color_sequences["tab10"])
)
# %%

from sklearn.inspection import PartialDependenceDisplay

for percentile in [0.25, 0.75]:
    horizon = np.quantile(observed_times, percentile)
    t = np.full(shape=X_test_trans.shape[0], fill_value=horizon)
    X_test_trans_at_t = pd.concat(
        [
            pd.DataFrame(dict(t=t)),
            X_test_trans.reset_index(drop=True),
        ],
        axis="columns",
    )
    plt.figure(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(
        surv_boost.estimator_,
        X_test_trans_at_t.to_numpy(),
        response_method="predict_proba",
        method="brute",
        features=["driver_skill", "usage_rate"],
        feature_names=X_test_trans_at_t.columns,
    )
    _ = plt.suptitle(f"Marginal effects at t={horizon:.0f} days")

    plt.figure(figsize=(8, 8))
    PartialDependenceDisplay.from_estimator(
        surv_boost.estimator_,
        X_test_trans_at_t.to_numpy(),
        response_method="predict_proba",
        method="brute",
        features=[("driver_skill", "usage_rate")],
        feature_names=X_test_trans_at_t.columns,
    )
    _ = plt.suptitle(f"Interaction effects at t={horizon:.0f} days")


# %%
plot_survival_curves(y_pred_survboost, time_grid)
# %% [markdown]
#
# ## 3. Discussions and limits
#
# TODO
