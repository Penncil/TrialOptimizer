import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch.autograd as autograd
import matplotlib.pyplot as plt
import time
import math
from torch.optim.lr_scheduler import StepLR 
from scipy.optimize import minimize
from scipy.stats import norm
import warnings   
 
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# ---  generate_subgroups --- 


def generate_subgroups(criteria_relaxed, criteria_fixed):
    """
    Generate all subgroups from fixed and relaxed criteria.
    
    Parameters
    ----------
    criteria_relaxed : list of str
        Criteria that can vary (optional).
    criteria_fixed : list of str
        Criteria that are always included.
    
    Returns
    -------
    pandas.DataFrame
        Table of all subgroups with subgroup_id, n_criteria, and criteria string.
    """
    
    subgroups = []
    n = len(criteria_relaxed)
    
    # Subgroup C0: only fixed criteria
    subgroups.append({
        "subgroup_id": "C0",
        "n_criteria": len(criteria_fixed),
        "criteria": ", ".join(criteria_fixed),
        "binary_inclusion": [0] * n
    })
    
     
    subgroup_counter = 1
    
    # Generate all non-empty subsets of relaxed criteria
    for r in range(1, n+1):
        for combo in itertools.combinations(criteria_relaxed, r):
            combined = list(combo) + list(criteria_fixed)
            binary = [1 if c in combo else 0 for c in criteria_relaxed]
            subgroups.append({
                "subgroup_id": f"C{subgroup_counter}",
                "n_criteria": len(combined),
                "criteria": ", ".join(combined),
                "binary_inclusion": binary
            })
            subgroup_counter += 1
    
    return pd.DataFrame(subgroups)



 
 
from scipy.stats import gaussian_kde
 
from statsmodels.nonparametric.kde import KDEUnivariate

def density_R(x, n=512):
    """
    Mimics R's density() using:
    - Gaussian kernel
    - Sheather Jones bandwidth (default in R)
    - 512 points (same as R default)
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]

    kde = KDEUnivariate(x)
    kde.fit(kernel="gau", bw=0.05,  gridsize=n) 

    return kde.support, kde.density


def calculate_g_index(ps_no_restrict, ps_tm, n_grid=512): 
    """
    G-index matching R's density() behavior.
    """

    # KDEs using R settings
    x_pop, f_pop = density_R(ps_no_restrict, n=n_grid)
    x_trial, f_trial = density_R(ps_tm, n=n_grid)

    # Overlapping support
    x_min = max(x_pop.min(), x_trial.min())
    x_max = min(x_pop.max(), x_trial.max())

    # If no overlap
    if x_min >= x_max:
        return 0.0

    # Target evaluation grid
    x_grid = np.linspace(x_min, x_max, n_grid)

    # Linear interpolation (like R's approx)
    f_pop_interp = np.interp(x_grid, x_pop, f_pop, left=0, right=0)
    f_trial_interp = np.interp(x_grid, x_trial, f_trial, left=0, right=0)

    # Pointwise minimum
    integrand = np.minimum(f_pop_interp, f_trial_interp)

    # Riemann sum
    g_index = np.sum(integrand) * (x_grid[1] - x_grid[0])

    return float(g_index)

 

# Compute AE rates

def AE_rates(df):
    # 1. ITT new onset indicator
    df["new_onset_irAE_ITT"] = np.where(
        (df["irAE_baseline_binary"] == 0) &
        (df["irAE"].notna()) &
        (df["irAE"] != ""),
        1, 0
    )

    # 2. PP new onset indicator
    df["new_onset_irAE_PP"] = np.where(
        (df["irAE_baseline_binary"] == 0) &
        (df["irAE"].notna()) &
        (df["irAE"] != "") &
        (
            (pd.to_datetime(df["irAE"]) <= pd.to_datetime(df["EarlierEndDate"])) |
            (df["irAE"] == "")
        ),
        1, 0
    )

    # 3. AE rate ITT
    AE_ITT = (df["irAE"] != "").sum() / len(df)

    # 4. New onset AE rate ITT
    AE_new_onset_ITT = df.loc[df["irAE_baseline_binary"] == 0, "new_onset_irAE_ITT"].mean()

    # 5. AE rate PP
    mask_pp = (pd.to_datetime(df["irAE"]) <= pd.to_datetime(df["EarlierEndDate"])) | (df["irAE"] == "")
    flag_pp = (df.loc[mask_pp, "irAE"] != "").astype(int)
    AE_PP = flag_pp.sum() / df.shape[0]

    # 6. New onset AE rate PP
    AE_new_onset_PP = df.loc[df["irAE_baseline_binary"] == 0, "new_onset_irAE_PP"].mean()

   
    return   AE_new_onset_PP
#UNO

def gaussian_product(mu1, mu2, sd1, sd2):
    sqr = lambda x: x ** 2
    denominator = np.sqrt(2 * np.pi) * np.sqrt(sqr(sd1) + sqr(sd2))
    exponent = -sqr(mu1 - mu2) / (2 * (sqr(sd1) + sqr(sd2)))
    return (1 / denominator) * np.exp(exponent)

def log_likelihood_null(theta, logRr, seLogRr):
    if theta[1] <= 0:
        return np.inf

    result = 0.0
    sd = 1 / np.sqrt(theta[1])

    if sd < 1e-6:
        for i in range(len(logRr)):
            result -= norm.logpdf(theta[0], loc=logRr[i], scale=seLogRr[i])
    else:
        for i in range(len(logRr)):
            gaussian_val = gaussian_product(logRr[i], theta[0], seLogRr[i], sd)
            if gaussian_val <= 0:
                return np.inf
            result -= np.log(gaussian_val)

    if result == 0:
        return np.inf
    return result

def fit_null(logRr, seLogRr):
    logRr = np.array(logRr)
    seLogRr = np.array(seLogRr)

    if np.any(np.isinf(seLogRr)):
        warnings.warn("Estimate(s) with infinite standard error detected. Removing before fitting null distribution")
        valid_indices = ~np.isinf(seLogRr)
        logRr = logRr[valid_indices]
        seLogRr = seLogRr[valid_indices]

    if np.any(np.isinf(logRr)):
        warnings.warn("Estimate(s) with infinite logRr detected. Removing before fitting null distribution")
        valid_indices = ~np.isinf(logRr)
        logRr = logRr[valid_indices]
        seLogRr = seLogRr[valid_indices]

    if np.any(np.isnan(seLogRr)):
        warnings.warn("Estimate(s) with NA standard error detected. Removing before fitting null distribution")
        valid_indices = ~np.isnan(seLogRr)
        logRr = logRr[valid_indices]
        seLogRr = seLogRr[valid_indices]

    if np.any(np.isnan(logRr)):
        warnings.warn("Estimate(s) with NA logRr detected. Removing before fitting null distribution")
        valid_indices = ~np.isnan(logRr)
        logRr = logRr[valid_indices]
        seLogRr = seLogRr[valid_indices]

    if len(logRr) == 0:
        warnings.warn("No estimates remaining")
        return {"mean": np.nan, "sd": np.nan}

    theta = np.array([0, 1])  # Updated initial guess
    result = minimize(
        log_likelihood_null,
        theta,
        args=(logRr, seLogRr),
        method='L-BFGS-B',
        bounds=[(None, None), (1e-6, None)]  # Ensure theta[1] is positive
    )

    if result.success:
        mean, precision = result.x
        sd = 1 / np.sqrt(precision)
        return {"mean": mean, "sd": sd}
    else:
        warnings.warn("Optimization failed")
        return {"mean": np.nan, "sd": np.nan}


def closed_form_integral(x, mu, sigma):
    """
    Closed-form integral for given x, mean (mu), and standard deviation (sigma).
    """
    return mu * norm.cdf(x, loc=mu, scale=sigma) - 1 - sigma**2 * norm.pdf(x, loc=mu, scale=sigma)

def closed_form_integral_absolute(mu, sigma):
    """
    Computes the closed-form integral for the absolute value of systematic error.
    """
    return (closed_form_integral(np.inf, mu, sigma)
            - 2 * closed_form_integral(0, mu, sigma)
            + closed_form_integral(-np.inf, mu, sigma))
    
def compute_expected_absolute_systematic_error_null(null, alpha=0.05):
    """
    Compute the expected absolute systematic error for a null distribution.

    Parameters:
    - null: A dictionary containing 'mean' and 'sd' of the null distribution.
    - alpha: Significance level (default is 0.05).

    Returns:
    - Expected absolute systematic error (float).
    """
    if null["mean"] == 0 and null["sd"] == 0:
        return 0

    mean = null["mean"]
    sd = null["sd"]

    # Closed-form integral for absolute value of systematic error
    return closed_form_integral_absolute(mean, sd)

 

def forest_plot(logRR, se_logRR, labels=None, title="Forest Plot"):
    """
    Generates a forest plot using log risk ratios and their standard errors.

    Parameters:
    - logRR: List or array of log risk ratios.
    - se_logRR: List or array of standard errors of the log risk ratios.
    - labels: List of labels for each entry (optional).
    - title: Title of the forest plot (default "Forest Plot").
    """
    # Calculate confidence intervals
    ci_low = logRR - 1.96 * se_logRR
    ci_high = logRR + 1.96 * se_logRR
    
    # Exponentiate to get back to RR scale
    rr = np.exp(logRR)
    ci_low_exp = np.exp(ci_low)
    ci_high_exp = np.exp(ci_high)
    
    # Number of data points
    n = len(logRR)
    
    # If labels are not provided, create default numeric labels
    if labels is None:
        labels = [f"Study {i+1}" for i in range(n)]
    
    # Create the forest plot
    plt.figure(figsize=(8, 6))
    y_positions = range(n)
    
    # Add vertical line at RR=1
    plt.axvline(1, color="red", linestyle="--", linewidth=1, label="RR = 1 (No Effect)")
    
    # Plot each point and its confidence interval
    for i in range(n):
        plt.errorbar(
            rr[i], y_positions[i], 
            xerr=[[rr[i] - ci_low_exp[i]], [ci_high_exp[i] - rr[i]]], 
            fmt='.', color='blue', capsize=3
        )


#define NN for propensity score

class PS(nn.Module):
    def __init__(self, in_N, m, depth=2):
        super(PS, self).__init__()
         
        self.stack = nn.ModuleList() 
        self.stack.append(nn.Linear(in_N, m))
        for i in range(depth-1):
            self.stack.append(nn.Linear(m, m))
        self.stack.append(nn.Linear(m, 1))
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.stack[0](x))
        x = self.dropout(x)   
        for i in range(1,len(self.stack)-1):
            x =  self.act(self.stack[i](x))   
            x = self.dropout(x)   
        x = self.stack[-1](x)
        x = 1e-2+ (0.98)*self.sigmoid(x)       
        return  x
    
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


#define NN for prune

class M_pruned(nn.Module):

    def __init__(self, in_N, m, depth=2):
        super(M_pruned, self).__init__()
         
        self.stack = nn.ModuleList() 
        self.stack.append(nn.Linear(in_N, m)) 
        for i in range(depth-1):
            self.stack.append(nn.Linear(m, m)) 
        self.stack.append(nn.Linear(m, 1))
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()
        # Dropout layer 
        self.dropout = nn.Dropout(0.1)
         

    def forward(self, x):
        x = self.act(self.stack[0](x))
        x = self.dropout(x)
        for i in range(1,len(self.stack)-1):
            x =  self.act(self.stack[i](x))    
            x = self.dropout(x)
        x = self.stack[-1](x)
        x = 1e-2+ (0.98)*self.sigmoid(x)   
        return  x
    
    def pre_act(self, x):
        xi = self.stack[0](x)
        pre_act_list = [xi]
        x = self.act(xi)
        x = self.dropout(x)
        for i in range(1,len(self.stack)-1):
            xi = self.stack[i](x)
            pre_act_list.append(xi)
            x =  self.act(xi)  
            x = self.dropout(x)
        x = self.stack[-1](x)
        x = 1e-2+ (0.98)*self.sigmoid(x)    
        return  x, pre_act_list

     

def test_NCO(nco, Xn, Wn):
    NCO_pred = nco.predict_proba(Xn)[:,1]
    # Convert probabilities to binary outcomes (0 or 1) based on threshold (0.5)
    NCO_pred_binary = (NCO_pred >= 0.5) 
    act = Wn.squeeze()
    # Calculate accuracy
    accuracy = np.sum(NCO_pred_binary == act )/act.shape[0]
    #print(f'Test Accuracy: {accuracy.item():.4f}')
    return NCO_pred
