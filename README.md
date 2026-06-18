# 1. Project Objective

## Marketing spend is across various channels, both online and offline. It is hard to measure the attribution per channel and allocate the budget to reach to max ROI. 

## Therefore, establish a Bayesian MMM to figure out the below questions:
## - the current ROI and marginal ROI per channel
## - Identify which marketing channels are currently saturated (diminishing returns).
## - Identify which channels have room to grow (under-saturated).
## - Provide actionable, tactical recommendations for next week's budget allocation (e.g., "Decrease Paid Search spend because it hit saturation; Increase Display spend by X% to capture remaining potential").

### Data source: data is generated from generate_synthetic.py and store in Supabase

# 2. Environment Set up

## 1) Python version and virtual environment set up

Run below to check the default Python version:

```bash
python --version
```

Initialise the Poetry Project

```bash
poetry init
```
You will populate the package name, Version, Description, etc.

After that, it will pop up below questions and select 'no':

Would you like to define your main dependencies interactively?
Would you like to define your development dependencies interactively?

Run below to set up Python environment and activate
```bash
poetry env use python
```
In pyprojct.toml, set the below version to ensure using Python 3.12:
```bash
requires-python = ">=3.12, <3.15"
```

```bash
poetry env activate
```
 
Using poetry to add libraries:
```bash
poetry add numpy pandas matplotlib seaborn datetime pymc arvia supabse streamlit
```
In pyproject.toml, set up package-mode
```
[tool.poetry]
package-mode = false
```

```bash
poetry install
```
## 2) Interpreter set up
Go to the IDE bottom right - Interpreter Setting - Add Interpreter - Local Interpreter
Select **Poetry**

(how to add picture?)

You will see the installed packages under that interpreter.

## 3) Install Visual Code C++ to speed up PyMC processing

Install Visual Code then add the below path to Environment Variable - System Variable
```
C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
```

## 3. Git Hygiene (.gitignore)
Recommend adding below files to .gitignore to avoid uploading unnecessary or sensitivity data
```bash
.venv/
.idea/
**/__pycache__/
**.pyc
data/synthetic/*.csv
**/.ipynb_checkpoints/
```

## 4. Methodology & Feature Engineering

Prior Distribution: we set up some priors such as baseline(intercept), saturation level(saturation beta), external factor(gamma control), seasonality (gamma fourier), noise (likelihood), adstock lagging effect(adstock alpha), saturatio lambda(control the s curve) 
Sampling setting: 4 chains, each one will sample 1000 times and using numpyro to speed up the fitting process.
In model setting, using HillSaturation() as it can capture the ROI per channel more accurately. 

Model fitting: Using plot_trace from arivz to view the trace for parameters.


# Marketing Mix Modeling (MMM) Analysis Report

This section outlines the workflow and core insights derived from our Bayesian Marketing Mix Model (PyMC-Marketing), tracking from prior configuration to business recommendations.

---

## 1. Model Configuration & Priors Setting

We initialized the model by injecting domain-specific priors to guide the Bayesian inference process:
* **Intercept & Baseline**: Set the baseline sales environment.
* **Media Transformations**: Configured Geometric Adstock (lagging effect) and Hill/Logistic Saturation curves.
* **Control Variables**: Incorporated external factors (e.g., competitor spend, Google Trends) via Gamma priors.
* **MCMC Sampling**: Configured **4 parallel chains** to explore the posterior distribution and identify parameter ranges.

> 💡 **Understanding the Posterior Predictive Chart**: 
> In the posterior predictive time-series chart, the **central line** represents the **mean (or median) prediction** of the model, while the shaded band represents the **94% Highest Density Interval (HDI)**. If the observed sales heavily fall within this 94% HDI, it indicates that the model successfully captured the system's variance within an acceptable confidence range.

---

## 2. Training Diagnostics (Convergence Check)

* **Trace Chart Analysis**: 
  In a healthy MCMC sampling process, the parameter traces across all 4 chains should overlay perfectly, resembling a "well-mixed caterpillar." 
  * *Observation*: If your trace charts show **separated lines** among chains, it indicates that the model **failed to converge** (each chain converged to a different local optimum). For synthetic data, this suggests potential multi-modality or structural under-specification in the priors/data relationship.

---

## 3. Model Results & Business Insights

### 📊 In-Sample Fit & Volatility
The observed sales successfully fall within the 94% HDI band. However, the historical trend does not perfectly align with the mean prediction, and the observed sales appear **less volatile** than the model's simulated expectations. This mismatch indicates that the model might be over-reacting to spend spikes or missing key smoothing control variables.

### 📈 Channel Contribution Over Time
* **PPC Brand**: Demonstrates a clear seasonal pattern, consistently peaking around March/April annually. However, it exhibits a wider HDI band (larger shadow), representing **lower estimation confidence** regarding its exact ROI.
* **Social**: Stands out as the only channel with a highly narrow HDI range, signaling **high statistical confidence** in its performance stability.
* **Channel Share Decomposition**: Based on the *Waterfall chart*, **TV (27%)** and **OOH (16%)** dominate total business impact, proving that offline channels remain heavy baseline drivers. **PPC Brand** and **Social** each account for roughly **10%** of total impact.
* *Note on Flat Trajectories*: While Meta, PPC Generic, and OOH look flat over time, this merely implies that historical weekly pacing was kept constant, not necessarily that they are inefficient.

---

## 4. Media Transformation & Behavior Analysis

### ⏳ Retention Rate (Adstock Effect)
* **PPC Brand** possesses an adstock alpha of **26%**. This implies that 26% of the advertisement's psychological impact carries over into the following week, exhibiting a relatively fast decay rate typical for direct-response search behavior.

### 🌊 Saturation Levels
* **PPC Brand** follows a distinct **S-shaped (Sigmoid) saturation curve**. 
  * *Low Spend Bracket*: Small investments fail to trigger an uplift (the threshold effect).
  * *Growth & Plateau Bracket*: Once budget clears a critical threshold, sales lift accelerates sharply before rapidly hitting a diminishing-returns plateau.



---

## 5. Strategic Recommendations (Sensitivity Analysis)

The sensitivity analysis sweeps budget from **0x to 1.5x** of historical averages to simulate budget optimization:

| Saturation Status (At 1.0x Historical Spend) | Channels | Actionable Business Strategy |
| :--- | :--- | :--- |
| **Fully Saturated** (Flat after 1.0x) | Display, Meta, PPC Generic, Social, Yahoo | **Hold or Cap Budget**: Additional dollars into these channels yield zero marginal sales. Maintain current spend levels to avoid waste. |
| **Under-Saturated** (Growth continues after 1.0x) | **PPC Brand, OOH, TV** | **Scale Up Budget**: These channels still operate on the linear/growth section of the curve. Scaling spend here will capture incremental volume. |