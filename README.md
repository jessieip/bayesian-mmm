# 1. Project Objective

Marketing spend is across various channels, both online and offline. It is hard to measure the attribution per channel and allocate the budget to reach to max ROI. 

This project implements a **Bayesian Marketing Mix Model (MMM)** to solve the following questions:
- the current ROI and marginal ROI per channel
- Identify which marketing channels are currently saturated (diminishing returns).
- Identify which channels have room to grow (under-saturated).
- Provide actionable, tactical recommendations for next week's budget allocation (e.g., "Decrease Paid Search spend because it hit saturation; Increase Display spend by X% to capture remaining potential").

> 📊 **Data Infrastructure**: Synthetic marketing data is generated via Python scripts (`generate_synthetic.py`) and seamlessly integrated with **Supabase** for centralized storage and analytics streaming.

---

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
Note: Fill in the project name, version, and description. Select "no" when asked to define dependencies interactively.


Run below to set up Python environment and activate
```bash
poetry env use python
```
Configure pyproject.toml to enforce Python 3.12 compatibility and disable package mode since this is a standalone application:
```bash
[tool.poetry]
package-mode = false

[tool.poetry.dependencies]
requires-python = ">=3.12, <3.15"
```

```bash
poetry env activate
```
 
Using poetry to add libraries:
```bash
poetry add numpy pandas matplotlib seaborn datetime pymc arvia supabse streamlit
```

```bash
poetry install
```
## 2) IDE Interpreter set up
1. Go to the bottom right corner of your IDE -> Interpreter Settings.

2. Click Add Interpreter -> Local Interpreter.

3. Select Poetry Environment and point it to your local environment executable.

## 3) Windows Performance Optimisation (MSVC C++ Compiler)

PyMC utilizes advanced MCMC samplers that require local C++ compilation for hyper-speed processing.

1. Install Visual Studio Community (with "Desktop development with C++" checked).

2. Add the MSVC compiler path to your Windows System Environment Variables:
```
C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
```

## 3. Git Hygiene (.gitignore)
To prevent multi-gigabyte virtual environments, IDE configuration file leaks, adding below files to .gitignore to avoid uploading unnecessary or sensitivity data
```bash
.venv/
.idea/
**/__pycache__/
**.pyc
data/synthetic/*.csv
**/.ipynb_checkpoints/
```

## 4. Methodology & Feature Engineering

Adstock Effect: Implemented geometric_adstock() to mimic media decay and carryover effects.

Saturation Curve: Applied S-curve hill_saturation() profiles to capture diminishing returns.

Control Features: Integrated external macro-dynamics (Competitor Spend, Google Trends) governed by Gamma Priors to isolate pure marketing incrementality.

MCMC Sampling Configuration:
Configured with 4 parallel chains, executing 1,000 tuning steps and 1,000 draw samples per chain.

Convergence diagnostic checks performed via ArviZ trace plots (az.plot_trace).


# 5. Model Results & Business Insights

This section outlines the workflow and core insights derived from our Bayesian Marketing Mix Model (PyMC-Marketing), tracking from prior configuration to business recommendations.

---

## 1). Model Configuration & Priors Setting

We initialized the model by injecting domain-specific priors to guide the Bayesian inference process:
* **Intercept & Baseline**: Set the baseline sales environment.
* **Media Transformations**: Configured Geometric Adstock (lagging effect) and Hill/Logistic Saturation curves.
* **Control Variables**: Incorporated external factors (e.g., competitor spend, Google Trends) via Gamma priors.
* **MCMC Sampling**: Configured **4 parallel chains** to explore the posterior distribution and identify parameter ranges.

> 💡 **Understanding the Posterior Predictive Chart**: 
> In the posterior predictive time-series chart, the **central line** represents the **mean (or median) prediction** of the model, while the shaded band represents the **94% Highest Density Interval (HDI)**. If the observed sales heavily fall within this 94% HDI, it indicates that the model successfully captured the system's variance within an acceptable confidence range.

---

## 2). Training Diagnostics (Convergence Check)

* **Trace Chart Analysis**: 
  In a healthy MCMC sampling process, the parameter traces across all 4 chains should overlay perfectly, resembling a "well-mixed caterpillar." 
  * *Observation*: If your trace charts show **separated lines** among chains, it indicates that the model **failed to converge** (each chain converged to a different local optimum). For synthetic data, this suggests potential multi-modality or structural under-specification in the priors/data relationship.

---

## 3). Model Results & Business Insights

### 📊 In-Sample Fit & Volatility
The historical sales timeline falls smoothly within the model's 94% Highest Density Interval (HDI) band, signaling robust posterior tracking. However, empirical data exhibits slightly lower volatility compared to simulation bounds, suggesting future iterations could benefit from additional smoothing control variables.

### 📈 Channel Contribution Over Time
* **PPC Brand**: Demonstrates prominent seasonality (peaking annually around Q1/Spring). A wider HDI shadow suggests **lower relative confidence** in exact point-returns, advising cautious scaling.
* **Social**: Displays a highly narrow HDI interval, indicating exceptional statistical confidence and stability.
* **Waterfall Decomposition**: **TV (27%)** and **OOH (16%)** dominate baseline volume generation, reinforcing the strength of top-of-funnel offline media. PPC Brand and Social drive agile digital volume, holding ~10% share each.
* *Note on Flat Trajectories*: While Meta, PPC Generic, and OOH look flat over time, this merely implies that historical weekly pacing was kept constant, not necessarily that they are inefficient.

---

## 6. Media Transformation & Behavior Analysis

### ⏳ Adstock Decay Rates (Retention)
* **PPC Brand (26%)**: Fast decay rate, aligning with transactional direct-response behaviors where search intent fades quickly.
* **PPC Generic** Multi-modal distribution peaks detected (12%, 15\%, and 40%). This suggests potential multicollinearity with other concurrent channels or an overly wide prior specification,

### 🌊 Saturation Levels
* **PPC Brand** follows a distinct **S-shaped (Sigmoid) saturation curve**. 
* It shows that a minimum baseline threshold is required before acceleration, followed by an aggressive plateau.

---

## 7. Strategic Recommendations (Sensitivity Analysis)

The sensitivity analysis sweeps budget from **0x to 1.5x** of historical averages to simulate budget optimization:

| Saturation Status (At 1.0x Historical Spend) | Channels | Actionable Business Strategy |
| :--- | :--- | :--- |
| **Fully Saturated** (Flat after 1.0x) | Display, Meta, PPC Generic, Social, Yahoo | **Hold or Cap Budget**: Additional dollars into these channels yield zero marginal sales. Maintain current spend levels to avoid waste. |
| **Under-Saturated** (Growth continues after 1.0x) | **PPC Brand, OOH, TV** | **Scale Up Budget**: These channels still operate on the linear/growth section of the curve. Scaling spend here will capture incremental volume. |

## 8. ROAS per channel
* **Display**: The mean ROAS is 0.016, and the model has 94% HDI (confidence) that the ROAS lies between 0.01 and 0.02.
* **Social**: Social has the highest ROAS at 0.037, while the model estimates the ROAS is between 0.00 and 0.07. It is highly volatile (fluctuates heavily).
* **Other channels**' peaks are relatively small. In other words, the model cannot find an obvious ROAS interval for them.

## 9. Out Sample Validation

The synthetic data might not have reached convergence, meaning the lagging effects and saturation levels might not be perfectly aligned with reality.

However, the 5-week forward forecast perfectly aligns within the 94% HDI predictive intervals, proving strong generalizability for corporate budget planning.