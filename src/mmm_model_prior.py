import pymc as pm
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, HillSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.prior import Prior
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import os


logger = logging.getLogger(__name__)
def mmm_model_prior(dataset: pd.DataFrame, prior_sigma: float) -> tuple[pd.DataFrame, pd.Series, MMM.mmm, plt.fig]:
    """
    Setting up MMM priors and doing Prior Predictive Check
    Args:
    Dataset: it is from Supabase with date, opportunities and channel spend.
    prior_sigma: prior sigma per channels which are from data_process function.
    Returns:
        tuple: (X, y, mmm model, fig)
    X: independent variables
    y: dependent variables
    mmm: MMM model
    fig: priors plot to check the model performance before training.
    """

    mingw_path = r"C:\msys64\mingw64\bin"
    if os.path.exists(mingw_path) and mingw_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.pathsep + mingw_path

    if not os.environ["PATH"]:
        raise ValueError("Path not found")

    alpha_list = [1, 1, 1, 1, 3, 3, 1, 1]
    beta_list = [3, 3, 3, 3, 1, 1, 3, 3]

    y = dataset["opportunities"].copy()
    y.name = "y"
    X = dataset.drop("opportunities", axis=1)

    my_model_config = {
        "intercept": Prior("Normal", mu=0.5, sigma=2),
        "saturation_beta": Prior("HalfNormal", sigma=prior_sigma, dims="channel"),
        "gamma_control": Prior("Normal", mu=0, sigma=0.5, dims="control"),
        "gamma_fourier": Prior("Laplace", mu=0, b=0.5, dims="fourier_mode"),
        "adstock_alpha": Prior(
            "Beta", alpha=alpha_list, beta=beta_list, dims="channel"
        ),
        "saturation_slope": Prior("Gamma", alpha=3, beta=1, dims="channel"),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=6)),
    }

    my_sampler_config = {
        "progressbar": True,  # display the status bar
        "draws": 1000,  # sampling 1000 times
        "chains": 4,  # 4 chains
        "nuts_sampler": "numpyro",
    }

    mmm = MMM(
        model_config=my_model_config,
        sampler_config=my_sampler_config,
        date_column="date",
        target_column="y",
        adstock=GeometricAdstock(l_max=8),
        saturation=HillSaturation(),
        channel_columns=[
            "PPC_Brand_Spend",
            "PPC_Generic_Spend",
            "Display_Spend",
            "Social_Spend",
            "TV_Spend",
            "OOH_Spend",
            "Meta_Spend",
            "Yahoo_Spend",
        ],
        control_columns=["competitor_spend", "google_trend_competitor"],
        yearly_seasonality=2,
    )

    mmm.build_model(X, y)
    mmm.add_original_scale_contribution_variable(
        var=[
            "channel_contribution",
            "control_contribution",
            "intercept_contribution",
            "yearly_seasonality_contribution",
            "y",
        ]
    )
    try:
        display(pm.model_to_graphviz(mmm.model))
    except NameError:
        pass

    logger.info("MMM Priors Imported")
    # Generate prior predictive samples
    mmm.sample_prior_predictive(X, y, samples=2_000)
    mmm.plot.prior_predictive()

    # Custom plot for prior predictive checks
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, hdi_prob in enumerate([0.94, 0.5]):
        az.plot_hdi(
            x=mmm.model.coords["date"],
            y=mmm.idata["prior"]["y_original_scale"].unstack().transpose(..., "date"),
            smooth=False,
            color="C0",
            hdi_prob=hdi_prob,
            fill_kwargs={"alpha": 0.3 + i * 0.1, "label": f"{hdi_prob: .0%} HDI"},
            ax=ax,
        )
    sns.lineplot(
        data=dataset, x="date", y="sales", color="black", label="observed", ax=ax
    )
    ax.legend(loc="upper left")
    ax.set(xlabel="date", ylabel="sales")
    ax.set_title("Prior Predictive Checks", fontsize=18, fontweight="bold")
    return X, y, mmm, fig