import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
from plt_conv_rates import theoretical_curve
from matplotlib.patches import Patch, PathPatch



correlations = [0, 0.3, 0.6, 0.9]
models = [
    "lr", "lasso", "dt", "rf", "et", "gb", "hgb",
    "ab", "bag", "mlp", "svr", "knn", "xgb",
    "SuperLearner", "TabICL"
]

settings = ["gaussian_quad", "nongaussian_quad","linear_sparse", "interaction_sparse","friedman","classification_lin","classification_cplx"]

for setting in settings:
    for correlation_strength in correlations:
        csv_files = glob.glob(f"csv/asymp_relevance_setting{setting}_corr{correlation_strength}_seed*.csv"
        )

        if not csv_files:
            print(f"No files found for {setting}, corr={correlation_strength}. Skipping.")
            continue

        try:
            df = pd.concat(
                (pd.read_csv(f) for f in csv_files),
                ignore_index=True,
            )
        except Exception as e:
            print(f"Error reading files for {setting}, corr={correlation_strength}: {e}")
            continue

        # Models available for this setting
        models_order = [
            "lr", "lasso", "dt", "rf", "et", "gb", "hgb",
            "ab", "bag", "mlp", "svr", "knn",
            "SuperLearner", "TabICL"
        ]

        models_present = [m for m in models_order if m in df["model"].unique()]

        sample_sizes = sorted(df["n_samples"].unique())

        # ------------------------------------
        # Compute mean importance per seed
        # ------------------------------------
        imp_cols = sorted(
            [c for c in df.columns if c.startswith("imp_V") and not c.startswith("imp_std")],
            key=lambda x: int(x.split("V")[1])
        )

        true_imp = []
        null_imp = []
        true_std_seed = []
        null_std_seed = []

        for _, row in df.iterrows():

            imp_true = []
            imp_null = []

            std_true = []
            std_null = []

            for imp in imp_cols:

                idx = imp.split("V")[1]

                if row[f"tr_V{idx}"] == 1:
                    imp_true.append(row[imp])
                    std_true.append(row[f"imp_std_V{idx}"])
                else:
                    imp_null.append(row[imp])
                    std_null.append(row[f"imp_std_V{idx}"])

            true_imp.append(np.mean(imp_true))
            null_imp.append(np.mean(imp_null))

            true_std_seed.append(np.mean(std_true))
            null_std_seed.append(np.mean(std_null))

        df["true_imp"] = true_imp
        df["null_imp"] = null_imp
        df["true_std_seed"] = true_std_seed
        df["null_std_seed"] = null_std_seed


        # ------------------------------------
        # Aggregate over seeds
        # ------------------------------------
        summary = (
            df.groupby(["model", "n_samples"])
            .agg(
                true_mean=("true_imp", "mean"),
                true_std=("true_imp", "std"),
                null_mean=("null_imp", "mean"),
                null_std=("null_imp", "std"),
                true_std_seed=("true_std_seed", "mean"),
                null_std_seed=("null_std_seed", "mean"),
            )
            .reset_index()
        )

        summary["true_std"] = summary["true_std"].fillna(0)
        summary["null_std"] = summary["null_std"].fillna(0)

        summary.loc[summary["true_std"] == 0, "true_std"] = \
            summary.loc[summary["true_std"] == 0, "true_std_seed"]

        summary.loc[summary["null_std"] == 0, "null_std"] = \
            summary.loc[summary["null_std"] == 0, "null_std_seed"]
        # ------------------------------------
        # Pivot
        # ------------------------------------
        true_mean = (
            summary.pivot(index="model", columns="n_samples", values="true_mean")
            .reindex(models_present)
        )

        true_std = (
            summary.pivot(index="model", columns="n_samples", values="true_std")
            .reindex(models_present)
        )

        null_mean = (
            summary.pivot(index="model", columns="n_samples", values="null_mean")
            .reindex(models_present)
        )

        null_std = (
            summary.pivot(index="model", columns="n_samples", values="null_std")
            .reindex(models_present)
        )

        true_mean.columns = [f"N={n}" for n in sample_sizes]
        true_std.columns = [f"N={n}" for n in sample_sizes]
        null_mean.columns = [f"N={n}" for n in sample_sizes]
        null_std.columns = [f"N={n}" for n in sample_sizes]

        true_annot = (
            true_mean.round(3).astype(str)
            + "\n("
            + true_std.round(3).astype(str)
            + ")"
        )

        null_annot = (
            null_mean.round(3).astype(str)
            + "\n("
            + null_std.round(3).astype(str)
            + ")"
        )

        # ------------------------------------
        # Shared color scale
        # ------------------------------------
        vmin = min(true_mean.min().min(), null_mean.min().min())
        vmax = max(true_mean.max().max(), null_mean.max().max())

        # ------------------------------------
        # Plot
        # ------------------------------------
        sns.set_theme(style="white")
        sns.set_context(
            "talk",
            rc={
                "axes.titlesize": 22,
                "axes.labelsize": 20,
                "xtick.labelsize": 16,
                "ytick.labelsize": 20,
            },
        )

        fig, axes = plt.subplots(
            1, 2,
            figsize=(20, 0.75*len(models_present)+4),
            sharey=True
        )
        #fig, axes = plt.subplots(1, 2, figsize=(18, 0.6*len(models_present)+3), sharey=True)
        
        hm1 = sns.heatmap(
            true_mean,
            annot=true_annot,
            fmt="",
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            annot_kws={"size":14},
            ax=axes[0],
        )

        axes[0].set_title("Relevant variables", fontsize=24)
        axes[0].set_ylabel("Model", fontsize=24)

        hm2 = sns.heatmap(
            null_mean,
            annot=null_annot,
            fmt="",
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            annot_kws={"size":14},
            ax=axes[1],
        )

        axes[1].set_title("Irrelevant variables",fontsize=24)
        axes[1].set_ylabel("")


        cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.75])
        fig.colorbar(hm2.collections[0], cax=cbar_ax)

        cbar = fig.colorbar(hm2.collections[0], cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        #cbar.set_label("Mean importance", fontsize=18)

        plt.tight_layout(rect=[0,0,0.9,1])

        plt.savefig(
            f"figures/asympt_relevance/importance_heatmap_{setting}_corr{correlation_strength}.pdf",
            bbox_inches="tight",
        )

        plt.close()