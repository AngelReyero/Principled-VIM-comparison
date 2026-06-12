import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
from plt_conv_rates import theoretical_curve
from matplotlib.patches import Patch, PathPatch



# Dummy data setup (replace with your actual df)
methods = [
    "Sobol-CPI(1)",
    "Sobol-CPI(100)",
    "PFI",
    "CFI",
    "cSAGEvf",
    "scSAGEj",
    "mSAGEvf",
    "cSAGE",
    "mSAGE",
    "LOCO",
    "LOCO-W",
    "LOCI"
]

features = [f'X{i}' for i in range(12)]
p=12

correlations = [0, 0.3, 0.6, 0.9]
models = [
    "lr", "lasso", "dt", "rf", "et", "gb", "hgb",
    "ab", "bag", "mlp", "svr", "knn", "xgb",
    "SuperLearner", "TabICL"
]

for regressor in models:
    for correlation_strength in correlations:
        csv_files = glob.glob(
            f"csv/bike/bike_spurious_model{regressor}_corr{correlation_strength}_seed*.csv"
        )

        if not csv_files:
            print(f"No files found for {regressor}, corr={correlation_strength}. Skipping.")
            continue

        try:
            df = pd.concat(
                (pd.read_csv(f) for f in csv_files),
                ignore_index=True,
            )
        except Exception as e:
            print(f"Error reading files for {regressor}, corr={correlation_strength}: {e}")
            continue

        # Process df here

        palette = {
            # Sobol-CPI variants → blues
            "Sobol-CPI(1)": "#1f77b4",    # medium blue
            "Sobol-CPI(100)": "#6baed6",  # lighter blue

            # SAGE family → shades of red-purple
            "cSAGE": "#c44e52",           # reddish-pink
            "mSAGE": "#dd1c77",           # deep pink
            "cSAGEvf": "#e377c2",         # pink
            "mSAGEvf": "#bc80bd",         # lavender
            "scSAGEj": "#9e6db5",         # purple

            # LOCO-related → green/cyan
            "LOCO": "#2ca02c",            # green
            "LOCO-W": "#17becf",          # cyan
            "LOCI": "#a1d99b",            # light green

            # Other methods
            "PFI": "#ff7f0e",             # orange
            "CFI": "#d62728",             # red
        }



        # Normalize
        imp_cols = [col for col in df.columns if col.startswith("imp_V")]
        df[imp_cols] = df[imp_cols].div(df[imp_cols].sum(axis=1), axis=0)
        # Plot boxplot
        methods_to_keep = ['Sobol-CPI(1)', 'Sobol-CPI(100)','CFI', 'scSAGEj', 'LOCO']
        df_filtered = df[df['method'].isin(methods_to_keep)]
        plt.figure(figsize=(16, 8))
        df_long = pd.melt(df_filtered, id_vars='method', 
                        value_vars=[f'imp_V{i}' for i in range(0, p)],
                        var_name='Variable', value_name='Importance')

        # Optional: clean up variable names (e.g., 'imp_V1' -> 'V1')
        df_long['Variable'] = df_long['Variable'].str.replace('imp_', '')

        # Plot

        plt.figure(figsize=(12, 6))
        var_mapping = {
            'V0': 'season',
            'V1': 'yr',
            'V2': 'mnth',
            'V3': 'holiday',
            'V4': 'weekday',
            'V5': 'workingday',
            'V6': 'weathersit',
            'V7': 'temp',
            'V8': 'hum',
            'V9': 'windspeed',
            'V10': 'days_since_2011',
            'V11': 'spurious'
        }

        # Replace the values in 'Variable'
        df_long['Variable'] = df_long['Variable'].map(var_mapping)
        ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)
        plt.ylim(-1, 1)


        ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

        plt.xticks(rotation=45)
        mean_r2 = df['r2'].mean()
        plt.title(fr"Feature importance for bike dataset, $R^2 = {mean_r2:.2f}$", fontsize=25)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.ylabel('Importance', fontsize=20)
        plt.xlabel('Variables', fontsize=20)
        plt.legend(title='Method',  loc='upper left', fontsize=16, title_fontsize=18, ncol=2)#bbox_to_anchor=(1.05, 1),
        plt.tight_layout()
        plt.savefig(f"figures/bike_spurious/bike_spurious_model{regressor}_corr{correlation_strength}.pdf", bbox_inches="tight")




        # Long boxplot:

        # Plot boxplot
        plt.figure(figsize=(16, 8))
        df_long = pd.melt(df, id_vars='method', 
                        value_vars=[f'imp_V{i}' for i in range(0, p)],
                        var_name='Variable', value_name='Importance')

        # Optional: clean up variable names (e.g., 'imp_V1' -> 'V1')
        df_long['Variable'] = df_long['Variable'].str.replace('imp_', '')

        # Plot

        plt.figure(figsize=(12, 6))
        var_mapping = {
            'V0': 'season',
            'V1': 'yr',
            'V2': 'mnth',
            'V3': 'holiday',
            'V4': 'weekday',
            'V5': 'workingday',
            'V6': 'weathersit',
            'V7': 'temp',
            'V8': 'hum',
            'V9': 'windspeed',
            'V10': 'days_since_2011',
            'V11': 'spurious'
        }

        # Replace the values in 'Variable'
        df_long['Variable'] = df_long['Variable'].map(var_mapping)
        ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)


        ax.axhline(0, color='black', linestyle=':', linewidth=0.5)


        # Define hatching
        hatched_methods = {"LOCI", "cSAGE", "cSAGEvf"}
        hatch_style = '///'

        # Fix box patch ordering:
        # Seaborn arranges: for each x (Variable), show all hues (methods)
        # So: box index i ⇒ variable = i // n_methods, method = i % n_methods
        variables = df_long['Variable'].unique().tolist()
        methods = df_long['method'].unique().tolist()
        n_vars = len(variables)
        n_methods = len(methods)

        # Get boxes
        boxes = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
        boxes_sorted = sorted(boxes, key=lambda b: b.get_path().vertices[:, 0].mean())
        for i, patch in enumerate(boxes_sorted):
            var_idx = i // n_methods
            method_idx = i % n_methods
            method = methods[method_idx]

            if method in hatched_methods:
                patch.set_hatch(hatch_style)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.0)
                patch.set_facecolor(patch.get_facecolor())  # to force redraw



        # Legend
        legend_elements = []
        hatch_map = {
            "cSAGE": '///',
            "cSAGEvf": '///',
            "LOCI": '///',
        }

        for method in methods:
            color = palette[method]
            if method in hatched_methods:
                patch = Patch(
                    facecolor=color,
                    edgecolor='black',
                    hatch=hatch_map[method],
                    label=method
                )
            else:
                patch = Patch(
                    facecolor=color,
                    edgecolor='black',
                    label=method
                )
            legend_elements.append(patch)


        plt.xticks(rotation=45)
        mean_r2 = df['r2'].mean()
        plt.title(fr"Feature importance for bike dataset, $R^2 = {mean_r2:.2f}$", fontsize=25)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.ylim(-1, 1)
        plt.ylabel('Importance', fontsize=20)
        plt.xlabel('Variables', fontsize=20)
        plt.legend(handles=legend_elements, title='Method',  loc='upper left', fontsize=16, title_fontsize=18, ncol=1, bbox_to_anchor=(1.05, 1))#bbox_to_anchor=(1.05, 1),
        plt.tight_layout()
        plt.savefig(f"figures/bike_spurious/complete_bike_spurious_model{regressor}_corr{correlation_strength}.pdf", bbox_inches="tight")





        # Plot boxplot
        plt.figure(figsize=(16, 8))
        df_long = pd.melt(df, id_vars='method', 
                        value_vars=[f'imp_V{i}' for i in range(1, 2)],
                        var_name='Variable', value_name='Importance')

        # Optional: clean up variable names (e.g., 'imp_V1' -> 'V1')
        df_long['Variable'] = df_long['Variable'].str.replace('imp_', '')

        # Plot

        plt.figure(figsize=(12, 6))
        var_mapping = {
            'V0': 'season',
            'V1': 'yr',
            'V2': 'mnth',
            'V3': 'holiday',
            'V4': 'weekday',
            'V5': 'workingday',
            'V6': 'weathersit',
            'V7': 'temp',
            'V8': 'hum',
            'V9': 'windspeed',
            'V10': 'days_since_2011',
            'V11': 'spurious'
        }

        # Replace the values in 'Variable'
        df_long['Variable'] = df_long['Variable'].map(var_mapping)
        ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)


        ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

        # Define hatching
        hatched_methods = {"LOCI", "cSAGE", "cSAGEvf"}
        hatch_style = '///'

        # Fix box patch ordering:
        # Seaborn arranges: for each x (Variable), show all hues (methods)
        # So: box index i ⇒ variable = i // n_methods, method = i % n_methods
        variables = df_long['Variable'].unique().tolist()
        methods = df_long['method'].unique().tolist()
        n_vars = len(variables)
        n_methods = len(methods)

        # Get boxes
        boxes = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
        boxes_sorted = sorted(boxes, key=lambda b: b.get_path().vertices[:, 0].mean())
        for i, patch in enumerate(boxes_sorted):
            var_idx = i // n_methods
            method_idx = i % n_methods
            method = methods[method_idx]

            if method in hatched_methods:
                patch.set_hatch(hatch_style)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.0)
                patch.set_facecolor(patch.get_facecolor())  # to force redraw



        # Legend
        legend_elements = []
        hatch_map = {
            "cSAGE": '///',
            "cSAGEvf": '///',
            "LOCI": '///',
        }

        for method in methods:
            color = palette[method]
            if method in hatched_methods:
                patch = Patch(
                    facecolor=color,
                    edgecolor='black',
                    hatch=hatch_map[method],
                    label=method
                )
            else:
                patch = Patch(
                    facecolor=color,
                    edgecolor='black',
                    label=method
                )
            legend_elements.append(patch)


        plt.xticks(rotation=45)
        mean_r2 = df['r2'].mean()
        plt.title(fr"Feature importance for bike dataset, $R^2 = {mean_r2:.2f}$", fontsize=25)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.ylim(-1, 1)
        plt.ylabel('Importance', fontsize=20)
        plt.xlabel('Variables', fontsize=20)
        plt.legend(handles=legend_elements, title='Method',  loc='upper left', fontsize=16, title_fontsize=18, ncol=1, bbox_to_anchor=(1.05, 1))#bbox_to_anchor=(1.05, 1),
        plt.tight_layout()
        plt.savefig(f"figures/bike_spurious/reduced_bike_spurious_model{regressor}_corr{correlation_strength}.pdf", bbox_inches="tight")



        # SPURIOUS

        # Plot boxplot
        plt.figure(figsize=(16, 8))
        df_long = pd.melt(df, id_vars='method', 
                        value_vars=[f'imp_V{i}' for i in range(11, 12)],
                        var_name='Variable', value_name='Importance')

        # Optional: clean up variable names (e.g., 'imp_V1' -> 'V1')
        df_long['Variable'] = df_long['Variable'].str.replace('imp_', '')

        # Plot

        plt.figure(figsize=(12, 6))
        var_mapping = {
            'V0': 'season',
            'V1': 'yr',
            'V2': 'mnth',
            'V3': 'holiday',
            'V4': 'weekday',
            'V5': 'workingday',
            'V6': 'weathersit',
            'V7': 'temp',
            'V8': 'hum',
            'V9': 'windspeed',
            'V10': 'days_since_2011',
            'V11': 'spurious'
        }

        # Replace the values in 'Variable'
        df_long['Variable'] = df_long['Variable'].map(var_mapping)
        ax = sns.boxplot(data=df_long, x='Variable', y='Importance', hue='method', palette=palette)


        ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

        # Define hatching
        hatched_methods = {"LOCI", "cSAGE", "cSAGEvf"}
        hatch_style = '///'

        # Fix box patch ordering:
        # Seaborn arranges: for each x (Variable), show all hues (methods)
        # So: box index i ⇒ variable = i // n_methods, method = i % n_methods
        variables = df_long['Variable'].unique().tolist()
        methods = df_long['method'].unique().tolist()
        n_vars = len(variables)
        n_methods = len(methods)

        # Get boxes
        boxes = [patch for patch in ax.patches if isinstance(patch, PathPatch)]
        boxes_sorted = sorted(boxes, key=lambda b: b.get_path().vertices[:, 0].mean())
        for i, patch in enumerate(boxes_sorted):
            var_idx = i // n_methods
            method_idx = i % n_methods
            method = methods[method_idx]

            if method in hatched_methods:
                patch.set_hatch(hatch_style)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.0)
                patch.set_facecolor(patch.get_facecolor())  # to force redraw



        # Legend
        legend_elements = []
        hatch_map = {
            "cSAGE": '///',
            "cSAGEvf": '///',
            "LOCI": '///',
        }

        for method in methods:
            color = palette[method]
            if method in hatched_methods:
                patch = Patch(
                    facecolor=color,
                    edgecolor='black',
                    hatch=hatch_map[method],
                    label=method
                )
            else:
                patch = Patch(
                    facecolor=color,
                    edgecolor='black',
                    label=method
                )
            legend_elements.append(patch)


        plt.xticks(rotation=45)
        mean_r2 = df['r2'].mean()
        plt.title(fr"Feature importance for bike dataset, $R^2 = {mean_r2:.2f}$", fontsize=25)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        plt.ylim(-1, 1)
        plt.ylabel('Importance', fontsize=20)
        plt.xlabel('Variables', fontsize=20)
        plt.legend(handles=legend_elements, title='Method',  loc='upper left', fontsize=16, title_fontsize=18, ncol=1, bbox_to_anchor=(1.05, 1))#bbox_to_anchor=(1.05, 1),
        plt.tight_layout()
        plt.savefig(f"figures/bike_spurious/spurious_bike_spurious_model{regressor}_corr{correlation_strength}.pdf", bbox_inches="tight")







