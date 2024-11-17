import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from hmmlearn import hmm
import matplotlib.patches as mpatches

# =============== PARAMETERS ===============
# File Parameters
FILE_PATH = (
    "/home/LilAiluropoda/Projects/fyp_project/app/data/stock_data/AAPL/AAPL19972007.csv"
)
OUTPUT_FILE = "statistical_regime_analysis.png"

# Kalman Filter Parameters
KF_TRANSITION_COV = 1e-5
KF_OBSERVATION_COV = 1e-2
KF_INITIAL_STATE_COV = 1e-2

# HMM Parameters
HMM_N_COMPONENTS = 3
HMM_COVARIANCE_TYPE = "diag"
HMM_N_ITER = 1000

# Minimum window of regimes
MIN_WINDOW_SIZE = 30  # Minimum number of days a regime must persist

# Plotting Parameters
FIGURE_SIZE = (15, 10)
PLOT_DPI = 300
GRID_ALPHA = 0.3
SCATTER_ALPHA = 0.6
TITLE_FONTSIZE = 12
LABEL_FONTSIZE = 10

# Visual Parameters
LINE_COLORS = {"adjusted_close": "blue", "kalman_estimate": "orange"}
LINE_ALPHA = 0.5

# Regime Labels
REGIME_LABELS = ["Regime 1", "Regime 2", "Regime 3"]
REGIME_COLORS = plt.cm.viridis(np.linspace(0, 1, HMM_N_COMPONENTS))


# =============== FUNCTIONS ===============
def enforce_minimum_window(hidden_states):
    """Enforce a minimum window for regime persistence."""
    current_state = hidden_states[0]
    count = 1

    for i in range(1, len(hidden_states)):
        if hidden_states[i] == current_state:
            count += 1
        else:
            if count < MIN_WINDOW_SIZE:
                # Change the previous states to the current state
                hidden_states[i - count : i] = current_state
            current_state = hidden_states[i]
            count = 1

    return hidden_states


def characterize_regime(regime_data):
    """Characterize regime based on statistical properties"""
    mean_return = regime_data["Returns"].mean() * 252  # Annualized return
    volatility = regime_data["Returns"].std() * np.sqrt(252)  # Annualized volatility
    sharpe = mean_return / volatility if volatility != 0 else 0

    return {"mean_return": mean_return, "volatility": volatility, "sharpe": sharpe}


def get_regime_description(stats):
    """Get statistical description of regime"""
    return (
        f"μ={stats['mean_return']*100:.1f}%, "
        f"σ={stats['volatility']*100:.1f}%, "
        f"SR={stats['sharpe']:.2f}"
    )


def run_hmm_analysis(year_data):
    """Run HMM analysis for a single year of data"""
    adj_close_year = year_data["Adj Close"].values.reshape(-1)
    returns = np.log(adj_close_year[1:] / adj_close_year[:-1]).reshape(-1, 1)

    model = hmm.GMMHMM(
        n_components=HMM_N_COMPONENTS,
        covariance_type=HMM_COVARIANCE_TYPE,
        n_iter=HMM_N_ITER,
    )

    try:
        model.fit(returns)
        hidden_states = model.predict(returns)
        hidden_states = enforce_minimum_window(hidden_states)  # Enforce minimum window
    except:
        print(f"Warning: HMM fitting failed. Using random states for continuation.")
        hidden_states = np.random.randint(0, HMM_N_COMPONENTS, size=len(returns))

    return returns, hidden_states, model


def create_regime_statistics(results_df, hidden_states):
    """Calculate regime statistics"""
    regime_stats = []
    for i in range(HMM_N_COMPONENTS):
        mask = results_df["HMM State"] == i
        regime_data = results_df[mask]

        stats = characterize_regime(regime_data)
        regime_stats.append(
            {
                "state": i,
                "mean_return": stats["mean_return"],
                "volatility": stats["volatility"],
                "sharpe": stats["sharpe"],
                "count": len(regime_data),
            }
        )

    regime_stats.sort(key=lambda x: x["sharpe"])
    return regime_stats


def plot_year_analysis(ax, year_data, hidden_states, year, regime_stats):
    """Plot HMM analysis for a single year"""
    scatter = ax.scatter(
        year_data["Date"].values[1:],
        year_data["Adj Close"].values[1:],
        c=hidden_states,
        cmap="viridis",
        alpha=SCATTER_ALPHA,
    )

    ax.set_title(f"{year}", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Price", fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA)

    legend_elements = []
    for i, label in enumerate(REGIME_LABELS):
        stats = regime_stats[i]
        label_text = (
            f"{label}\n" f"{get_regime_description(stats)}\n" f"Days: {stats['count']}"
        )
        legend_elements.append(
            mpatches.Patch(
                color=REGIME_COLORS[i], label=label_text, alpha=SCATTER_ALPHA
            )
        )

    ax.legend(handles=legend_elements, loc="upper left", fontsize="small")
    return scatter


# =============== MAIN CODE ===============
# Load data
data = pd.read_csv(FILE_PATH, header=None)
data.columns = ["Date"] + ["Feature"] * (data.shape[1] - 2) + ["Adj Close"]
data["Date"] = pd.to_datetime(data["Date"])

# Split data by year
years = sorted(data["Date"].dt.year.unique())
n_years = len(years)

# Create subplot grid
n_cols = 2
n_rows = (n_years + 1) // 2
fig = plt.figure(figsize=(15, 5 * n_rows))

# Store yearly analysis
yearly_analysis = {}

# Analyze each year
for i, year in enumerate(years, 1):
    print(f"\nAnalyzing year {year}...")

    year_data = data[data["Date"].dt.year == year].copy()
    returns, hidden_states, model = run_hmm_analysis(year_data)

    results_df = pd.DataFrame(
        {
            "Date": year_data["Date"].values[1:],
            "Adjusted Close": year_data["Adj Close"].values[1:],
            "HMM State": hidden_states,
            "Returns": returns.flatten(),
        }
    )

    regime_stats = create_regime_statistics(results_df, hidden_states)

    ax = plt.subplot(n_rows, n_cols, i)
    scatter = plot_year_analysis(ax, year_data, hidden_states, year, regime_stats)

    yearly_analysis[year] = {
        "results_df": results_df,
        "regime_stats": regime_stats,
        "model": model,
    }

    print(f"\nYear {year} Statistical Regime Analysis:")
    print("-" * 40)
    for j, stats in enumerate(regime_stats):
        print(f"\n{REGIME_LABELS[j]}:")
        print(f"Number of days: {stats['count']}")
        print(f"Annualized return: {stats['mean_return']*100:.2f}%")
        print(f"Annualized volatility: {stats['volatility']*100:.2f}%")
        print(f"Sharpe ratio: {stats['sharpe']:.2f}")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, bbox_inches="tight", dpi=PLOT_DPI)
plt.show()

# Regime distribution analysis
print("\nStatistical Regime Distribution Analysis")
print("-" * 50)

regime_distributions = pd.DataFrame(
    index=years,
    columns=[f"Regime {i+1}" for i in range(HMM_N_COMPONENTS)],
    data=np.zeros((len(years), HMM_N_COMPONENTS)),
)

for year in years:
    analysis = yearly_analysis[year]
    for i in range(HMM_N_COMPONENTS):
        mask = (
            analysis["results_df"]["HMM State"] == analysis["regime_stats"][i]["state"]
        )
        regime_distributions.loc[year, f"Regime {i+1}"] = mask.mean() * 100

print("\nRegime Distribution by Year (%):")
print(regime_distributions.round(2))

# Transition analysis
print("\nRegime Transition Analysis:")
print("-" * 50)

for year in years:
    analysis = yearly_analysis[year]
    model = analysis["model"]

    if hasattr(model, "transmat_"):
        print(f"\nYear {year} Transition Probabilities:")
        transition_matrix = pd.DataFrame(
            model.transmat_,
            columns=[f"To Regime {i+1}" for i in range(HMM_N_COMPONENTS)],
            index=[f"From Regime {i+1}" for i in range(HMM_N_COMPONENTS)],
        )
        print(transition_matrix.round(3))

# Year-to-year regime evolution
print("\nYear-to-Year Regime Evolution:")
print("-" * 50)

for i in range(len(years) - 1):
    year1, year2 = years[i], years[i + 1]
    analysis1 = yearly_analysis[year1]
    analysis2 = yearly_analysis[year2]

    last_state = analysis1["results_df"]["HMM State"].iloc[-1]
    first_state = analysis2["results_df"]["HMM State"].iloc[0]

    print(f"\nTransition from {year1} to {year2}:")
    print(f"From {REGIME_LABELS[last_state]} → {REGIME_LABELS[first_state]}")
