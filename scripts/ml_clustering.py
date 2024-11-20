import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from hmmlearn import hmm
import matplotlib.patches as mpatches

# Load and prepare data
file_path = (
    "/home/LilAiluropoda/Projects/fyp_project/app/data/stock_data/AAPL/AAPL19972007.csv"
)
data = pd.read_csv(file_path, header=None)
data.columns = ["Date"] + ["Feature"] * (data.shape[1] - 2) + ["Adj Close"]
data["Date"] = pd.to_datetime(data["Date"])

# Calculate returns and additional features
adj_close = data["Adj Close"].values
returns = np.log(adj_close[1:] / adj_close[:-1])
volatility = pd.Series(returns).rolling(window=20).std().fillna(method="bfill").values

# Create feature matrix for HMM
# Using both returns and volatility as features
X = np.column_stack([returns, volatility[1:]])

# Standardize features
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# Initialize and fit HMM
model = hmm.GaussianHMM(
    n_components=3, covariance_type="full", n_iter=1000, random_state=42
)
model.fit(X_std)

# Predict states
hidden_states = model.predict(X_std)

# Calculate regime characteristics
regime_stats = []
for i in range(model.n_components):
    mask = hidden_states == i
    regime_returns = returns[mask]
    regime_stats.append(
        {
            "state": i,
            "mean_return": np.mean(regime_returns),
            "volatility": np.std(regime_returns),
            "count": np.sum(mask),
        }
    )

# Classify regimes based on return/risk characteristics
regime_stats.sort(key=lambda x: x["mean_return"])
regime_labels = ["Bear Market", "Neutral Market", "Bull Market"]
state_to_label = {
    stat["state"]: label for stat, label in zip(regime_stats, regime_labels)
}

# Create results DataFrame
results_df = pd.DataFrame(
    {
        "Date": data["Date"].values[1:],
        "Adjusted Close": adj_close[1:],
        "Returns": returns,
        "Volatility": volatility[1:],
        "HMM State": hidden_states,
    }
)

# Plotting
plt.figure(figsize=(15, 10))

# Create custom colormap
colors = {"Bull Market": "green", "Neutral Market": "gray", "Bear Market": "red"}
state_colors = [colors[state_to_label[state]] for state in results_df["HMM State"]]

# Plot price with regime coloring
plt.plot(results_df["Date"], results_df["Adjusted Close"], "k-", alpha=0.3)
plt.scatter(
    results_df["Date"], results_df["Adjusted Close"], c=state_colors, alpha=0.6, s=20
)

# Create custom legend
legend_elements = []
for label, color in colors.items():
    mask = (
        results_df["HMM State"]
        == [k for k, v in state_to_label.items() if v == label][0]
    )
    regime_data = results_df[mask]
    label_text = (
        f"{label}\n"
        f"Avg Return: {100*np.mean(regime_data['Returns']):.2f}%\n"
        f"Volatility: {100*np.std(regime_data['Returns']):.2f}%\n"
        f"Days: {len(regime_data)}"
    )
    legend_elements.append(mpatches.Patch(color=color, label=label_text, alpha=0.6))

plt.legend(
    handles=legend_elements,
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    title="Market Regimes",
)

plt.title("Hidden Markov Model Market Regimes", fontsize=12)
plt.xlabel("Date", fontsize=10)
plt.ylabel("Adjusted Close Price", fontsize=10)
plt.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Save the plot
plt.savefig("market_regime_analysis_improved.png", bbox_inches="tight", dpi=300)
plt.show()

# Print transition matrix
print("\nRegime Transition Probabilities:")
print(
    pd.DataFrame(
        model.transmat_,
        columns=[state_to_label[i] for i in range(3)],
        index=[state_to_label[i] for i in range(3)],
    )
)

# Print detailed statistics
print("\nDetailed Regime Statistics:")
for state in range(model.n_components):
    mask = results_df["HMM State"] == state
    regime_data = results_df[mask]
    print(f"\n{state_to_label[state]}:")
    print(f"Average Daily Return: {100*np.mean(regime_data['Returns']):.3f}%")
    print(f"Return Volatility: {100*np.std(regime_data['Returns']):.3f}%")
    print(f"Number of Days: {len(regime_data)}")
    print(
        f"Price Range: ${regime_data['Adjusted Close'].min():.2f} - ${regime_data['Adjusted Close'].max():.2f}"
    )
