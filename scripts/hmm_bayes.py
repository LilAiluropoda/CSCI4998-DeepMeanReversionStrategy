from hmmlearn import hmm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = (
    "/home/LilAiluropoda/Projects/DeepMlpGA/app/data/stock_data/AAPL/AAPL19972007.csv"
)

OUTPUT_FILE = "hmm_bayes"

HMM_N_COMPONENTS = 2
HMM_COVARIANCE_TYPE = "diag"
HMM_N_ITER = 3000

class dc_calculator():
	def __init__(self):
		self.prices = None
		self.time = None
		self.TMV_list = []
		self.T_list = []
		self.colors = []
		self.events = []

	def compute_dc_variables(self, threshold: float = 0.005):
		"""

		Method to compute all relevant DC parameters.

		"""

		if self.prices is None:
			print('Please load the time series data first before proceeding with the DC parameters computation')
		else:
			self.TMV_list = []
			self.T_list = []
			self.colors = []
			self.events = []

			ext_point_n = self.prices[0]
			curr_event_max = self.prices[0]
			curr_event_min = self.prices[0]
			time_point_max = 0
			time_point_min = 0
			trend_status = 'up'
			T = 0

			for i in range(len(self.prices)):
				TMV = (self.prices[i] - ext_point_n) / (ext_point_n * threshold)
				self.TMV_list.append(TMV)
				self.T_list.append(T)
				T += 1

				if trend_status == 'up':
					self.colors.append('lime')
					self.events.append('Upward Overshoot')

					if self.prices[i] < ((1 - threshold) * curr_event_max):
						trend_status = 'down'
						curr_event_min = self.prices[i]

						ext_point_n = curr_event_max
						T = i - time_point_max

						num_points_change = i - time_point_max
						for j in range(1, num_points_change + 1):
							self.colors[-j] = 'red'
							self.events[-j] = 'Downward DCC'
					else:
						if self.prices[i] > curr_event_max:
							curr_event_max = self.prices[i]
							time_point_max = i
				else:
					self.colors.append('lightcoral')
					self.events.append('Downward Overshoot')

					if self.prices[i] > ((1 + threshold) * curr_event_min):
						trend_status = 'up'
						curr_event_max = self.prices[i]

						ext_point_n = curr_event_min			
						T = i - time_point_min

						num_points_change = i - time_point_min
						for j in range(1, num_points_change + 1):
							self.colors[-j] = 'green'
							self.events[-j] = 'Upward DCC'
					else:
						if self.prices[i] < curr_event_min:
							curr_event_min = self.prices[i]
							time_point_min = i

			self.colors = np.array(self.colors)

			print('DC variables computation has finished.')

class online_dc_calculator():
    def __init__(self, threshold: float = 0.005):
        # Data storage
        self.prices = []
        self.time = []
        self.TMV_list = []
        self.T_list = []
        self.colors = []
        self.events = []
        
        # State variables
        self.threshold = threshold
        self.ext_point_n = None
        self.curr_event_max = None
        self.curr_event_min = None
        self.time_point_max = 0
        self.time_point_min = 0
        self.trend_status = 'up'
        self.T = 0
        self.current_index = 0

    def update_point(self, new_price, timestamp=None):
        """
        Process a single new price point and update DC states
        Returns: current event type
        """
        # Initialize if first point
        if self.ext_point_n is None:
            self.ext_point_n = new_price
            self.curr_event_max = new_price
            self.curr_event_min = new_price
            
        # Store price and time
        self.prices.append(new_price)
        if timestamp is None:
            timestamp = self.current_index
        self.time.append(timestamp)
        
        # Calculate TMV
        TMV = (new_price - self.ext_point_n) / (self.ext_point_n * self.threshold)
        self.TMV_list.append(TMV)
        self.T_list.append(self.T)
        self.T += 1

        current_event = None
        
        if self.trend_status == 'up':
            self.colors.append('lime')
            self.events.append('Upward Overshoot')
            current_event = 'Upward Overshoot'

            if new_price < ((1 - self.threshold) * self.curr_event_max):
                self.trend_status = 'down'
                self.curr_event_min = new_price
                self.ext_point_n = self.curr_event_max
                self.T = self.current_index - self.time_point_max

                # Update previous events in current trend
                num_points_change = self.current_index - self.time_point_max
                for j in range(1, num_points_change + 1):
                    self.colors[-j] = 'red'
                    self.events[-j] = 'Downward DCC'
                current_event = 'Downward DCC'
            else:
                if new_price > self.curr_event_max:
                    self.curr_event_max = new_price
                    self.time_point_max = self.current_index
        else:
            self.colors.append('lightcoral')
            self.events.append('Downward Overshoot')
            current_event = 'Downward Overshoot'

            if new_price > ((1 + self.threshold) * self.curr_event_min):
                self.trend_status = 'up'
                self.curr_event_max = new_price
                self.ext_point_n = self.curr_event_min            
                self.T = self.current_index - self.time_point_min

                # Update previous events in current trend
                num_points_change = self.current_index - self.time_point_min
                for j in range(1, num_points_change + 1):
                    self.colors[-j] = 'green'
                    self.events[-j] = 'Upward DCC'
                current_event = 'Upward DCC'
            else:
                if new_price < self.curr_event_min:
                    self.curr_event_min = new_price
                    self.time_point_min = self.current_index

        self.current_index += 1
        return current_event

    def compute_dc_variables(self, prices=None):
        """
        Batch computation of DC variables for a series of prices
        """
        if prices is not None:
            # Reset state for new computation
            self.__init__(self.threshold)
            self.prices = prices
            
        if not self.prices:
            print('Please load the time series data first')
            return
            
        for price in self.prices:
            self.update_point(price)
            
        self.colors = np.array(self.colors)
        print('DC variables computation has finished.')

    def get_current_state(self):
        """
        Return current state information
        """
        return {
            'trend_status': self.trend_status,
            'current_event': self.events[-1] if self.events else None,
            'TMV': self.TMV_list[-1] if self.TMV_list else None,
            'T': self.T,
            'current_price': self.prices[-1] if self.prices else None,
            'ext_point': self.ext_point_n
        }

    def reset(self):
        """
        Reset the calculator state
        """
        self.__init__(self.threshold)

def load_data_by_year(file_path, year=2006):
    """
    Load and filter data for a specific year
    
    Parameters:
    file_path: str, path to the CSV file
    year: int, year to filter (default: 2006)
    
    Returns:
    DataFrame containing only the specified year's data
    """
    # Load the data
    data = pd.read_csv(file_path, header=None)
    data.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close"]
    
    # Convert Date to datetime
    data["Date"] = pd.to_datetime(data["Date"])
    
    # Filter for specified year
    data = data[data["Date"].dt.year == year]
    
    # Sort by date to ensure chronological order
    data = data.sort_values('Date').reset_index(drop=True)
    
    return data

def add_volatility_features(df):
    # # Calculate Parkinson Volatility
    # df['parkinson_vol'] = np.sqrt(1/(4 * np.log(2))) * np.log(df['High']/df['Low'])**2
    
    # Calculate log returns
    df['log_return'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # Calculate log return change (delta of log returns)
    df['log_return_change'] = df['log_return'] - df['log_return'].shift(1)
    
    # # Handle NaN values created by shift operations
    df['log_return'] = df['log_return'].fillna(0)
    df['log_return_change'] = df['log_return_change'].fillna(0)

    df['parkinson_vol'] = df['log_return_change'].fillna(0)
    
    return df

def plot_time_series(df):
    # Set style and figure size
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Price Series
    ax1.plot(df['Date'], df['Adj Close'], label='Adjusted Close', color='blue')
    ax1.set_title('Stock Price Over Time')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Volume
    ax2.bar(df['Date'], df['Volume'], color='gray', alpha=0.5)
    ax2.set_title('Trading Volume')
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    
    # Plot 3: Volatility
    ax3.plot(df['Date'], df['parkinson_vol'], label='Parkinson Volatility', color='red')
    ax3.set_title('Parkinson Volatility')
    ax3.set_ylabel('Volatility')
    ax3.legend()
    ax3.grid(True)
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_hmm_results(df, hidden_states, state_probs):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
    
    # Plot 1: Original Price and States
    ax1.plot(df['Date'], df['Adj Close'], label='Adjusted Close', color='blue', alpha=0.5)
    ax1.set_title('Stock Price with HMM States')
    ax1.set_ylabel('Price')
    
    # Color the background based on states
    for i in range(HMM_N_COMPONENTS):
        mask = (hidden_states == i)
        ax1.fill_between(df['Date'], df['Adj Close'].min(), df['Adj Close'].max(),
                        where=mask, alpha=0.3, label=f'State {i}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Volatility and States
    ax2.plot(df['Date'], df['parkinson_vol'], label='Parkinson Volatility', 
            color='red', alpha=0.5)
    ax2.set_title('Volatility with HMM States')
    ax2.set_ylabel('Volatility')
    
    # Color the background based on states
    for i in range(HMM_N_COMPONENTS):
        mask = (hidden_states == i)
        ax2.fill_between(df['Date'], df['parkinson_vol'].min(), df['parkinson_vol'].max(),
                        where=mask, alpha=0.3, label=f'State {i}')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: State Probabilities
    ax3.plot(df['Date'], state_probs[:, 0], label='State 0 Prob', color='green')
    ax3.plot(df['Date'], state_probs[:, 1], label='State 1 Prob', color='orange')
    ax3.set_title('HMM State Probabilities')
    ax3.set_ylabel('Probability')
    ax3.legend()
    ax3.grid(True)
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_dc_events(data):
    # Extract prices and events
    prices = [item[0] for item in data]
    events = [item[1] for item in data]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot all prices with blue line
    plt.plot(prices, color='blue', alpha=0.5, label='Price')
    
    # Find and highlight DC events
    for i in range(len(events)):
        if events[i] == 'Upward DCC':
            # Highlight the first point of Upward DCC
            plt.scatter(i, prices[i], color='green', s=100, label='Upward DCC' if i == events.index('Upward DCC') else '')
        elif events[i] == 'Downward DCC':
            # Highlight the first point of Downward DCC
            plt.scatter(i, prices[i], color='red', s=100, label='Downward DCC' if i == events.index('Downward DCC') else '')
    
    # Customize plot
    plt.title('Price Movement with DC Events')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    plt.savefig("meme.png")

for i in range(1997, 2007):
    df = load_data_by_year(FILE_PATH, i)
    df = add_volatility_features(df)

    calc = online_dc_calculator()
    res = []
    for j in range(len(df['Date'])):
        res.append([df['Adj Close'].iloc[j], calc.update_point(df['Adj Close'].iloc[j], df['Date'].iloc[j])])
    plot_dc_events(res)
    

    # model.fit(pd.DataFrame(df['parkinson_vol']))
    # hidden_states_probs = model.predict_proba(pd.DataFrame(df['parkinson_vol']))
    # hidden_states = np.argmax(hidden_states_probs, axis=1)

    # fig = plot_hmm_results(df, hidden_states, hidden_states_probs)
    # plt.savefig(f"{OUTPUT_FILE}_{i}.png")
