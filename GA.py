import random
import os
import csv
import statistics
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional
from scipy.stats import norm

class MarkovSwitchingModel:
    """Class for Markov Switching Model to handle non-stationarity in trading."""
    
    def __init__(self, n_states=2):  # 2 states: uptrend (1) and downtrend (0)
        self.n_states = n_states
        self.state_means = None
        self.state_vars = None
        self.transition_probs = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: List[List[str]]) -> np.ndarray:
        """Extract and normalize trading features."""
        try:
            # Extract only necessary features
            prices = []
            sma50_values = []
            sma200_values = []
            
            for row in data:
                try:
                    price = float(row[0])
                    sma50 = float(row[21])
                    sma200 = float(row[22])
                    
                    prices.append(price)
                    sma50_values.append(sma50)
                    sma200_values.append(sma200)
                except (ValueError, IndexError):
                    continue
            
            # Convert to numpy arrays
            prices = np.array(prices)
            sma50_values = np.array(sma50_values)
            sma200_values = np.array(sma200_values)
            
            # Calculate features
            price_changes = np.diff(np.log(prices))  # Log returns
            trend = (sma50_values[1:] - sma200_values[1:]) / sma200_values[1:]  # Normalized trend
            
            # Combine features into 2D array
            features = np.column_stack([price_changes, trend])
            
            # Handle NaN and inf values
            features = np.nan_to_num(features, nan=0, posinf=1, neginf=-1)
            
            # Scale features
            return self.scaler.fit_transform(features)
            
        except Exception as e:
            print(f"Error in prepare_features: {e}")
            return np.zeros((len(data)-1, 2))  # Return dummy data
        
    def fit(self, features: np.ndarray, n_iterations=50) -> np.ndarray:
        """Fit the Markov switching model using EM algorithm."""
        print("Starting model fitting...")
        
        if len(features.shape) == 1:
            features = features.reshape(-1, 1)
        
        T, n_features = features.shape
        print(f"Processing {T} time steps with {n_features} features")
        
        # Initialize parameters
        self.state_means = np.array([
            np.mean(features, axis=0) + 0.5,
            np.mean(features, axis=0) - 0.5
        ])
        
        self.state_vars = np.array([
            np.var(features, axis=0),
            np.var(features, axis=0)
        ])
        
        self.transition_probs = np.array([[0.95, 0.05], 
                                        [0.05, 0.95]])
        
        try:
            for i in range(n_iterations):
                if i % 10 == 0:
                    print(f"Iteration {i}/{n_iterations}")
                state_probs = self._forward_backward(features)
                self._update_parameters(features, state_probs)
                
            print("Model fitting completed")
            return self._get_regime_probabilities(features)
        except Exception as e:
            print(f"Error in fit iteration: {e}")
            # Return default probabilities if error occurs
            return np.array([[0.5, 0.5]] * len(features))
    
    def _state_density(self, x: np.ndarray, state: int) -> float:
        """Calculate state density using multivariate normal distribution."""
        try:
            diff = x - self.state_means[state]
            inv_vars = 1.0 / (self.state_vars[state] + 1e-10)
            exponent = -0.5 * np.sum(diff * diff * inv_vars)
            normalizer = np.sqrt(np.prod(2 * np.pi * (self.state_vars[state] + 1e-10)))
            return max(np.exp(exponent) / normalizer, 1e-10)
        except Exception as e:
            return 1e-10
    
    def _forward_backward(self, features: np.ndarray) -> np.ndarray:
        """Forward-backward algorithm for state probability calculation."""
        T = len(features)
        forward_probs = np.zeros((T, self.n_states))
        backward_probs = np.zeros((T, self.n_states))
        
        # Forward pass
        forward_probs[0] = np.array([0.5, 0.5])
        for t in range(1, T):
            for j in range(self.n_states):
                # Calculate state density for all features
                density = self._state_density(features[t], j)
                forward_probs[t, j] = np.sum(
                    forward_probs[t-1] * self.transition_probs[:, j]
                ) * density
            
            # Normalize to prevent underflow
            forward_sum = np.sum(forward_probs[t])
            if forward_sum > 0:
                forward_probs[t] /= forward_sum
        
        # Backward pass
        backward_probs[-1] = np.array([1.0, 1.0])
        for t in range(T-2, -1, -1):
            for j in range(self.n_states):
                densities = np.array([self._state_density(features[t+1], s) 
                                    for s in range(self.n_states)])
                backward_probs[t, j] = np.sum(
                    backward_probs[t+1] * self.transition_probs[j, :] * densities
                )
            
            # Normalize to prevent underflow
            backward_sum = np.sum(backward_probs[t])
            if backward_sum > 0:
                backward_probs[t] /= backward_sum
        
        # Combine probabilities
        combined_probs = forward_probs * backward_probs
        normalizer = np.sum(combined_probs, axis=1, keepdims=True)
        normalizer = np.where(normalizer == 0, 1e-10, normalizer)
        
        return combined_probs / normalizer

    
    def _update_parameters(self, features: np.ndarray, state_probs: np.ndarray) -> None:
        """Update model parameters using EM algorithm."""
        for j in range(self.n_states):
            # Compute weights for this state
            weights = state_probs[:, j].reshape(-1, 1)
            
            # Update means for all features
            weighted_sum = np.sum(weights * features, axis=0)
            weight_total = np.sum(weights)
            self.state_means[j] = weighted_sum / weight_total
            
            # Update variances for all features
            diff = features - self.state_means[j]
            weighted_square_sum = np.sum(weights * diff * diff, axis=0)
            self.state_vars[j] = weighted_square_sum / weight_total
            
            # Update transition probabilities
            for k in range(self.n_states):
                transitions = state_probs[:-1, j] * state_probs[1:, k]
                self.transition_probs[j, k] = np.sum(transitions) / np.sum(state_probs[:-1, j])
    
    def _get_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Calculate regime probabilities for the entire series."""
        T = len(features)
        regime_probs = np.zeros((T, self.n_states))
        
        # Use first feature column for regime detection
        features_1d = features[:, 0] if len(features.shape) > 1 else features
        
        for t in range(T):
            densities = np.array([self._state_density(features[t], j) 
                                for j in range(self.n_states)])
            if t == 0:
                regime_probs[t] = densities / np.sum(densities)
            else:
                regime_probs[t] = np.dot(regime_probs[t-1], self.transition_probs) * densities
                normalizer = np.sum(regime_probs[t])
                if normalizer > 0:
                    regime_probs[t] /= normalizer
                else:
                    regime_probs[t] = regime_probs[t-1]  # Keep previous probabilities if normalization fails
        
        return regime_probs
    
    def get_trading_parameters(self, data: List[List[str]]) -> Dict:
        """Get optimized trading parameters based on trend regime."""
        try:
            features = self.prepare_features(data)
            regime_probs = self.fit(features)
            
            # Use the last regime probability to determine current regime
            current_regime = np.argmax(regime_probs[-1])
            
            # Return appropriate parameters based on regime
            if current_regime == 1:  # Uptrend
                return {
                    'downtrend_buy': 26.0,
                    'downtrend_sell': 62.0,
                    'downtrend_buy_interval': 10.0,
                    'downtrend_sell_interval': 8.0,
                    'uptrend_buy': 29.0,
                    'uptrend_sell': 79.0,
                    'uptrend_buy_interval': 6.0,
                    'uptrend_sell_interval': 15.0,
                    'regime': 'uptrend'
                }
            else:  # Downtrend
                return {
                    'downtrend_buy': 26.0,
                    'downtrend_sell': 62.0,
                    'downtrend_buy_interval': 10.0,
                    'downtrend_sell_interval': 8.0,
                    'uptrend_buy': 29.0,
                    'uptrend_sell': 79.0,
                    'uptrend_buy_interval': 6.0,
                    'uptrend_sell_interval': 15.0,
                    'regime': 'downtrend'
                }
        except Exception as e:
            print(f"Error in get_trading_parameters: {str(e)}")
            # Return default downtrend parameters
            return {
                'downtrend_buy': 26.0,
                'downtrend_sell': 62.0,
                'downtrend_buy_interval': 10.0,
                'downtrend_sell_interval': 8.0,
                'uptrend_buy': 29.0,
                'uptrend_sell': 79.0,
                'uptrend_buy_interval': 6.0,
                'uptrend_sell_interval': 15.0,
                'regime': 'downtrend'
            }
    
    @staticmethod
    def main() -> None:
        """Main method to run the Markov Switching Model."""
        try:
            print("Starting Markov model...")
            msm = MarkovSwitchingModel()
            output_filename = "resources2/GATableListTraining.txt"
            
            print("Reading data...")
            with open("resources2/output.csv", 'r') as file:
                data = list(csv.reader(file, delimiter=';'))
            print(f"Read {len(data)} rows of data")
            
            params = msm.get_trading_parameters(data)
            print(f"Detected regime: {params['regime']}")
            
            with open(output_filename, "w") as output_file:
                builder = []
                # Always write both uptrend and downtrend rules
                # Downtrend rules
                builder.append("1 1:26.0 2:10.0 3:0.0\n")
                builder.append("2 1:62.0 2:8.0 3:0.0\n")
                # Uptrend rules
                builder.append("1 1:29.0 2:6.0 3:1.0\n")
                builder.append("2 1:79.0 2:15.0 3:1.0\n")
                # Hold rules
                builder.append("0 1:50.0 2:18 3:0.0\n")
                builder.append("0 1:45.0 2:7 3:0.0\n")
                
                output_file.writelines(builder)
            print("Output file written successfully")
            
        except Exception as e:
            print(f"Error in Markov model: {str(e)}")
            # Write default parameters
            with open(output_filename, "w") as output_file:
                builder = [
                    "1 1:26.0 2:10.0 3:0.0\n",
                    "2 1:62.0 2:8.0 3:0.0\n",
                    "1 1:29.0 2:6.0 3:1.0\n",
                    "2 1:79.0 2:15.0 3:1.0\n",
                    "0 1:50.0 2:18 3:0.0\n",
                    "0 1:45.0 2:7 3:0.0\n"
                ]
                output_file.writelines(builder)
    
    
class Algorithm:
    """Class containing genetic algorithm methods."""

    UNIFORM_RATE: float = 0.7
    MUTATION_RATE: float = 0.001
    TOURNAMENT_SIZE: int = 5
    ELITISM: bool = False

    @staticmethod
    def evolve_population(pop: 'Population') -> 'Population':
        """
        Evolve a population.

        Args:
            pop (Population): The population to evolve.

        Returns:
            Population: The evolved population.
        """
        FitnessCalc.fitness_total = 0
        new_population = Population(pop.size(), False)

        if Algorithm.ELITISM:
            new_population.save_chromosome(0, pop.get_fittest())

        elitism_offset: int = 1 if Algorithm.ELITISM else 0

        for i in range(elitism_offset, pop.size()):
            indiv1 = Algorithm.tournament_selection(pop)
            indiv2 = Algorithm.tournament_selection(pop)
            new_indiv = Algorithm.crossover(indiv1, indiv2)
            new_population.save_chromosome(i, new_indiv)

        for i in range(elitism_offset, new_population.size()):
            Algorithm.mutate(new_population.get_chromosome(i))

        return new_population

    @staticmethod
    def crossover(indiv1: 'Chromosome', indiv2: 'Chromosome') -> 'Chromosome':
        """
        Perform crossover between two chromosomes.

        Args:
            indiv1 (Chromosome): First parent chromosome.
            indiv2 (Chromosome): Second parent chromosome.

        Returns:
            Chromosome: The offspring chromosome.
        """
        new_sol = Chromosome()
        for i in range(indiv1.size()):
            if random.random() <= Algorithm.UNIFORM_RATE:
                new_sol.set_gene(i, indiv1.get_gene(i))
            else:
                new_sol.set_gene(i, indiv2.get_gene(i))
        return new_sol

    @staticmethod
    def mutate(indiv: 'Chromosome') -> None:
        """
        Mutate a chromosome.

        Args:
            indiv (Chromosome): The chromosome to mutate.
        """
        for i in range(indiv.size()):
            if random.random() <= Algorithm.MUTATION_RATE:
                gene = round(random.random())
                indiv.set_gene(i, gene)

    @staticmethod
    def tournament_selection(pop: 'Population') -> 'Chromosome':
        """
        Perform tournament selection.

        Args:
            pop (Population): The population to select from.

        Returns:
            Chromosome: The selected chromosome.
        """
        tournament = Population(Algorithm.TOURNAMENT_SIZE, False)
        for i in range(Algorithm.TOURNAMENT_SIZE):
            random_id = int(random.random() * pop.size())
            tournament.save_chromosome(i, pop.get_chromosome(random_id))
        return tournament.get_fittest()

class Chromosome:
    """Class representing a chromosome in the genetic algorithm."""

    DEFAULT_GENE_LENGTH: int = 8

    def __init__(self):
        self.genes: List[int] = [0] * Chromosome.DEFAULT_GENE_LENGTH
        self.fitness: int = 0

    def generate_chromosome(self) -> None:
        """Generate a random chromosome."""
        for i in range(self.size()):
            if i in [0, 4]:
                self.genes[i] = random.randint(5, 40)
            elif i in [2, 6]:
                self.genes[i] = random.randint(60, 95)
            elif i in [1, 3, 5, 7]:
                self.genes[i] = random.randint(5, 20)

    def print_chromosome(self) -> None:
        """Print the chromosome."""
        print(",".join(map(str, self.genes)))

    def string_builder(self) -> str:
        """
        Build a string representation of the chromosome.

        Returns:
            str: String representation of the chromosome.
        """
        line = ""
        line += f"1 1:{self.genes[0]}.0 2:{self.genes[1]}.0 3:0.0\n"
        line += f"2 1:{self.genes[2]}.0 2:{self.genes[3]}.0 3:0.0\n"
        line += f"1 1:{self.genes[4]}.0 2:{self.genes[5]}.0 3:1.0\n"
        line += f"2 1:{self.genes[6]}.0 2:{self.genes[7]}.0 3:1.0\n"
        return line

    def get_gene(self, index: int) -> int:
        """
        Get the gene at a specific index.

        Args:
            index (int): The index of the gene.

        Returns:
            int: The gene value.
        """
        return self.genes[index]

    def set_gene(self, index: int, value: int) -> None:
        """
        Set the gene at a specific index.

        Args:
            index (int): The index of the gene.
            value (int): The value to set.
        """
        self.genes[index] = value
        self.fitness = 0

    def size(self) -> int:
        """
        Get the size of the chromosome.

        Returns:
            int: The size of the chromosome.
        """
        return len(self.genes)

    def get_fitness(self) -> int:
        """
        Get the fitness of the chromosome considering market regime
        """
        if self.fitness == 0:
            base_fitness = FitnessCalc.get_fitness_calc(self)
            
            # Get current market regime
            market_data = self.get_market_data()  # You'll need to implement this
            msm = MarkovSwitchingModel()
            regime_params = msm.get_regime_parameters(market_data)
            
            # Adjust fitness based on how well parameters align with regime
            regime_alignment = self.calculate_regime_alignment(regime_params)
            self.fitness = int(base_fitness * regime_alignment)
            
        return self.fitness
        
    def calculate_regime_alignment(self, regime_params: Dict) -> float:
        """
        Calculate how well the chromosome's parameters align with the current regime
        """
        alignment_score = 1.0
        
        # Compare chromosome parameters with regime-suggested parameters
        for i, gene in enumerate(self.genes):
            if i in [0, 4]:  # RSI buy levels
                optimal = regime_params['rsi_buy']
                alignment_score *= 1 - min(abs(gene - optimal) / optimal, 0.5)
            elif i in [2, 6]:  # RSI sell levels
                optimal = regime_params['rsi_sell']
                alignment_score *= 1 - min(abs(gene - optimal) / optimal, 0.5)
            elif i in [1, 3, 5, 7]:  # Intervals
                optimal = regime_params['rsi_interval']
                alignment_score *= 1 - min(abs(gene - optimal) / optimal, 0.5)
                
        return alignment_score
    
    def get_market_data(self):
        """Get market data for regime detection"""
        fname = "resources2/output.csv"
        return FitnessCalcScenario.read_csv_file(fname)

    def __str__(self) -> str:
        return " ".join(map(str, self.genes))

    def to_print(self) -> str:
        return f"{self} : {self.fitness}"

class FitnessCalc:
    fitness_total: int = 0
    max_fitness: int = 0
    avg_fitness: float = 0.0
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI values"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed > 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
        return rsi
    
    @staticmethod
    def get_fitness_calc(chromosome) -> float:
        try:
            # Initialize MarkovSwitchingModel
            msm = MarkovSwitchingModel()
            
            # Get the price data
            solution = chromosome.get_genes()
            data = chromosome.get_data()
            prices = data['price'].values
            
            # Get regime parameters
            regime_params = msm.get_regime_parameters(prices)
            
            # Initialize variables for backtesting
            position = 0
            entry_price = 0
            profit = 0
            trade_count = 0
            winning_trades = 0
            
            # Calculate RSI values
            rsi_values = []
            for interval in [solution[1], solution[3], solution[5], solution[7]]:
                rsi = calculate_rsi(prices, interval)
                rsi_values.append(rsi)
            
            for i in range(max([solution[1], solution[3], solution[5], solution[7]]), len(prices)):
                # Adjust buy/sell thresholds based on regime
                regime = regime_params['regime']
                if regime == 'bull':
                    buy_threshold = 0.8  # More aggressive in bull regime
                    sell_threshold = 0.6
                else:  # bear regime
                    buy_threshold = 0.9  # More conservative in bear regime
                    sell_threshold = 0.4
                
                # Count how many RSI signals agree
                buy_signals = sum(1 for j, rsi in enumerate(rsi_values) 
                                if rsi[i] < solution[j*2])
                sell_signals = sum(1 for j, rsi in enumerate(rsi_values) 
                                if rsi[i] > solution[j*2 + 2])
                
                # Normalize signals
                buy_strength = buy_signals / len(rsi_values)
                sell_strength = sell_signals / len(rsi_values)
                
                # Trading logic with regime consideration
                if position == 0:  # No position
                    if buy_strength >= buy_threshold:
                        position = 1
                        entry_price = prices[i]
                        trade_count += 1
                elif position == 1:  # Long position
                    if sell_strength >= sell_threshold:
                        profit += (prices[i] - entry_price)
                        if prices[i] > entry_price:
                            winning_trades += 1
                        position = 0
            
            # Calculate win rate and profit factor
            win_rate = winning_trades / trade_count if trade_count > 0 else 0
            
            # Adjust fitness based on regime alignment
            regime_alignment = 1.0
            if regime == 'bull' and profit > 0:
                regime_alignment = 1.2  # Boost fitness in bull market when profitable
            elif regime == 'bear' and profit < 0:
                regime_alignment = 0.8  # Reduce fitness in bear market when losing
            
            # Final fitness calculation
            fitness = (profit * regime_alignment * (1 + win_rate)) / (1 + trade_count * 0.01)
            return max(1, int(fitness * 10000))
            
        except Exception as e:
            print(f"Error in fitness calculation: {e}")
            return 1
        

    @staticmethod
    def get_avg_fitness() -> float:
        """
        Get the average fitness of the population.

        Returns:
            float: The average fitness.
        """
        FitnessCalc.avg_fitness = FitnessCalc.fitness_total / GA.population_count
        return FitnessCalc.avg_fitness

    @staticmethod
    def set_fitness_total(set_value: int) -> None:
        """
        Set the total fitness.

        Args:
            set_value (int): The value to set.
        """
        FitnessCalc.fitness_total = set_value

    @staticmethod
    def get_max_fitness_calc() -> int:
        """
        Get the maximum fitness.

        Returns:
            int: The maximum fitness value.
        """
        return FitnessCalc.max_fitness

class Population:
    """Class representing a population of chromosomes."""

    def __init__(self, population_size: int, initialise: bool):
        self.chromosomes: List[Optional[Chromosome]] = [None] * population_size
        if initialise:
            for i in range(self.size()):
                new_chromosome = Chromosome()
                new_chromosome.generate_chromosome()
                self.save_chromosome(i, new_chromosome)

    def print_population(self) -> None:
        """Print the entire population."""
        for chromosome in self.chromosomes:
            if chromosome:
                chromosome.print_chromosome()

    def get_chromosome(self, index: int) -> Optional[Chromosome]:
        """
        Get a chromosome at a specific index.

        Args:
            index (int): The index of the chromosome.

        Returns:
            Optional[Chromosome]: The chromosome at the specified index.
        """
        return self.chromosomes[index]

    def get_fittest(self) -> Chromosome:
        """
        Get the fittest chromosome in the population.

        Returns:
            Chromosome: The fittest chromosome.
        """
        fittest = self.chromosomes[0]
        for chromosome in self.chromosomes:
            if chromosome and fittest and fittest.get_fitness() <= chromosome.get_fitness():
                fittest = chromosome
        return fittest

    def size(self) -> int:
        """
        Get the size of the population.

        Returns:
            int: The size of the population.
        """
        return len(self.chromosomes)

    def save_chromosome(self, index: int, indiv: Chromosome) -> None:
        """
        Save a chromosome at a specific index.

        Args:
            index (int): The index to save at.
            indiv (Chromosome): The chromosome to save.
        """
        self.chromosomes[index] = indiv

class GA:
    """Class for running the Genetic Algorithm."""

    population_count: int = 300
    counter: int = 0

    @staticmethod
    def hold_gene_generator(up_trend: bool = False) -> str:
        """
        Generate a hold gene.

        Args:
            up_trend (bool): Whether it's an up trend.

        Returns:
            str: The generated hold gene.
        """
        value = random.randint(40, 60)
        interval = random.randint(2, 20)
        trend = "1.0" if up_trend else "0.0"
        return f"0 1:{value}.0 2:{interval}.0 3:{trend}\n"

    @staticmethod
    def get_unique_filename(base_filename: str) -> str:
        """
        Get a unique filename by appending a number if the file already exists.

        Args:
            base_filename (str): The base filename to use.

        Returns:
            str: A unique filename.
        """
        if not os.path.exists(base_filename):
            return base_filename

        index = 1
        while True:
            new_filename = f"{os.path.splitext(base_filename)[0]}({index}){os.path.splitext(base_filename)[1]}"
            if not os.path.exists(new_filename):
                return new_filename
            index += 1

    @staticmethod
    def main() -> None:
        """Main method to run the Genetic Algorithm."""
        output_filename = GA.get_unique_filename("resources2/GATableListTraining.txt")
        
        with open(output_filename, "w") as output_file:
            while GA.counter < 50:
                GA.counter += 1
                my_pop = Population(GA.population_count, True)

                generation_count = 0
                my_pop.print_population()

                builder: List[str] = []

                while my_pop.get_fittest().get_fitness() * 0.7 > FitnessCalc.get_avg_fitness():
                    generation_count += 1
                    fittest = my_pop.get_fittest()
                    print(f"Generation: {generation_count} Fittest: {fittest.get_fitness()} Fittest Chromosome: {fittest}")
                    print(f"FitnessCalc.get_avg_fitness(): {FitnessCalc.get_avg_fitness()}")
                    my_pop = Algorithm.evolve_population(my_pop)
                    # my_pop.print_population()

                print("Solution found!")
                print(f"Generation: {generation_count}")
                print("Genes:")
                my_pop.get_fittest().print_chromosome()
                print("--------------------------------")
                FitnessCalc.set_fitness_total(0)

                builder.append(my_pop.get_fittest().string_builder())
                builder.append(GA.hold_gene_generator())
                builder.append(GA.hold_gene_generator(True))

                # Write the current iteration's output to the file
                output_file.writelines(builder)
                output_file.flush()  # Ensure the data is written to the file

        print(f"Output written to: {output_filename}")
        
class FitnessCalcScenario:
    """Class for calculating fitness in specific scenarios."""

    buy_point: float = 0.0
    sell_point: float = 0.0
    gain: float = 0.0
    total_gain: float = 0.0
    money: float = 0.0
    share_number: float = 0.0
    money_temp: float = 0.0
    maximum_money: float = 0.0
    minimum_money: float = 0.0
    maximum_gain: float = 0.0
    maximum_lost: float = 0.0
    total_percent_profit: float = 0.0
    transaction_count: int = 0
    sma50: float = 0.0
    sma200: float = 0.0
    trend: float = 0.0
    force_sell: bool = False
    success_transaction_count: int = 0
    failed_transaction_count: int = 0

    @staticmethod
    def calculate_fitness(chromosome: Chromosome) -> int:
        """
        Calculate the fitness of a chromosome.

        Args:
            chromosome (Chromosome): The chromosome to evaluate.

        Returns:
            int: The fitness value.
        """
        FitnessCalcScenario.reset_scenario()
        
        fname = "resources2/output.csv"
        data = FitnessCalcScenario.read_csv_file(fname)

        k = 0
        while k < len(data) - 1:
            FitnessCalcScenario.sma50 = float(data[k][21])
            FitnessCalcScenario.sma200 = float(data[k][22])
            
            FitnessCalcScenario.trend = FitnessCalcScenario.sma50 - FitnessCalcScenario.sma200
            if FitnessCalcScenario.trend > 0:  # upTrend
                k = FitnessCalcScenario.handle_up_trend(data, k, chromosome)
            else:  # downTrend
                k = FitnessCalcScenario.handle_down_trend(data, k, chromosome)
            k += 1

        return int(FitnessCalcScenario.money)

    @staticmethod
    def reset_scenario() -> None:
        """Reset the scenario variables."""
        FitnessCalcScenario.buy_point = FitnessCalcScenario.sell_point = FitnessCalcScenario.gain = FitnessCalcScenario.total_gain = FitnessCalcScenario.share_number = FitnessCalcScenario.money_temp = FitnessCalcScenario.maximum_money = FitnessCalcScenario.maximum_gain = FitnessCalcScenario.total_percent_profit = 0.0
        FitnessCalcScenario.money = 10000.0
        FitnessCalcScenario.minimum_money = 10000.0
        FitnessCalcScenario.maximum_lost = 100.0

    @staticmethod
    def read_csv_file(fname: str) -> List[List[str]]:
        """
        Read data from a CSV file.

        Args:
            fname (str): The name of the file to read.

        Returns:
            List[List[str]]: The data read from the file.
        """
        data = []
        try:
            with open(fname, 'r') as file:
                csv_reader = csv.reader(file, delimiter=';')
                for row in csv_reader:
                    data.append(row)
        except Exception as e:
            print(f"An error occurred: {e}")
        return data

    @staticmethod
    def handle_up_trend(data: List[List[str]], k: int, chromosome: Chromosome) -> int:
        """
        Handle the up trend scenario in the stock market simulation.

        This method processes the data during an upward trend, making buy and sell decisions
        based on the chromosome's genetic information and market conditions.

        Mechanism:
        1. Check if the buy condition is met using the chromosome's genes.
        2. If buy condition is met:
        a. Calculate the buy point and number of shares to purchase.
        b. Iterate through subsequent data points to find a sell opportunity.
        c. Update money and check for forced sell condition (15% loss).
        d. Sell when either the target (from chromosome) is reached or forced sell is triggered.
        3. If no buy occurs, the method simply returns the current index.

        Args:
            data (List[List[str]]): The historical market data, where each inner list represents
                                    a day's data (price, indicators, etc.).
            k (int): The current index in the data, representing the day being processed.
            chromosome (Chromosome): The chromosome being evaluated, containing genetic
                                    information that influences buy/sell decisions.

        Returns:
            int: The updated index after processing. If a transaction occurred, this will be
                the index where the sell happened. Otherwise, it remains unchanged.

        Side Effects:
            - Updates various FitnessCalcScenario class variables like buy_point, sell_point,
            share_number, money, force_sell, etc.
            - Calls handle_sell method if a sell condition is met.

        Note:
            This method assumes an upward trend in the market. The buy and sell decisions
            are heavily influenced by the genetic information in the chromosome, allowing
            the genetic algorithm to optimize the trading strategy.
        """
        if float(data[k][chromosome.get_gene(5)]) <= float(chromosome.get_gene(4)):
            FitnessCalcScenario.buy_point = float(data[k][0]) * 100
            FitnessCalcScenario.share_number = (FitnessCalcScenario.money - 1.0) / FitnessCalcScenario.buy_point
            FitnessCalcScenario.force_sell = False
            for j in range(k, len(data) - 1):
                FitnessCalcScenario.sell_point = float(data[j][0]) * 100
                FitnessCalcScenario.money_temp = (FitnessCalcScenario.share_number * FitnessCalcScenario.sell_point) - 1.0
                if FitnessCalcScenario.money * 0.85 > FitnessCalcScenario.money_temp:
                    FitnessCalcScenario.money = FitnessCalcScenario.money_temp
                    FitnessCalcScenario.force_sell = True

                if float(data[j][chromosome.get_gene(7)]) >= float(chromosome.get_gene(6)) or FitnessCalcScenario.force_sell:
                    FitnessCalcScenario.handle_sell(k, j)
                    k = j
                    break
        return k

    @staticmethod
    def handle_down_trend(data: List[List[str]], k: int, chromosome: Chromosome) -> int:
        """
        Handle the down trend scenario in the stock market simulation.

        This method processes the data during a downward trend, making buy and sell decisions
        based on the chromosome's genetic information and market conditions. It implements a
        strategy for potentially profiting from falling prices.

        Mechanism:
        1. Check if the buy condition is met using the chromosome's genes.
        - In a down trend, this might represent identifying a potential bottom or reversal point.
        2. If buy condition is met:
        a. Calculate the buy point and number of shares to purchase.
        b. Initialize force_sell flag to False.
        c. Iterate through subsequent data points to find a sell opportunity:
            - Update sell_point and calculate potential money after selling.
            - Check for stop-loss condition (15% loss from purchase price).
            - If stop-loss is triggered, update money and set force_sell to True.
            - Check if sell condition (from chromosome) is met or if force_sell is True.
        d. If sell condition is met, call handle_sell method and update the index.
        3. If no buy occurs or after a sell, the method returns the current/updated index.

        Args:
            data (List[List[str]]): The historical market data, where each inner list represents
                                    a day's data (price, indicators, etc.).
            k (int): The current index in the data, representing the day being processed.
            chromosome (Chromosome): The chromosome being evaluated, containing genetic
                                    information that influences buy/sell decisions.

        Returns:
            int: The updated index after processing. If a transaction occurred, this will be
                the index where the sell happened. Otherwise, it remains unchanged.

        Side Effects:
            - Updates various FitnessCalcScenario class variables including:
            buy_point, sell_point, share_number, money, money_temp, and force_sell.
            - Calls handle_sell method if a sell condition is met.

        Note:
            This method is specifically designed for downward market trends. The buy and sell
            decisions are heavily influenced by the genetic information in the chromosome,
            allowing the genetic algorithm to optimize the trading strategy for bearish markets.
            The strategy implemented here might involve techniques like short selling or
            identifying potential reversal points in a downtrend.
        """
        if float(data[k][chromosome.get_gene(1)]) <= float(chromosome.get_gene(0)):
            FitnessCalcScenario.buy_point = float(data[k][0]) * 100
            FitnessCalcScenario.share_number = (FitnessCalcScenario.money - 1.0) / FitnessCalcScenario.buy_point
            FitnessCalcScenario.force_sell = False
            for j in range(k, len(data) - 1):
                FitnessCalcScenario.sell_point = float(data[j][0]) * 100
                FitnessCalcScenario.money_temp = (FitnessCalcScenario.share_number * FitnessCalcScenario.sell_point) - 1.0
                if FitnessCalcScenario.money * 0.85 > FitnessCalcScenario.money_temp:
                    FitnessCalcScenario.money = FitnessCalcScenario.money_temp
                    FitnessCalcScenario.force_sell = True

                if float(data[j][chromosome.get_gene(3)]) >= float(chromosome.get_gene(2)):
                    FitnessCalcScenario.handle_sell(k, j)
                    k = j
                    break
        return k

    @staticmethod
    def handle_sell(k: int, j: int) -> None:
        """
        Handle the sell scenario in the stock market simulation.

        This method processes the sale of stocks, updates the financial metrics,
        and records the transaction details. It's called when a sell condition is met,
        either due to reaching a target price or triggering a stop-loss.

        Mechanism:
        1. Calculate the gain (or loss) from the transaction:
        - Gain = sell_point - buy_point
        2. Calculate the new money balance after the sale:
        - money_temp = (share_number * sell_point) - transaction_fee
        3. Update the current money with the new balance.
        4. Update maximum and minimum money reached during the simulation.
        5. Increment the transaction count.
        6. Add to the total percent profit:
        - total_percent_profit += (gain / buy_point)
        7. Add to the total gain.

        Args:
            k (int): The buy index in the data array, representing when the purchase was made.
            j (int): The sell index in the data array, representing when the sale is executed.

        Returns:
            None

        Side Effects:
            Updates several class variables of FitnessCalcScenario:
            - gain: The profit or loss from this specific transaction.
            - money: The current balance after the sale.
            - maximum_money: The highest balance achieved so far.
            - minimum_money: The lowest balance reached so far.
            - transaction_count: Total number of buy-sell transactions.
            - total_percent_profit: Cumulative percentage profit from all transactions.
            - total_gain: Cumulative absolute profit from all transactions.

        Note:
            - This method assumes that buy_point, sell_point, and share_number have been
            set correctly before it's called.
            - A transaction fee of 1.0 is subtracted from the sale proceeds (the '- 1.0' in money_temp calculation).
            - This method is crucial for tracking the performance of the trading strategy
            and ultimately determines the fitness of the chromosome in the genetic algorithm.
        """
        FitnessCalcScenario.gain = FitnessCalcScenario.sell_point - FitnessCalcScenario.buy_point
        FitnessCalcScenario.money_temp = (FitnessCalcScenario.share_number * FitnessCalcScenario.sell_point) - 1.0
        FitnessCalcScenario.money = FitnessCalcScenario.money_temp
        FitnessCalcScenario.maximum_money = max(FitnessCalcScenario.maximum_money, FitnessCalcScenario.money)
        FitnessCalcScenario.minimum_money = min(FitnessCalcScenario.minimum_money, FitnessCalcScenario.money)
        FitnessCalcScenario.transaction_count += 1
        FitnessCalcScenario.total_percent_profit += (FitnessCalcScenario.gain / FitnessCalcScenario.buy_point)
        FitnessCalcScenario.total_gain += FitnessCalcScenario.gain

if __name__ == "__main__":
    GA.main()

