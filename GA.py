import random
import math
import os
import csv
from typing import List, Optional

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
        Get the fitness of the chromosome.

        Returns:
            int: The fitness value.
        """
        if self.fitness == 0:
            self.fitness = FitnessCalc.get_fitness_calc(self)
        return self.fitness

    def __str__(self) -> str:
        return " ".join(map(str, self.genes))

    def to_print(self) -> str:
        return f"{self} : {self.fitness}"

class FitnessCalc:
    """Class for calculating fitness."""

    max_fitness: int = 0
    avg_fitness: float = 0
    fitness_total: int = 0

    @staticmethod
    def get_fitness_calc(chromosome: Chromosome) -> int:
        """
        Calculate the fitness of a chromosome.

        Args:
            chromosome (Chromosome): The chromosome to evaluate.

        Returns:
            int: The fitness value.
        """
        fitness = FitnessCalcScenario.calculate_fitness(chromosome)
        FitnessCalc.fitness_total += fitness
        if FitnessCalc.max_fitness < fitness:
            FitnessCalc.max_fitness = fitness
        return fitness

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

    # @staticmethod
    # def main() -> None:
        # """Main method to run the Genetic Algorithm."""
        # output_filename = GA.get_unique_filename("resources2/GATableListTraining.txt")
        
        # with open(output_filename, "w") as output_file:
        #     while GA.counter < 50:
        #         GA.counter += 1
        #         my_pop = Population(GA.population_count, True)

        #         generation_count = 0
        #         my_pop.print_population()

        #         builder: List[str] = []

        #         while my_pop.get_fittest().get_fitness() * 0.7 > FitnessCalc.get_avg_fitness():
        #             generation_count += 1
        #             fittest = my_pop.get_fittest()
        #             print(f"GA Generation: {generation_count} Fittest: {fittest.get_fitness()} Fittest Chromosome: {fittest}")
        #             print(f"FitnessCalc.get_avg_fitness(): {FitnessCalc.get_avg_fitness()}")
        #             my_pop = Algorithm.evolve_population(my_pop)
        #             # my_pop.print_population()

        #         print("Solution found!")
        #         print(f"GA Generation: {generation_count}")
        #         print("Genes:")
        #         my_pop.get_fittest().print_chromosome()
        #         print("--------------------------------")
        #         FitnessCalc.set_fitness_total(0)

        #         builder.append(my_pop.get_fittest().string_builder())
        #         builder.append(GA.hold_gene_generator())
        #         builder.append(GA.hold_gene_generator(True))

        #         # Write the current iteration's output to the file
        #         output_file.writelines(builder)
        #         output_file.flush()  # Ensure the data is written to the file

        # print(f"Output written to: {output_filename}")
        

# class PDGA(GA):
#     """Class for running the Primal-Dual Genetic Algorithm."""

#     def __init__(self, population_size: int = 300, crossover_rate: float = 0.7, mutation_rate: float = 0.001):
#         super().__init__()
#         self.population_size = population_size
#         self.crossover_rate = crossover_rate
#         self.mutation_rate = mutation_rate

#     @staticmethod
#     def primal_dual_mapping(chromosome: Chromosome) -> Chromosome:
#         """
#         Create a dual chromosome by flipping all bits of the primal chromosome.

#         Args:
#             chromosome (Chromosome): The primal chromosome.

#         Returns:
#             Chromosome: The dual chromosome.
#         """
#         dual = Chromosome()
#         for i in range(chromosome.size()):
#             dual.set_gene(i, 1 - chromosome.get_gene(i))
#         return dual

#     def select_for_dual_evaluation(self, population: Population, num_select: int) -> List[Chromosome]:
#         """
#         Select chromosomes for dual evaluation based on their fitness.

#         Args:
#             population (Population): The current population.
#             num_select (int): Number of chromosomes to select.

#         Returns:
#             List[Chromosome]: Selected chromosomes for dual evaluation.
#         """
#         sorted_chromosomes = sorted(population.chromosomes, key=lambda c: c.get_fitness())
#         return sorted_chromosomes[:num_select]

#     def adaptive_dual_selection_size(self, generation: int, population_size: int) -> int:
#         """
#         Adaptively determine the number of chromosomes for dual evaluation.

#         Args:
#             generation (int): Current generation number.
#             population_size (int): Size of the population.

#         Returns:
#             int: Number of chromosomes to select for dual evaluation.
#         """
#         return max(1, int(population_size * math.exp(-0.05 * generation)))

#     def evolve_population(self, pop: Population) -> Population:
#         """
#         Evolve a population using the PDGA approach.

#         Args:
#             pop (Population): The population to evolve.

#         Returns:
#             Population: The evolved population.
#         """
#         # Call evolve_population from Algorithm, not GA
#         new_population = Algorithm.evolve_population(pop)
        
#         # Perform dual evaluation
#         num_dual_eval = self.adaptive_dual_selection_size(self.counter, pop.size())
#         selected_for_dual = self.select_for_dual_evaluation(new_population, num_dual_eval)
        
#         for chromosome in selected_for_dual:
#             dual_chromosome = self.primal_dual_mapping(chromosome)
#             if dual_chromosome.get_fitness() > chromosome.get_fitness():
#                 # Replace the primal chromosome with its superior dual
#                 index = new_population.chromosomes.index(chromosome)
#                 new_population.save_chromosome(index, dual_chromosome)

#         return new_population

#     @staticmethod
#     def main() -> None:
#         """Main method to run the Primal-Dual Genetic Algorithm."""
#         pdga = PDGA()
#         output_filename = PDGA.get_unique_filename("resources2/GATableListTraining.txt")
        
#         with open(output_filename, "w") as output_file:
#             while pdga.counter < 50:
#                 pdga.counter += 1
#                 my_pop = Population(pdga.population_size, True)

#                 generation_count = 0
#                 my_pop.print_population()

#                 builder: List[str] = []

#                 while my_pop.get_fittest().get_fitness()*0.7 > FitnessCalc.get_avg_fitness():
#                     generation_count += 1
#                     fittest = my_pop.get_fittest()
#                     print(f"Generation: {generation_count} Fittest: {fittest.get_fitness()} Fittest Chromosome: {fittest}")
#                     print(f"FitnessCalc.get_avg_fitness(): {FitnessCalc.get_avg_fitness()}")
#                     my_pop = pdga.evolve_population(my_pop)

#                 print("Solution found!")
#                 print(f"Generation: {generation_count}")
#                 print("Genes:")
#                 my_pop.get_fittest().print_chromosome()
#                 print("--------------------------------")
#                 FitnessCalc.set_fitness_total(0)

#                 builder.append(my_pop.get_fittest().string_builder())
#                 builder.append(PDGA.hold_gene_generator())
#                 builder.append(PDGA.hold_gene_generator(True))

#                 output_file.writelines(builder)
#                 output_file.flush()  # Ensure the data is written to the file

#         print(f"Output written to: {output_filename}")

class PDGA(GA):
    def __init__(self, population_size: int = 300):
        super().__init__()
        self.population_size = population_size
        # Modified parameters
        self.tournament_size = 3  # Reduced from 5
        self.mutation_rate = 0.02  # Increased from 0.001
        self.crossover_rate = 0.8  # Added
        
        # Dual evaluation parameterss
        self.n_valid_duals = 0
        self.n_total_duals = 0
        self.threshold = 0.7  # Reduced from 0.9
        self.decrease_rate = 0.8  # Changed from 0.9
        self.increase_rate = 1.2  # Changed from 1.1
        self.n_select = self.population_size // 4  # Changed from population_size // 2
        self.n_min = max(1, self.population_size // 10)
        self.n_max = self.population_size // 2

    @staticmethod
    def primal_dual_mapping(chromosome: Chromosome) -> Chromosome:
        """Modified dual mapping to increase diversity"""
        dual = Chromosome()
        for i in range(chromosome.size()):
            if i in [0, 4]:  # RSI buy values (5-40)
                value = chromosome.get_gene(i)
                dual_value = max(5, min(40, 45 - value + random.randint(-2, 2)))
                dual.set_gene(i, dual_value)
            elif i in [2, 6]:  # RSI sell values (60-95)
                value = chromosome.get_gene(i)
                dual_value = max(60, min(95, 155 - value + random.randint(-2, 2)))
                dual.set_gene(i, dual_value)
            elif i in [1, 3, 5, 7]:  # RSI intervals (5-20)
                value = chromosome.get_gene(i)
                dual_value = max(5, min(20, 25 - value + random.randint(-1, 1)))
                dual.set_gene(i, dual_value)
        return dual

    def evolve_population(self, pop: Population) -> Population:
        """
        Evolve population using PDGA approach with error handling.
        
        Args:
            pop (Population): The population to evolve.
        
        Returns:
            Population: The evolved population.
        """
        try:
            # Regular GA evolution
            new_population = Algorithm.evolve_population(pop)
            
            # Reset dual evaluation counters
            self.n_valid_duals = 0
            self.n_total_duals = 0
            
            # Get number of chromosomes for dual evaluation
            num_dual_eval = self.adaptive_dual_selection_size()
            
            # Select worst performing chromosomes
            valid_chromosomes = [c for c in new_population.chromosomes if c is not None]
            valid_chromosomes.sort(key=lambda x: x.get_fitness())
            chromosomes_for_dual = valid_chromosomes[:num_dual_eval]
            
            # Evaluate duals
            for chromosome in chromosomes_for_dual:
                try:
                    self.n_total_duals += 1
                    dual = self.primal_dual_mapping(chromosome)
                    dual_fitness = dual.get_fitness()
                    
                    if dual_fitness > chromosome.get_fitness():
                        self.n_valid_duals += 1
                        # Replace inferior chromosome with superior dual
                        idx = new_population.chromosomes.index(chromosome)
                        new_population.save_chromosome(idx, dual)
                except Exception as e:
                    print(f"Error evaluating dual chromosome: {str(e)}")
                    continue
            
            return new_population
            
        except Exception as e:
            print(f"Error in evolve_population: {str(e)}")
            return pop  # Return original population if evolution fails

    def adaptive_dual_selection_size(self) -> int:
        """
        Adaptively determine number of chromosomes for dual evaluation.
        
        Returns:
            int: Number of chromosomes to select for dual evaluation.
        """
        if self.n_total_duals == 0:
            return self.n_max
        
        valid_ratio = self.n_valid_duals / self.n_total_duals
        if valid_ratio < self.threshold:
            self.n_select = max(self.n_min, 
                              int(self.n_select * self.decrease_rate))
        elif valid_ratio > self.threshold:
            self.n_select = min(self.n_max, 
                              int(self.n_select * self.increase_rate))
        
        return self.n_select

    @staticmethod
    def main() -> None:
        """Main method to run the Primal-Dual Genetic Algorithm."""
        pdga = PDGA()
        output_filename = PDGA.get_unique_filename("resources2/GATableListTraining.txt")
        
        with open(output_filename, "w") as output_file:
            while pdga.counter < 50:
                pdga.counter += 1
                my_pop = Population(pdga.population_size, True)

                generation_count = 0
                min_generations = 1  # Minimum generations to run
                max_generations = 6  # Maximum generations allowed
                
                builder: List[str] = []

                while generation_count < max_generations:
                    generation_count += 1
                    fittest = my_pop.get_fittest()
                    current_avg_fitness = FitnessCalc.get_avg_fitness()
                    
                    # Calculate metrics
                    valid_ratio = pdga.n_valid_duals / pdga.n_total_duals if pdga.n_total_duals > 0 else 0
                    convergence = current_avg_fitness / fittest.get_fitness() if fittest.get_fitness() > 0 else 0
                    
                    print(f"Generation: {generation_count} Fittest: {fittest.get_fitness()} "
                        f"Fittest Chromosome: {fittest}")
                    print(f"Valid Duals Ratio: {valid_ratio:.2f}")
                    print(f"Convergence: {convergence:.2f}")
                    print(f"Average Fitness: {current_avg_fitness}")

                    # Check if we've reached optimal conditions
                    if generation_count >= min_generations and \
                    0.25 <= valid_ratio and \
                    0.6 <= convergence <= 0.9:
                        print("Optimal conditions reached!")
                        break
                    
                    my_pop = pdga.evolve_population(my_pop)

                print("Solution found!")
                print(f"Generation: {generation_count}")
                print("Genes:")
                my_pop.get_fittest().print_chromosome()
                print("--------------------------------")
                FitnessCalc.set_fitness_total(0)

                builder.append(my_pop.get_fittest().string_builder())
                builder.append(PDGA.hold_gene_generator())
                builder.append(PDGA.hold_gene_generator(True))

                output_file.writelines(builder)
                output_file.flush()

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
        Handle the up trend scenario.

        Args:
            data (List[List[str]]): The data to process.
            k (int): The current index.
            chromosome (Chromosome): The chromosome being evaluated.

        Returns:
            int: The updated index.
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
        Handle the down trend scenario.

        Args:
            data (List[List[str]]): The data to process.
            k (int): The current index.
            chromosome (Chromosome): The chromosome being evaluated.

        Returns:
            int: The updated index.
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
        Handle the sell scenario.

        Args:
            k (int): The buy index.
            j (int): The sell index.
        """
        FitnessCalcScenario.gain = FitnessCalcScenario.sell_point - FitnessCalcScenario.buy_point
        FitnessCalcScenario.money_temp = (FitnessCalcScenario.share_number * FitnessCalcScenario.sell_point) - 1.0
        FitnessCalcScenario.money = FitnessCalcScenario.money_temp
        FitnessCalcScenario.maximum_money = max(FitnessCalcScenario.maximum_money, FitnessCalcScenario.money)
        FitnessCalcScenario.minimum_money = min(FitnessCalcScenario.minimum_money, FitnessCalcScenario.money)
        FitnessCalcScenario.transaction_count += 1
        FitnessCalcScenario.total_percent_profit += (FitnessCalcScenario.gain / FitnessCalcScenario.buy_point)
        FitnessCalcScenario.total_gain += FitnessCalcScenario.gain


import multiprocessing
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple

class EnhancedPDGA(PDGA):
    def __init__(self, population_size: int = 300, crossover_rate: float = 0.7, mutation_rate: float = 0.001,
                 selection_method: str = 'tournament', adaptive_rates: bool = False):
        super().__init__(population_size, crossover_rate, mutation_rate)
        self.selection_method = selection_method
        self.adaptive_rates = adaptive_rates
        self.best_fitnesses: List[float] = []
        self.avg_fitnesses: List[float] = []

    def select_parent(self, population: Population) -> Chromosome:
        if self.selection_method == 'tournament':
            return Algorithm.tournament_selection(population)
        elif self.selection_method == 'roulette':
            return self.roulette_wheel_selection(population)
        # Add more selection methods as needed

    def roulette_wheel_selection(self, population: Population) -> Chromosome:
        total_fitness = sum(c.get_fitness() for c in population.chromosomes)
        pick = random.uniform(0, total_fitness)
        current = 0
        for chromosome in population.chromosomes:
            current += chromosome.get_fitness()
            if current > pick:
                return chromosome
        return population.chromosomes[-1]  # Fallback to last chromosome

    def adapt_rates(self) -> None:
        if self.adaptive_rates:
            # Example: Decrease mutation rate as algorithm progresses
            self.mutation_rate = max(0.0001, self.mutation_rate * 0.99)

    def evolve_population(self, pop: Population) -> Population:
        new_population = super().evolve_population(pop)
        self.adapt_rates()
        return new_population

    def calculate_fitness_parallel(self, chromosomes: List[Chromosome]) -> List[Tuple[Chromosome, int]]:
        with multiprocessing.Pool() as pool:
            results = pool.map(FitnessCalcScenario.calculate_fitness, chromosomes)
        return list(zip(chromosomes, results))

    def run(self, max_generations: int = 1000, early_stop: int = 50) -> Chromosome:
        population = Population(self.population_size, True)
        stagnant_generations = 0
        best_fitness = float('-inf')

        for generation in range(max_generations):
            # Parallel fitness calculation
            fitness_results = self.calculate_fitness_parallel(population.chromosomes)
            for chromosome, fitness in fitness_results:
                chromosome.fitness = fitness

            fittest = population.get_fittest()
            current_best_fitness = fittest.get_fitness()
            avg_fitness = sum(c.get_fitness() for c in population.chromosomes) / len(population.chromosomes)

            self.best_fitnesses.append(current_best_fitness)
            self.avg_fitnesses.append(avg_fitness)

            print(f"Generation {generation}: Best Fitness = {current_best_fitness}, Avg Fitness = {avg_fitness}")

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if stagnant_generations >= early_stop:
                print(f"Early stopping at generation {generation}")
                break

            population = self.evolve_population(population)

        return population.get_fittest()

    def plot_progress(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitnesses, label='Best Fitness')
        plt.plot(self.avg_fitnesses, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('PDGA Progress')
        plt.legend()
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Run Enhanced Primal-Dual Genetic Algorithm')
    parser.add_argument('--population', type=int, default=300, help='Population size')
    parser.add_argument('--generations', type=int, default=1000, help='Maximum number of generations')
    parser.add_argument('--selection', choices=['tournament', 'roulette'], default='tournament', help='Selection method')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive rates')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    pdga = EnhancedPDGA(population_size=args.population, selection_method=args.selection, adaptive_rates=args.adaptive)
    best_solution = pdga.run(max_generations=args.generations)
    print(f"Best solution fitness: {best_solution.get_fitness()}")
    pdga.plot_progress()

# if __name__ == "__main__":
#     GA.main()

# if __name__ == "__main__":
#     pdga = PDGA()
#     pdga.main()
