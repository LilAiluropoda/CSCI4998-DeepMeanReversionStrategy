import random
import os
import csv
from typing import List, Optional
from multiprocessing import Pool, cpu_count
import numpy as np
from functools import lru_cache
from app.utils.path_config import PathConfig
from pathlib import Path


class Algorithm:
    """Class containing genetic algorithm methods."""

    UNIFORM_RATE: float = 0.7
    MUTATION_RATE: float = 0.001
    TOURNAMENT_SIZE: int = 5
    ELITISM: bool = False

    @staticmethod
    def evolve_population(pop: "Population") -> "Population":
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
    def crossover(indiv1: "Chromosome", indiv2: "Chromosome") -> "Chromosome":
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
    def mutate(indiv: "Chromosome") -> None:
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
    def tournament_selection(pop: "Population") -> "Chromosome":
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
    """Optimized class for calculating fitness."""

    max_fitness: int = 0
    avg_fitness: float = 0
    fitness_total: int = 0
    _cached_data: Optional[np.ndarray] = None

    @staticmethod
    @lru_cache(maxsize=1)
    def load_data(fname: str) -> np.ndarray:
        """Cache and load CSV data for repeated use."""
        if FitnessCalc._cached_data is None:
            try:
                FitnessCalc._cached_data = np.genfromtxt(
                    fname, delimiter=";", dtype=float
                )
            except Exception as e:
                print(f"Error loading data: {e}")
                return np.array([])
        return FitnessCalc._cached_data

    @staticmethod
    def parallel_fitness_calc(
        chromosomes: List["Chromosome"], chunk_size: int = 50
    ) -> List[int]:
        """Calculate fitness for multiple chromosomes in parallel."""
        with Pool(processes=cpu_count() - 1) as pool:
            fitnesses = pool.map(
                FitnessCalcScenario.calculate_fitness, chromosomes, chunk_size
            )

            # Update fitness statistics
            FitnessCalc.fitness_total = sum(fitnesses)
            FitnessCalc.max_fitness = max(FitnessCalc.max_fitness, max(fitnesses))

            return fitnesses

    @staticmethod
    def get_fitness_calc(chromosome: "Chromosome") -> int:
        """Calculate fitness for a single chromosome."""
        fitness = FitnessCalcScenario.calculate_fitness(chromosome)
        FitnessCalc.fitness_total += fitness
        FitnessCalc.max_fitness = max(FitnessCalc.max_fitness, fitness)
        return fitness

    @staticmethod
    def get_avg_fitness() -> float:
        """Calculate average fitness."""
        FitnessCalc.avg_fitness = FitnessCalc.fitness_total / GA.population_count
        return FitnessCalc.avg_fitness

    @staticmethod
    def set_fitness_total(set_value: int) -> None:
        """Set the total fitness value."""
        FitnessCalc.fitness_total = set_value

    @staticmethod
    def reset_fitness_stats():
        """Reset all fitness statistics."""
        FitnessCalc.max_fitness = 0
        FitnessCalc.avg_fitness = 0
        FitnessCalc.fitness_total = 0

    @staticmethod
    def clear_cache():
        """Clear the cached data."""
        FitnessCalc._cached_data = None
        FitnessCalc.load_data.cache_clear()


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
            if (
                chromosome
                and fittest
                and fittest.get_fitness() <= chromosome.get_fitness()
            ):
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
        """Generate a hold gene."""
        value = random.randint(40, 60)
        interval = random.randint(2, 20)
        trend = "1.0" if up_trend else "0.0"
        return f"0 1:{value}.0 2:{interval}.0 3:{trend}\n"

    @staticmethod
    def get_unique_filename(base_filename: Path) -> Path:
        """Get a unique filename by appending a number if the file already exists."""
        if not base_filename.exists():
            return base_filename

        index = 1
        while True:
            new_filename = (
                base_filename.parent
                / f"{base_filename.stem}({index}){base_filename.suffix}"
            )
            if not new_filename.exists():
                return new_filename
            index += 1

    @staticmethod
    def main() -> None:
        """Main method to run the Genetic Algorithm."""
        output_filename = GA.get_unique_filename(PathConfig.GA_TRAINING_LIST)
        GA.counter = 0  # reset GA counter

        with open(output_filename, "w") as output_file:
            while GA.counter < 5:
                GA.counter += 1
                my_pop = Population(GA.population_count, True)

                generation_count = 0
                # my_pop.print_population()

                builder: List[str] = []

                while (
                    my_pop.get_fittest().get_fitness() * 0.7
                    > FitnessCalc.get_avg_fitness()
                ):
                    generation_count += 1
                    fittest = my_pop.get_fittest()
                    # print(f"Generation: {generation_count} Fittest: {fittest.get_fitness()} Fittest Chromosome: {fittest}")
                    # print(f"FitnessCalc.get_avg_fitness(): {FitnessCalc.get_avg_fitness()}")
                    my_pop = Algorithm.evolve_population(my_pop)

                print("Solution found!")
                print(f"Generation: {generation_count}")
                print("Genes:")
                my_pop.get_fittest().print_chromosome()
                print("--------------------------------")
                FitnessCalc.set_fitness_total(0)

                builder.append(my_pop.get_fittest().string_builder())
                builder.append(GA.hold_gene_generator())
                builder.append(GA.hold_gene_generator(True))

                output_file.writelines(builder)
                output_file.flush()

        print(f"Output written to: {output_filename}")


class FitnessCalcScenario:
    """Optimized scenario calculator using numpy operations."""

    @staticmethod
    def calculate_fitness(chromosome: "Chromosome") -> int:
        """Optimized fitness calculation using numpy."""
        # Load and cache data
        data = FitnessCalc.load_data(str(PathConfig.OUTPUT_CSV))
        if data.size == 0:
            return 0

        # Rest of the method remains the same
        money = 10000.0
        k = 0

        while k < len(data) - 1:
            sma50 = data[k, 21]
            sma200 = data[k, 22]
            trend = sma50 - sma200

            if trend > 0:  # upTrend
                k, money = FitnessCalcScenario.handle_trend(
                    data, k, chromosome, money, is_uptrend=True
                )
            else:  # downTrend
                k, money = FitnessCalcScenario.handle_trend(
                    data, k, chromosome, money, is_uptrend=False
                )
            k += 1

        return int(money)

    @staticmethod
    def handle_trend(
        data: np.ndarray,
        k: int,
        chromosome: "Chromosome",
        money: float,
        is_uptrend: bool,
    ) -> tuple[int, float]:
        """Unified trend handler with optimized calculations."""
        offset = 4 if is_uptrend else 0

        # Check buy condition
        if data[k, chromosome.get_gene(offset + 1)] <= chromosome.get_gene(offset):
            buy_point = data[k, 0] * 100
            share_number = (money - 1.0) / buy_point
            force_sell = False

            # Process potential sell points
            for j in range(k, len(data) - 1):
                sell_point = data[j, 0] * 100
                money_temp = (share_number * sell_point) - 1.0

                # Check stop loss
                if money * 0.85 > money_temp:
                    money = money_temp
                    force_sell = True

                # Check sell condition
                if (
                    data[j, chromosome.get_gene(offset + 3)]
                    >= chromosome.get_gene(offset + 2)
                    or force_sell
                ):
                    gain = sell_point - buy_point
                    money = money_temp
                    k = j
                    break

        return k, money

    @staticmethod
    def reset_scenario() -> None:
        """No longer needed as we're not using class variables."""
        pass


if __name__ == "__main__":
    GA.main()
