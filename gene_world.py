import numpy as np
import gene
from typing import Callable
from joblib import Parallel, delayed


class BinaryWorld:

    """World of 2D-genes composed of binary number."""

    def __init__(self, selector: Callable[[np.ndarray], int], genesize_x: int =5, genesize_y: int =5, worldsize: int =100, mutation_scale: float=0.5):
        self.selector = selector
        self.genesize_x = genesize_x
        self.genesize_y = genesize_y
        self.generation_count = 0
        self.worldsize = worldsize if worldsize % 2 == 0 else worldsize + 1  # worldsize must be even.
        self.world = [0 for i in range(self.worldsize)]
        self.mutation_scale = mutation_scale
        for i in range(self.worldsize):
            self.world[i] = gene.Gene(np.random.randint(0, 2, (genesize_y, genesize_x)))
        return

    def crossbreed(self, gene1: gene.Gene, gene2: gene.Gene) -> (gene.Gene, gene.Gene):
        """Crossbreed given two genes and generate new two genes."""
        subgenesize_x = np.random.randint(0, self.genesize_x + 1)
        subgenesize_y = np.random.randint(0, self.genesize_y + 1)
        takepoint_x = np.random.randint(0, self.genesize_x)
        takepoint_y = np.random.randint(0, self.genesize_y)
        subgene1 = gene1.take_subgene(takepoint_x, takepoint_y, takepoint_x + subgenesize_x, takepoint_y + subgenesize_y)
        subgene2 = gene2.take_subgene(takepoint_x, takepoint_y, takepoint_x + subgenesize_x, takepoint_y + subgenesize_y)
        child1 = gene1.copy()
        child1.insert_subgene(subgene2, takepoint_x, takepoint_y)
        child2 = gene2.copy()
        child2.insert_subgene(subgene1, takepoint_x, takepoint_y)
        self.mutation(child1)
        self.mutation(child2)
        return (child1, child2)

    def get_genes(self) -> [gene.Gene]:
        """Return list of all genes in the world."""
        return self.world

    def mutation(self, mutate_gene: gene.Gene):
        """Insert random small gene into given gene."""
        mutation_size = int(np.random.exponential(self.mutation_scale))
        if mutation_size == 0:
            return
        mutation_gene = gene.Gene(np.random.randint(0, 2, (mutation_size, mutation_size)))
        mutatepoint_x = np.random.randint(0, self.genesize_x)
        mutatepoint_y = np.random.randint(0, self.genesize_y)
        mutate_gene.insert_subgene(mutation_gene, mutatepoint_x, mutatepoint_y)
        return

    def selection(self):
        """Rank genes, select parents and reproduct the new generation."""
        map(self.mutation, self.world)
        score = Parallel(n_jobs=-1)([delayed(self.selector)(g) for g in self.world])
        score = np.array(score)
        score /= score.sum()
        probability_density = np.zeros(self.worldsize)
        probability_density[0] = score[0]
        for i in range(1, len(score)):
            probability_density[i] = probability_density[i - 1] + score[i]
        self._reproduce(probability_density)
        self.generation_count += 1
        return

    def _reproduce(self, probability_density: np.ndarray):
        new_generation = [0 for i in range(self.worldsize)]
        for i in range(self.worldsize // 2):
            parent1 = self.world[np.where(probability_density > np.random.rand())[0][0]]
            parent2 = self.world[np.where(probability_density > np.random.rand())[0][0]]
            child1, child2 = self.crossbreed(parent1, parent2)
            new_generation[i * 2], new_generation[i * 2 + 1] = child1, child2
        self.world = new_generation
        return

    def immigration(self, immigrant: gene.Gene):
        """Replace a random gene by given gene."""
        expelee = np.random.randint(0, self.worldsize)
        self.world[expelee] = immigrant
        return

    def evaluate_world(self):
        score = Parallel(n_jobs=-1)([delayed(self.selector)(g) for g in self.world])
        score = np.array(score)
        return np.average(score), np.std(score)


class MultiBinaryWorld():
    def __init__(self, selector: Callable[[np.ndarray], int], genesize_x: int =5, genesize_y: int =5, island_count: int =5, total_worldsize: int =10000):
        self.island_count = island_count
        self.worldsize = total_worldsize // island_count
        self.islands = [BinaryWorld(selector, genesize_x=genesize_x, genesize_y=genesize_y, worldsize=self.worldsize, mutation_scale=(i + 1) / 2)
                        for i in range(self.island_count)]
        self.generation_count = 0
        self.mean_immigrants = self.worldsize // 100
        self.immigrant_counts = []
        return

    def selection(self):
        import itertools

        self.immigrant_counts = []
        for immigrantee, emmigrantee in itertools.permutations(self.islands, 2):
            immigrant_size = int(np.random.poisson(self.mean_immigrants))
            self.immigrant_counts.append(immigrant_size)
            immigrants = np.random.choice(immigrantee.world, size=immigrant_size, replace=False)
            for immigrant in immigrants:
                emmigrantee.immigration(immigrant)

        ([(lambda w: w.selection())(w) for w in self.islands])
        self.generation_count += 1
        return

    def evaluate_islands(self):
        return ([(lambda w: w.evaluate_world())(w) for w in self.islands])
