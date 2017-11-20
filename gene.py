import numpy as np


class Gene:

    def __init__(self, gene: np.ndarray):
        self.gene = gene

    def get_genecode(self):
        return self.gene

    def get_genesize(self):
        return np.shape(self.gene)

    def copy(self):
        return Gene(self.gene.copy())

    def take_subgene(self, x1: int, y1: int, x2: int, y2: int):
        ysize, xsize = np.shape(self.gene)
        left = max(0, min(x1, x2))
        right = min(xsize, max(x1, x2))
        top = max(0, min(y1, y2))
        bottom = min(ysize, max(y1, y2))
        subgene = self.gene.copy()
        subgene = subgene[top:bottom, left:right]
        return Gene(subgene)

    def insert_subgene(self, subgene: 'Gene', x: int, y: int):
        ysize, xsize = np.shape(self.gene)
        subysize, subxsize = np.shape(subgene.get_genecode())
        right = min(xsize, x + subxsize)
        bottom = min(ysize, y + subysize)
        self.gene[y:bottom, x:right] = subgene.get_genecode()[:bottom - y, :right - x]
