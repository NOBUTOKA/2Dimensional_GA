import numpy as np
import gene_world as g
import gene


def selector(gene: gene.Gene) -> int:
    y, x = gene.get_genesize()
    point = 0
    gene_plane = gene.get_genecode()
    for i in range(10):
        sample = np.random.rand(x, y)
        point += np.sum((np.dot(sample, gene_plane) - sample) ** 2)
    return 1 / (point / 10 + 1)


if __name__ == '__main__':
    world = g.BinaryWorld(selector, genesize_x=7, genesize_y=7, worldsize=1000)
    for i in range(1000):
        world.selection()
        ev = world.evaluate_world()
        print('{0}: {1}'.format(i, ev))
        if ev > 0.99:
            break
    gs = world.get_genes()
    for gx in gs:
        print(gx.get_genecode())
