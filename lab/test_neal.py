def main():
    import neal
    sampler = neal.SimulatedAnnealingSampler()
    h = {'a': 0.0, 'b': 0.0, 'c': 0.0}
    J = {('a', 'b'): 1.0, ('b', 'c'): 1.0, ('a', 'c'): 1.0}
    sampleset = sampler.sample_ising(h, J, num_reads=1)
    print(sampleset.first.energy)


if __name__ == "__main__":
    main()