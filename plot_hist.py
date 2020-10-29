# -*- coding: utf-8 -*-
import os
import fnmatch
import shutil
import numpy as np
import pandas as pd
from Bio import SeqIO
from matplotlib import pyplot as plt
import re


def main():
    n_bins = 100

    pssm_file = 'data/training.fasta'

    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(pssm_file), 'fasta')
    # loop through fasta sequences
    lengths = []
    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, sequence = fasta.id, str(fasta.seq)
        lengths.append(len(sequence))

    plt.hist(lengths, n_bins)
    plt.show()


if __name__ == '__main__':
    main()
