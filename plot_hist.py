# -*- coding: utf-8 -*-
import os
import fnmatch
import shutil
import numpy as np
import pandas as pd
from Bio import SeqIO
from matplotlib import pyplot as plt
import re
import numpy as np
import math
import matplotlib.patches as mpatches


def main():
    n_bins = 100

    pssm_file = 'data/training.fasta'

    # read the fasta sequences from input file
    fasta_sequences = SeqIO.parse(open(pssm_file), 'fasta')
    # loop through fasta sequences
    lengths = []
    anti = []
    nonanti = []

    less = 0
    more = 0

    for fasta in fasta_sequences:
        # get name and value of each sequence
        name, sequence = fasta.id.split('|'), str(fasta.seq)
        lengths.append(len(sequence))
        if len(sequence) > 400:
            more += 1
        else:
            less += 1
        if int(name[1]) == 1:
            anti.append(len(sequence))
        else:
            nonanti.append(len(sequence))

    print(lengths[np.argmax(lengths)])
    print(lengths[np.argmin(lengths)])

    f = plt.figure()
    # plt.subplot2grid((1, 4), (0, 0), rowspan=1, colspan=3)
    plt.rc('xtick',labelsize=24)
    plt.rc('ytick',labelsize=24)
    N, bins, patches = plt.hist(lengths, bins=np.arange(0, 1463 + 10, 10))
    for i in range(0, 40):
        patches[i].set_facecolor('blue')
    for i in range(40, 147):
        patches[i].set_facecolor('red')
    plt.xlabel('Sample length', fontsize=24)
    plt.ylabel('Number of Sample', fontsize=24)
    blue_patch = mpatches.Patch(color='blue', label='88.53%')
    red_patch = mpatches.Patch(color='red', label='11.47%')
    plt.legend(handles=[blue_patch, red_patch],  prop={'size': 20})
    plt.show()
    f.savefig('hinh 2.png')
    # f.savefig('hinh 2.pdf', dpi=700)


if __name__ == '__main__':
    main()
