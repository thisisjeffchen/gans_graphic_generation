import numpy as np
import pandas as pd
import collections
import matplotlib.pyplot as plt

def plot(cats, title,  filename, max):
    keys = list(cats.keys())
    values = list(cats.values())

    fig = plt.figure(num=None, figsize=(8, 20))
    plt.barh(keys, values)

    for i, v in enumerate(values):
        plt.text(max*0.9, i-0.25, str(v))

    plt.title(title)
    plt.legend()
    plt.xlabel('Value')
    plt.xlim(0, max)
    plt.ylabel('Category')
    plt.ylim(-1, len(cats.keys()))
    fig.savefig(filename, bbox_inches='tight')

def read_file(filename):
    cats = {}
    with open(filename) as file:
        for line in file:
            cat, _, num = line.strip('\n').rpartition(' ')
            cats[cat] = int(num)
    return collections.OrderedDict(sorted(cats.items()))



def main():
    train_cat = read_file('train_cat.txt')
    plot(train_cat, "Training Categories", "traincat.png", 15000)

    train_cat = read_file('val_cat.txt')
    plot(train_cat, "Validation Categories", "valcat.png", 700)

if __name__ == "__main__":
    # execute only if run as a script
    main()
