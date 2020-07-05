import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def hist(data, bins, margin=1, label='Distribution'):
    bins = sorted(bins)
    length = len(bins)
    intervals = np.zeros(length+1)
    for value in data:
        i = 0
        while i < length and value >= bins[i]:
            i += 1
        intervals[i] += 1
    intervals = intervals / float(len(data))
    plt.xlim(min(bins) - margin, max(bins) + margin)
    bins.insert(0, -999)
    plt.title("probability-distribution")
    plt.xlabel('Interval')
    plt.ylabel('Probability')
    plt.bar(bins, intervals, color=['b'], label=label)
    plt.legend()
    plt.show()