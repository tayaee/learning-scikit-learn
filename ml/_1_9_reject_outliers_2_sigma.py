import numpy as np
from matplotlib import pyplot as plt

avg = 27000
std = 15000
cnt = 10000
incomes = np.random.normal(loc=avg, scale=std, size=cnt)
incomes = np.append(incomes, [1000000000])

plt.hist(incomes, 50)
plt.show()


def reject_outliers(incomes):
    u = np.median(incomes)
    s = np.std(incomes)
    filtered = [e for e in incomes if (u - 2 * s < e < u + 2 * s)]
    return filtered


filtererd = reject_outliers(incomes)
plt.hist(filtererd, 50)
plt.show()
