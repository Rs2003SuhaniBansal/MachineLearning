import numpy as np

mean = 0
sigma = 15

x = np.linspace(-100, 100, 100)

pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * \
      np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

for xi, yi in zip(x, pdf):
    print(f"x = {xi:.2f}, pdf = {yi:.6f}")