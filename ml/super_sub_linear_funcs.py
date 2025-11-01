import numpy as np
import matplotlib.pyplot as plt

# Define the range of x
# Since the log function is undefined at x=0, we start from 1.
x = np.linspace(1, 10, 100)

# 1. Super Linear function: f(x) = x^2
f_super = x**2

# 2. Linear function: f(x) = x
f_linear = x

# 3. Sub Linear function 1: f(x) = sqrt(x)
f_sub_sqrt = np.sqrt(x)

# 4. Sub Linear function 2: f(x) = log(x) (using natural logarithm ln)
f_sub_log = np.log(x)

# Create the plot
plt.figure(figsize=(9, 7))

# Plot the functions
plt.plot(x, f_super, label="Super Linear: $f(x) = x^2$", color="red")
plt.plot(x, f_linear, label="Linear: $f(x) = x$", color="blue")
plt.plot(x, f_sub_sqrt, label="Sub Linear: $f(x) = \sqrt{x}$", color="green", linestyle="--")
plt.plot(x, f_sub_log, label="Sub Linear: $f(x) = \ln(x)$", color="purple")

# Set the title and labels
plt.title("Super, Linear, Sub-Linear Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output (f(x))")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.ylim(0, 105)

plt.show()
