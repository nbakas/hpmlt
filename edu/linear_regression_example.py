from import_libraries import *
num_gpus = round(10 + 20 * rand(100), 0)
scaling_algorithm = 5 + 2 * num_gpus + 10 * (rand(100) - 1/2)
Xtr = c_[ones(len(num_gpus)), num_gpus]
a = inv(Xtr.T @ Xtr) @ Xtr.T @ scaling_algorithm
pred = Xtr @ a
plt.scatter(num_gpus, scaling_algorithm)
plt.scatter(num_gpus, pred, color='red')
plt.show()