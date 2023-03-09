import numpy as np
import matplotlib.pyplot as plt
boundary = 10
positions = np.random.uniform(0,10, size=(6, 2))
print(positions)

distances = np.linalg.norm(positions[:, None] - positions[:], axis=2)

xx1, xx2 = np.meshgrid(positions[: ,0], positions[: ,0])
d_ij_x = np.abs(xx1- xx2)

yy1, yy2 = np.meshgrid(positions[:, 1], positions[:, 1])
d_ij_y = np.abs(yy1 - yy2)
# print(d_ij_y, "\n")
# d_ij_y = d_ij_y[d_ij_y != 0]
# print(d_ij_y, "\n")
d_ij_y = np.where(d_ij_y > 10/2, 10 - d_ij_y, d_ij_y)

# d_ij_x = d_ij_x[d_ij_x != 0]
d_ij_x = np.where(d_ij_x > 10/2, 10 - d_ij_x, d_ij_x)

D_ij = np.sqrt(np.multiply(d_ij_x, d_ij_x) + np.multiply(d_ij_y,d_ij_y))

print("original")
# print(distances[distances!=0])
print(distances)
print("periodic")

print(D_ij)



color_lis = ["red", "green", "blue", "yellow", "orange", "purple"]
plt.scatter(positions[:,0], positions[:, 1], color=color_lis)
plt.axis([0,10, 0, 10])
plt.show()
