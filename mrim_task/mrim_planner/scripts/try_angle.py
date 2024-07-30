import numpy as np
import matplotlib.pyplot as plt

points = np.array([[ 7.  , 10.52,  4.04],
       [ 5.01, 10.32,  4.11],
       [ 5.01, 10.32,  4.11],
       [ 3.09, 10.87,  4.07],
       [ 3.09, 10.87,  4.07],
       [ 1.77, 12.24,  3.45],
       [ 1.77, 12.24,  3.45],
       [-7.5 , 17.5 ,  1.  ]])


x = points[:,0]
y = points[:,1]
yaw_angles_rad = points[:,2]
# Plot the circular arc path
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Path', color='blue')

# Plot yaw angle vectors
for i in range(len(x)):
    dx = np.cos(yaw_angles_rad[i])
    dy = np.sin(yaw_angles_rad[i])
    plt.arrow(x[i], y[i], dx, dy, head_width=2, head_length=2, fc='red', ec='red')

plt.title('Yaw Angle Vectors Along Curved Path')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()
