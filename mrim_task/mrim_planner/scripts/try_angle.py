import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the circular arc path
radius = 50  # Radius of the arc
arc_length = 100  # Total length of the arc
theta_i = 10  # Initial yaw angle in degrees
theta_f = 90  # Final yaw angle in degrees
d = 10  # Incremental distance along the arc

# Calculate the number of increments along the arc
n = int(arc_length / d)

# Calculate the total yaw change and yaw change per increment
delta_theta = theta_f - theta_i
delta_theta_inc = delta_theta / n

# Generate the positions along the arc and corresponding yaw angles
angles = np.linspace(0, arc_length / radius, n + 1)
x = radius * np.sin(angles)
y = radius * (1 - np.cos(angles))
yaw_angles = [theta_i + k * delta_theta_inc for k in range(n + 1)]

# Convert yaw angles to radians for plotting vectors
yaw_angles_rad = np.deg2rad(yaw_angles)

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
