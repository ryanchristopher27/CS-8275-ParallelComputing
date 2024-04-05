import matplotlib.pyplot as plt
import numpy as np

# Data for Single Image Augmentation (Single Run)
single_run_methods = ['Kornia', 'CuPy', 'CPU']
single_run_rotation = [69.7888, 81.8527, 21.0574]
single_run_vertical_flip = [6.7930, 0.0453, 0.0033]
single_run_horizontal_flip = [0.5511, 0.0885, 0.0029]
single_run_gaussian_noise = [8.4961, 50.9297, 3.6677]

# Data for Single Image Augmentation (100 Runs)
multiple_runs_methods = ['Kornia', 'CuPy', 'CPU']
multiple_runs_rotation = [3.4432, 2.6751, 11.9200]
multiple_runs_vertical_flip = [0.3721, 0.0357, 0.0015]
multiple_runs_horizontal_flip = [0.3212, 0.0350, 0.0015]
multiple_runs_gaussian_noise = [0.4128, 0.2634, 3.7798]

# Data for CIFAR10 Augmentation (Include Data Preprocessing)
cifar10_methods = ['Kornia', 'CuPy', 'CPU']
cifar10_rotation_avg = [2.5959, 1.3236, 0.5248]
cifar10_vertical_flip_avg = [0.5714, 0.3691, 0.0874]
cifar10_horizontal_flip_avg = [0.5733, 0.3682, 0.0902]
cifar10_gaussian_noise_avg = [0.5604, 0.4652, 0.1546]

# Create a figure with subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot Single Image Augmentation (Single Run)
axs[0].plot(single_run_methods, single_run_rotation, marker='o', label='90 Deg Rotation')
axs[0].plot(single_run_methods, single_run_vertical_flip, marker='o', label='Vertical Flip')
axs[0].plot(single_run_methods, single_run_horizontal_flip, marker='o', label='Horizontal Flip')
axs[0].plot(single_run_methods, single_run_gaussian_noise, marker='o', label='Gaussian Noise')
axs[0].set_title('Single Image Augmentation (Single Run)')
axs[0].set_xlabel('Method')
axs[0].set_ylabel('Time (ms)')
axs[0].legend()

# Plot Single Image Augmentation (100 Runs)
axs[1].plot(multiple_runs_methods, multiple_runs_rotation, marker='o', label='90 Deg Rotation')
axs[1].plot(multiple_runs_methods, multiple_runs_vertical_flip, marker='o', label='Vertical Flip')
axs[1].plot(multiple_runs_methods, multiple_runs_horizontal_flip, marker='o', label='Horizontal Flip')
axs[1].plot(multiple_runs_methods, multiple_runs_gaussian_noise, marker='o', label='Gaussian Noise')
axs[1].set_title('Single Image Augmentation (100 Runs)')
axs[1].set_xlabel('Method')
axs[1].set_ylabel('Time (ms)')
axs[1].legend()

# Plot CIFAR10 Augmentation (Include Data Preprocessing)
axs[2].plot(cifar10_methods, cifar10_rotation_avg, marker='o', label='90 Deg Rotation')
axs[2].plot(cifar10_methods, cifar10_vertical_flip_avg, marker='o', label='Vertical Flip')
axs[2].plot(cifar10_methods, cifar10_horizontal_flip_avg, marker='o', label='Horizontal Flip')
axs[2].plot(cifar10_methods, cifar10_gaussian_noise_avg, marker='o', label='Gaussian Noise')
axs[2].set_title('CIFAR10 Augmentation (Include Data Preprocessing)')
axs[2].set_xlabel('Method')
axs[2].set_ylabel('Average Time (ms)')
axs[2].legend()

plt.tight_layout()
plt.show()