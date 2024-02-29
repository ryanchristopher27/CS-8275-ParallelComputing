# Imports
import numpy as np
import matplotlib.pyplot as plt
import os

class Result:
    def __init__(self, filter_size, stride, block_sie, amdahls_speedup, actual_speedup):
        self.filter_size = filter_size
        self.stride = stride
        self.block_size = block_sie
        self.amdahls_speedup = amdahls_speedup
        self.actual_speedup = actual_speedup

# ['Global', 'Shared', 'Texture']
def main():
    
    all_results = []

    all_results.append(Result(4, 1, 4, {"global": 84.10, "shared": 168.77, "texture": 78.93}, {"global": 3.29, "shared": 3.37, "texture": 3.43}))
    all_results.append(Result(4, 1, 32, {"global": 173.92, "shared": 169.06, "texture": 191.74}, {"global": 1.67, "shared": 1.68, "texture": 1.36}))

    all_results.append(Result(4, 2, 4, {"global": 45.09, "shared": 39.55, "texture": 40.04}, {"global": 2.75, "shared": 3.09, "texture": 2.78}))
    all_results.append(Result(4, 2, 32, {"global": 88.17, "shared": 232.33, "texture": 86.37}, {"global": 1.41, "shared": 1.00, "texture": 1.29}))
    
    all_results.append(Result(4, 4, 4, {"global": 14.41, "shared": 15.88, "texture":11.84}, {"global": 2.71, "shared": 2.46, "texture": 2.87}))
    all_results.append(Result(4, 4, 32, {"global": 9.29, "shared": 9.85, "texture": 8.49}, {"global": 4.06, "shared": 3.94, "texture": 3.91}))

    all_results.append(Result(8, 1, 4, {"global": 86.00, "shared": 90.74, "texture": 84.25}, {"global": 41.34, "shared": 38.98, "texture": 39.44}))
    all_results.append(Result(8, 1, 32, {"global": 166.91, "shared": 197.15, "texture": 169.59}, {"global": 21.12, "shared": 17.06, "texture": 19.55}))

    all_results.append(Result(8, 2, 4, {"global": 33.29, "shared": 44.64, "texture": 42.21}, {"global": 36.80, "shared": 35.42, "texture": 32.03}))
    all_results.append(Result(8, 2, 32, {"global": 116.53, "shared": 128.59, "texture": 110.15}, {"global": 13.08, "shared": 11.79, "texture": 12.35}))

    all_results.append(Result(8, 4, 4, {"global": 22.91, "shared": 24.16, "texture": 21.64}, {"global": 21.61, "shared": 20.52, "texture": 19.47}))
    all_results.append(Result(8, 4, 32, {"global": 10.77, "shared": 12.03, "texture": 9.81}, {"global": 45.14, "shared": 40.17, "texture":43.54 }))

    stride = [1, 2, 4]
    filter_size = [4, 8]
    block_size = [4, 32]


    am_global = []
    am_shared = []
    am_texture = []
    ac_global = []
    ac_shared = []
    ac_texture = []

    for result in all_results:
        if result.filter_size == 8 and result.block_size == 4:
            am_global.append(result.amdahls_speedup["global"])
            am_shared.append(result.amdahls_speedup["shared"])
            am_texture.append(result.amdahls_speedup["texture"])
            ac_global.append(result.actual_speedup["global"])
            ac_shared.append(result.actual_speedup["shared"])
            ac_texture.append(result.actual_speedup["texture"])

    
    bar_width = float(1/8)
    index = np.arange(len(stride))

    plt.bar(index, am_global, width=bar_width, label='Amdahls Global', color='blue')
    plt.bar(index + bar_width, am_shared, width=bar_width, label='Amdahls Shared', color='red')
    plt.bar(index + 2 * bar_width, am_texture, width=bar_width, label='Amdahls Texture', color='green')
    plt.bar(index + 4 * bar_width, ac_global, width=bar_width, label='Actual Global', color='purple')
    plt.bar(index + 5 * bar_width, ac_shared, width=bar_width, label='Actual Shared', color='yellow')
    plt.bar(index + 6 * bar_width, ac_texture, width=bar_width, label='Actual Texture', color='orange')


    plt.xlabel('Stride')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Stride')
    plt.xticks(index + bar_width * 3, stride)
    plt.legend()
    plt.suptitle('Filter Size = 8, Block Size = 4', fontsize=10)

    plt.savefig(os.path.join('plots', 'amdahls_stride_vs_filter_4_and_block_4.png'))

    plt.show()


    bar_width = float(1/4)
    index = np.arange(len(stride))

    plt.bar(index, ac_global, width=bar_width, label='Actual Global', color='purple')
    plt.bar(index + bar_width, ac_shared, width=bar_width, label='Actual Shared', color='yellow')
    plt.bar(index + 2 * bar_width, ac_texture, width=bar_width, label='Actual Texture', color='orange')

    plt.xlabel('Stride')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Stride (Actual Only)')
    plt.xticks(index + bar_width, stride)
    plt.legend()
    plt.suptitle('Filter Size = 4, Block Size = 4', fontsize=10)

    plt.savefig(os.path.join('plots', 'actual_stride_vs_filter_4_and_block_4.png'))

    plt.show()

if __name__ == "__main__":
    main()