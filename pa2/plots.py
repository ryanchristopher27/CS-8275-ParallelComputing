# Imports
import numpy as np
import matplotlib.pyplot as plt
import os


class Result:
    def __init__(self, filter_size, stride, block_size, gpu_times, cpu_time):
        self.filter_size = filter_size
        self.stride = stride
        self.block_size = block_size
        self.gpu_time = gpu_times
        self.cpu_time = cpu_time




def main():
    all_results = []

    all_results.append(Result(4, 1, 4, {"global": 0.0535, "shared": 0.0609, "texture": 0.0575}, 0.1895))
    all_results.append(Result(4, 1, 16, {"global": 0.0991, "shared": 0.1010, "texture": 0.0985}, 0.1895))
    all_results.append(Result(4, 1, 32, {"global": 0.1341, "shared": 0.1325, "texture": 0.1591}, 0.1895))

    all_results.append(Result(4, 2, 4, {"global": 0.0190, "shared": 0.0156, "texture": 0.0178}, 0.0478))
    all_results.append(Result(4, 2, 16, {"global": 0.0383, "shared": 0.0377, "texture": 0.0434}, 0.0478))
    all_results.append(Result(4, 2, 32, {"global": 0.0407, "shared": 0.0326, "texture": 0.0332}, 0.0478))
    
    all_results.append(Result(4, 4, 4, {"global": 0.0037, "shared": 0.0048, "texture": 0.0047}, 0.0119))
    all_results.append(Result(4, 4, 16, {"global": 0.0028, "shared": 0.0029, "texture": 0.0030}, 0.0119))
    all_results.append(Result(4, 4, 32, {"global": 0.0027, "shared": 0.0029, "texture": 0.0027}, 0.0119))

    all_results.append(Result(8, 1, 4, {"global": 0.0535, "shared": 0.0580, "texture": 0.0561}, 2.3087))
    all_results.append(Result(8, 1, 16, {"global": 0.0934, "shared": 0.1022, "texture": 0.1053}, 2.3087))
    all_results.append(Result(8, 1, 32, {"global": 0.1255, "shared": 0.1390, "texture": 0.1316}, 2.3087))

    all_results.append(Result(8, 2, 4, {"global": 0.0169, "shared": 0.0168, "texture": 0.0165}, 0.5895))
    all_results.append(Result(8, 2, 16, {"global": 0.0247, "shared": 0.0385, "texture": 0.0364}, 0.5895))
    all_results.append(Result(8, 2, 32, {"global": 0.0372, "shared": 0.0422, "texture": 0.0390}, 0.5895))

    all_results.append(Result(8, 4, 4, {"global": 0.0062, "shared": 0.0075, "texture": 0.0074}, 0.1505))
    all_results.append(Result(8, 4, 16, {"global": 0.0031, "shared": 0.0032, "texture": 0.0033}, 0.1505))
    all_results.append(Result(8, 4, 32, {"global": 0.0033, "shared": 0.0033, "texture": 0.0036}, 0.1505))


    # plt.plot(stride, global_mem, label='Global_4')
    # plt.plot(stride, shared_mem, label='Shared_4')
    # plt.plot(stride, texture_mem, label='Texture_4')

    # plt.xlabel('Stride')
    # plt.ylabel('Execution Time (seconds)')
    # plt.title('Execution Time vs. Stride')
    # plt.legend()
    # plt.show()

    # Plot 1
        # Filter Size of 4
        # X = Stride
        # Y = Execution Time
    """
    stride = [1, 2, 4]

    global_mem = []
    shared_mem = []
    texture_mem = []
    cpu = []

    for result in all_results:
        if result.filter_size == 4 and result.block_size == 4:
            global_mem.append(result.gpu_time["global"])
            shared_mem.append(result.gpu_time["shared"])
            texture_mem.append(result.gpu_time["texture"])
            cpu.append(result.cpu_time)

    bar_width = 0.25
    index = np.arange(len(stride))

    plt.bar(index, global_mem, width=bar_width, label='Global')
    plt.bar(index + bar_width, shared_mem, width=bar_width, label='Shared')
    plt.bar(index + 2 * bar_width, texture_mem, width=bar_width, label='Texture')
    plt.bar(index + 3 * bar_width, cpu, width=bar_width, label='CPU')

    plt.xlabel('Stride')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Memory Execution Time vs. Stride')
    plt.xticks(index + bar_width, stride)
    plt.legend()
    plt.suptitle('Filter Size = 4, Block Size = 4', fontsize=10)

    plt.savefig(os.path.join('plots', 'stride_vs_filter_4_and_block_4_and_cpu.png'))

    plt.show()
    """

    # Plot 2
        # Filter Size of 4
        # X = Stride
        # Y = Execution Time
    """
    stride = [1, 2, 4]

    global_mem = {4: [], 16: [], 32: []}
    shared_mem = {4: [], 16: [], 32: []}
    texture_mem = {4: [], 16: [], 32: []}

    for result in all_results:
        if result.filter_size == 4:
            global_mem[result.block_size].append(result.gpu_time["global"])
            shared_mem[result.block_size].append(result.gpu_time["shared"])
            texture_mem[result.block_size].append(result.gpu_time["texture"])

    block_sizes = [4, 16, 32]

    for block_size in block_sizes:
        plt.plot(stride, global_mem[block_size], label=f'Global_{block_size}')
        plt.plot(stride, shared_mem[block_size], label=f'Shared_{block_size}')
        plt.plot(stride, texture_mem[block_size], label=f'Texture_{block_size}')

    plt.xlabel('Data Points')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs. Data Points for Different Block Sizes')
    plt.legend()
    plt.show()
    """

    # Plot 3
        # Stride of 2
        # X = Filter Size
        # Y = Execution Time
    """
    filter_size = [4, 8]

    global_mem = []
    shared_mem = []
    texture_mem = []
    cpu = []

    for result in all_results:
        if result.stride == 2 and result.block_size == 16:
            global_mem.append(result.gpu_time["global"])
            shared_mem.append(result.gpu_time["shared"])
            texture_mem.append(result.gpu_time["texture"])
            cpu.append(result.cpu_time)

    bar_width = 0.25
    index = np.arange(len(filter_size))

    plt.bar(index, global_mem, width=bar_width, label='Global')
    plt.bar(index + bar_width, shared_mem, width=bar_width, label='Shared')
    plt.bar(index + 2 * bar_width, texture_mem, width=bar_width, label='Texture')
    plt.bar(index + 3 * bar_width, cpu, width=bar_width, label='CPU')

    plt.xlabel('Filter Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Memory Execution Time vs. Filter Size')
    plt.xticks(index + bar_width, filter_size)
    plt.legend()
    plt.suptitle('Stride = 2, Block Size = 16', fontsize=10)

    
    # plt.savefig(os.path.join('plots', 'filter_vs_stride_2_and_block_16_and_cpu.png'))

    plt.show()
    """

    # Plot 4
        # Stride = 1, filter_Size = 8
        # X = Block Size
        # Y = Execution Time
    """
    block_size = [4, 16, 32]

    global_mem = []
    shared_mem = []
    texture_mem = []

    for result in all_results:
        if result.stride == 1 and result.filter_size == 8:
            global_mem.append(result.gpu_time["global"])
            shared_mem.append(result.gpu_time["shared"])
            texture_mem.append(result.gpu_time["texture"])

    bar_width = 0.25
    index = np.arange(len(block_size))

    plt.bar(index, global_mem, width=bar_width, label='Global')
    plt.bar(index + bar_width, shared_mem, width=bar_width, label='Shared')
    plt.bar(index + 2 * bar_width, texture_mem, width=bar_width, label='Texture')

    plt.xlabel('Block Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Memory Execution Time vs. Block Size')
    plt.xticks(index + bar_width, block_size)
    plt.legend()
    plt.suptitle('Stride = 1, Filter Size = 8', fontsize=10)

    
    plt.savefig(os.path.join('plots', 'block_vs_stride_1_and_filter_8.png'))

    plt.show()
    """

    # Plot 5
        # Average out global, shared, and texture
        # X = Block Size
        # Y = Execution Time
    # """
    x_ticks = ['Average', 'Max', 'Min']

    global_mem = []
    shared_mem = []
    texture_mem = []

    for result in all_results:
            global_mem.append(result.gpu_time["global"])
            shared_mem.append(result.gpu_time["shared"])
            texture_mem.append(result.gpu_time["texture"])
        
    # [Average, Max, Min]
    global_mem_stats = [np.mean(global_mem), np.max(global_mem), np.min(global_mem)]
    shared_mem_stats = [np.mean(shared_mem), np.max(shared_mem), np.min(shared_mem)]
    texture_mem_stats = [np.mean(texture_mem), np.max(texture_mem), np.min(texture_mem)]

    bar_width = 0.25
    index = np.arange(len(x_ticks))

    plt.bar(index, global_mem_stats, width=bar_width, label='Global')
    plt.bar(index + bar_width, shared_mem_stats, width=bar_width, label='Shared')
    plt.bar(index + 2 * bar_width, texture_mem_stats, width=bar_width, label='Texture')

    plt.xlabel('Block Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Memory Execution Time vs. Block Size')
    plt.xticks(index + bar_width, labels=x_ticks)
    plt.legend()
    plt.suptitle('Stride = 1, Filter Size = 8', fontsize=10)

    
    # plt.savefig(os.path.join('plots', 'block_vs_stride_1_and_filter_8.png'))

    plt.show()
    # """
        

if __name__ == "__main__":
    main()