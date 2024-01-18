import subprocess
import time

def run_single_threaded(iteration):
    print(f"Running Single-Threaded Version - Iteration {iteration}")
    start_time = time.time()
    subprocess.run(['./your_single_threaded_executable'], check=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

def run_multi_threaded(iteration):
    print(f"\nRunning Multi-Threaded Version - Iteration {iteration}")
    start_time = time.time()
    subprocess.run(['./your_multi_threaded_executable'], check=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

if __name__ == "__main__":
    n_iterations = 5  # Change this to the desired number of iterations
    output_file = "execution_times.txt"

    with open(output_file, "w") as f:
        f.write("Iteration,SingleThreadedTime,MultiThreadedTime\n")
        for iteration in range(1, n_iterations + 1):
            single_threaded_time = run_single_threaded(iteration)
            multi_threaded_time = run_multi_threaded(iteration)

            print(f"\nIteration {iteration} - Single-Threaded Time: {single_threaded_time:.2f} seconds")
            print(f"Iteration {iteration} - Multi-Threaded Time: {multi_threaded_time:.2f} seconds")

            f.write(f"{iteration},{single_threaded_time},{multi_threaded_time}\n")

    print(f"\nExecution times saved to {output_file}")