import subprocess
import time

def compile_and_run(executable_name, source_filename, iteration, max_iterations_per_thread, tabu_list_size, file_name):
    print(f"Compiling {source_filename}...")
    compile_command = f"g++ -std=c++11 -o {executable_name} {source_filename} -L/path/to/jsoncpp/lib -ljsoncpp"
    subprocess.run(compile_command, shell=True, check=True)

    print(f"Running {executable_name}...")
    start_time = time.time()
    subprocess.run(f'./{executable_name} {str(iteration)} {str(max_iterations_per_thread)} {str(tabu_list_size)}  {str(file_name)}', shell=True, check=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

if __name__ == "__main__":
    # Set parameters for the loop
    num_iterations = 8

    max_iterations_per_thread = 100
    tabu_list_size = 10
    json_file_name = "data/80_10.json"

    
    with open("execution_times_100_100.txt", "a") as f:
        f.write("NumThreads,SingleThreadedTime,MultiThreadedTime\n")
        

        for iteration in range(1, num_iterations + 1):
            
            multi_threaded_executable = f'mark2_{iteration}'
            single_threaded_executable = f'makr2single_{iteration}'
            
            total_single_threaded_time = 0
            total_multi_threaded_time = 0
            
            single_threaded_time = compile_and_run(single_threaded_executable, 'makr2single.cpp', iteration, max_iterations_per_thread, tabu_list_size, json_file_name )
            multi_threaded_time = compile_and_run(multi_threaded_executable, 'mark2.cpp', iteration, max_iterations_per_thread, tabu_list_size, json_file_name )
            
            print(f"\nIteration {iteration} - NumThreads: {iteration}")
            print(f"Single-Threaded Time: {single_threaded_time:.2f} seconds")
            print(f"Multi-Threaded Time: {multi_threaded_time:.2f} seconds")
            
            
            f.write(f"{iteration},{max_iterations_per_thread},{tabu_list_size},{single_threaded_time},{multi_threaded_time}\n")

    print(f"\nExecution times saved to execution_times.txt")
import subprocess
import time

def compile_and_run(executable_name, source_filename, iteration, max_iterations_per_thread, tabu_list_size, file_name):
    print(f"Compiling {source_filename}...")
    compile_command = f"g++ -std=c++11 -o {executable_name} {source_filename} -L/path/to/jsoncpp/lib -ljsoncpp"
    subprocess.run(compile_command, shell=True, check=True)

    print(f"Running {executable_name}...")
    start_time = time.time()
    subprocess.run(f'./{executable_name} {str(iteration)} {str(max_iterations_per_thread)} {str(tabu_list_size)}  {str(file_name)}', shell=True, check=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

if __name__ == "__main__":
    # Set parameters for the loop
    num_iterations = 8

    max_iterations_per_thread = 100
    tabu_list_size = 10
    json_file_name = "data/80_10.json"

    
    with open("execution_times_100_100.txt", "a") as f:
        f.write("NumThreads,SingleThreadedTime,MultiThreadedTime\n")
        

        for iteration in range(1, num_iterations + 1):
            
            multi_threaded_executable = f'mark2_{iteration}'
            single_threaded_executable = f'makr2single_{iteration}'
            
            total_single_threaded_time = 0
            total_multi_threaded_time = 0
            
            single_threaded_time = compile_and_run(single_threaded_executable, 'makr2single.cpp', iteration, max_iterations_per_thread, tabu_list_size, json_file_name )
            multi_threaded_time = compile_and_run(multi_threaded_executable, 'mark2.cpp', iteration, max_iterations_per_thread, tabu_list_size, json_file_name )
            
            print(f"\nIteration {iteration} - NumThreads: {iteration}")
            print(f"Single-Threaded Time: {single_threaded_time:.2f} seconds")
            print(f"Multi-Threaded Time: {multi_threaded_time:.2f} seconds")
            
            
            f.write(f"{iteration},{max_iterations_per_thread},{tabu_list_size},{single_threaded_time},{multi_threaded_time}\n")

    print(f"\nExecution times saved to execution_times.txt")
