import subprocess
import time

def compile_and_run(executable_name, source_filename):
    print(f"Compiling {source_filename}...")
    compile_command = f"g++ -std=c++11 -o {executable_name} {source_filename}"
    subprocess.run(compile_command, shell=True, check=True)

    print(f"Running {executable_name}...")
    start_time = time.time()
    subprocess.run([f'./{executable_name}'], shell=True, check=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time

if __name__ == "__main__":
    # Replace 'mark2' and 'mark2single' with your actual executable names
    multi_threaded_executable = 'mark2'
    single_threaded_executable = 'mark2single'

    multi_threaded_time = compile_and_run(multi_threaded_executable, 'mark2.cpp')
    single_threaded_time = compile_and_run(single_threaded_executable, 'mark2single.cpp')

    print(f"\nSingle-Threaded Time: {single_threaded_time:.2f} seconds")
    print(f"Multi-Threaded Time: {multi_threaded_time:.2f} seconds")
