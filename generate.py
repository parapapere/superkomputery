import random
import json

tasks = 100
machines = 10

def generate_processing_times(num_tasks, num_machines):
    # Generowanie losowych czasów przetwarzania
    processing_times = [[random.randint(1, 50) for _ in range(num_machines)] for _ in range(num_tasks)]
    
    return processing_times

def save_processing_times_to_file(processing_times):
    # Zapis danych do pliku JSON
    with open(f"data/{tasks}_{machines}.json", 'w') as file:
        json.dump(processing_times, file)

times = generate_processing_times(tasks, machines)

save_processing_times_to_file(times)
