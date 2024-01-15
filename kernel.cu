#include <iostream>
#include <vector>
#include <cstdio>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <numeric>
#include <stdio.h>
#include <limits>
#include <ctime>
#include <stdexcept>
#include <thrust/device_vector.h>
#include <thrust/random.h>

#define SHOW_MESSAGES 1

using namespace std;

// Funkcja celu - minimalizacja sumy czasów przetwarzania wszystkich zadań na wszystkich maszynach.
__device__ int objectiveFunction(const int* permutation, const int* processingTimes, int n, int m) {

    int* completionTimes = new int[m];
    for (int i = 0; i < m; ++i) {
        completionTimes[i] = 0;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            // Dodajemy czas przetwarzania dla danego zadania i maszyny.
            completionTimes[j] += processingTimes[permutation[i] * m + j];

            // Jeżeli to nie pierwsza operacja na maszynie i nie pierwsza maszyna, uwzględniamy czas oczekiwania na poprzednie zadanie.
            if (i > 0 && j > 0) {
                completionTimes[j] = max(completionTimes[j], completionTimes[j - 1]);
            }
            if (i == 0 && j == 0)
            {
                completionTimes[j] = 0;
            }
        }
    }

    int totalCompletionTime = completionTimes[m - 1];

    if (SHOW_MESSAGES == 1)
    {
        printf("\nObjectiveFunction || CompletionTimes\n");
        for (int i = 0; i < m; ++i) {
            printf("%d ", completionTimes[i]);
        }
        printf("\n");
    }

    delete[] completionTimes;

    return totalCompletionTime;
}

// Funkcja generująca sąsiedztwo - zamiana dwóch zadań.
__device__ void generateNeighbor(int* currentPermutation, int n, thrust::default_random_engine& rng) {
    // Wygenerowanie dwóch losowych indeksów.
    // Generowanie liczb całkowitych z przedziału [0, n - 1].
    thrust::uniform_int_distribution<int> distribution(0, n - 1);

    int idx1 = distribution(rng);
    int idx2 = distribution(rng);

    int temp = currentPermutation[idx1];
    currentPermutation[idx1] = currentPermutation[idx2];
    currentPermutation[idx2] = temp;
}

// Algorytm Tabu Search.
__global__ void tabuSearchKernel(int* bestSolution, int* bestObjective, const int* processingTimes, int* bestTimes, int n, int m, int maxIterations, int tabuSize) {
    // Inicjalizacja generatora liczb losowych. 
    // clock64 - zwraca liczbę cykli zegara GPU., threadIdx.x - numer wątku w bloku, blockIdx.x - numer bloku, blockDim.x - liczba wątków w bloku
    //thrust::default_random_engine rng(clock64() + threadIdx.x + blockIdx.x * blockDim.x);

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = threadIdx.x + blockId * blockDim.x;
    thrust::default_random_engine rng(clock64() + threadId);

    // Przygotowanie miejsca na permutację i najlepsze rozwiązanie na urządzeniu.

    int* currentSolution = new int[n];
    int* tabuList = new int[tabuSize * n];
    int* bestSolutionLocal = new int[n];
    int currentObjective, bestObjectiveLocal, best_time;

    // Inicjalizacja permutacji.
    for (int i = 0; i < n; ++i) {
        currentSolution[i] = i;
    }

    currentObjective = objectiveFunction(currentSolution, processingTimes, n, m);
    bestObjectiveLocal = currentObjective;

    if (SHOW_MESSAGES == 1)
    {
        // Wyświetl CurrentSolution
        for (int i = 0; i < n; ++i) {
            printf("%d ", currentSolution[i]);
        }
        printf("\n");

        printf("Funkcja celu: %d\n", currentObjective);
    }

    // Główna pętla algorytmu.
    for (int iter = 0; iter < maxIterations; ++iter) {
        int* neighborSolution = new int[n];
        if (threadId < n)
        {
            memcpy(neighborSolution, currentSolution, n * sizeof(int));
            generateNeighbor(neighborSolution, n, rng);

            int neighborObjective = objectiveFunction(neighborSolution, processingTimes, n, m);

            // Sprawdź, czy ruch jest dozwolony na podstawie listy tabu.
            bool isTabu = false;
            for (int i = 0; i < tabuSize; ++i) {
                bool isEqual = true;
                for (int j = 0; j < n; ++j) {
                    if (tabuList[i * n + j] != neighborSolution[j]) {
                        isEqual = false;
                        break;
                    }
                }
                if (isEqual) {
                    isTabu = true;
                    break;
                }
            }

            // Aktualizacja rozwiązania.
            if (!isTabu && neighborObjective < currentObjective) {

                memcpy(currentSolution, neighborSolution, n * sizeof(int));
                currentObjective = neighborObjective;

                // Aktualizacja najlepszego rozwiązania.
                if (currentObjective < bestObjectiveLocal) {
                    bestObjectiveLocal = currentObjective;
                    memcpy(bestSolutionLocal, currentSolution, n * sizeof(int));
                }
            }
        }

        __syncthreads();  // Synchronizacja wątków przed kontynuacją


        if (SHOW_MESSAGES == 1)
        {
            printf("Iteracja: %d | Czas: %d | Czas_bestObjectiveLocal = %d\n", iter, currentObjective, bestObjectiveLocal);
        }


        // Dodanie ruchu do listy tabu.

        if (threadId < n)
        {
            memcpy(&tabuList[(iter % tabuSize) * n], currentSolution, n * sizeof(int));
            memcpy(&tabuList[((iter % tabuSize) + 1) * n], neighborSolution, n * sizeof(int));

            // Ograniczenie rozmiaru listy tabu
            if ((iter + 1) > tabuSize) {
                // Przycięcie listy do ustalonego rozmiaru
                for (int i = 0; i < (tabuSize - 1); ++i) {
                    memcpy(&tabuList[i], &tabuList[i + 1], n * sizeof(int));
                }
            }
        }

        __syncthreads();  // Synchronizacja wątków przed kontynuacją

        delete[] neighborSolution;
    }

    // Zapisz najlepsze rozwiązanie do wyniku.

    if (threadIdx.x == 0)
    {
        memcpy(bestSolution, bestSolutionLocal, n * sizeof(int));
        *bestObjective = bestObjectiveLocal;

        best_time = objectiveFunction(bestSolution, processingTimes, n, m);
        *bestTimes = best_time;
    }

    delete[] currentSolution;
    delete[] tabuList;
    delete[] bestSolutionLocal;
}


vector<vector<int>> readCSV(string nazwaPliku)
{
    vector<std::vector<int>> dane;
    ifstream plik(nazwaPliku);
    if (!plik.is_open()) {
        cout << "Nie można otworzyć pliku: " << nazwaPliku << endl;
        return dane;  // Zwraca pustą tablicę, jeśli nie udało się otworzyć pliku
    }
    string linia;
    while (getline(plik, linia)) {
        vector<int> wiersz;
        stringstream ss(linia);
        string pole;

        while (getline(ss, pole, ',')) {
            try {
                int wartosc = stoi(pole);
                wiersz.push_back(wartosc);
            }
            catch (const invalid_argument& e) {
                cout << "Błąd konwersji danych: " << e.what() << endl;
                // Możesz obsłużyć błąd konwersji w inny sposób lub zignorować błąd
            }
        }
        dane.push_back(wiersz);
    }

    plik.close();
    return dane;
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    setlocale(LC_ALL, "Polish");

    // Wczytaj dane z pliku csv
    string filename = "100_10.csv";
    vector<vector<int>> processingTimes = readCSV(filename);

    int n = processingTimes.size(); // Liczba zadań.
    int m = processingTimes[0].size(); // Liczba maszyn.

    // Parametry algorytmu.
    int maxIterations = 100;
    int tabuSize = 5;

    /*
    // Przeniesienie danych czasów przetwarzania na urządzenie.
    int* d_processingTimes;
    cudaMalloc((void**)&d_processingTimes, n * m * sizeof(int));
    cudaMemcpy(d_processingTimes, processingTimes.data(), n * m * sizeof(int), cudaMemcpyHostToDevice);
    */

    // Kopiowanie danych z wektora do jednowymiarowej tablicy
    int* h_processingTimes = new int[n * m];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            h_processingTimes[i * m + j] = processingTimes[i][j];
        }
    }

    // Kopiowanie danych z CPU do GPU
    int* d_processingTimes;
    cudaMalloc((void**)&d_processingTimes, n * m * sizeof(int));
    cudaMemcpy(d_processingTimes, h_processingTimes, n * m * sizeof(int), cudaMemcpyHostToDevice);



    // Przygotowanie miejsca na wynik na urządzeniu.
    int* d_bestSolution;
    cudaMalloc((void**)&d_bestSolution, n * sizeof(int));
    int* d_bestObjective;
    cudaMalloc((void**)&d_bestObjective, sizeof(int));
    int* d_bestTimes;
    cudaMalloc((void**)&d_bestTimes, sizeof(int));

    // Definicja liczby bloków i wątków.
    int blocks = 1;
    int threads = 1;


    // Uruchomienie algorytmu Tabu Search na urządzeniu.
    tabuSearchKernel << <blocks, threads >> > (d_bestSolution, d_bestObjective, d_processingTimes, d_bestTimes, n, m, maxIterations, tabuSize);


    // Pobranie wyniku z urządzenia.
    int* h_bestSolution = new int[n];
    int h_bestObjective, h_bestTimes;

    cudaMemcpy(h_bestSolution, d_bestSolution, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_bestObjective, d_bestObjective, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_bestTimes, d_bestTimes, sizeof(int), cudaMemcpyDeviceToHost);

    // Oczekiwanie na zakończenie operacji na GPU
    cudaDeviceSynchronize();

    // Wyświetlenie wyniku.
    cout << "\nPARAMETRY: blocks = " << blocks << " threads = " << threads << " || maxIterations = " << maxIterations << " tabuSize = " << tabuSize << endl;
    cout << "Najlepsza permutacja: ";
    for (int i = 0; i < n; ++i) {
        cout << h_bestSolution[i] << " ";
    }
    cout << "\nCzas przetwarzania wszystkich zadań na wszystkich maszynach: " << h_bestTimes << endl;
    cout << "\nCzas przetwarzania wszystkich zadań na wszystkich maszynach: " << h_bestObjective << endl;


    // Zwolnienie pamięci na urządzeniu i hostingu.
    cudaFree(d_processingTimes);
    cudaFree(d_bestSolution);
    cudaFree(d_bestObjective);
    cudaFree(d_bestTimes);
    delete[] h_bestSolution;

    return 0;
}