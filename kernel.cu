#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdio.h>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/random.h>

using namespace std;

// Struktura reprezentująca rozwiązanie (permutację zadań).
struct Solution {
    thrust::device_vector<int> permutation; // Kolejność zadań na urządzeniu.

    Solution(int n) : permutation(n) {}

    ~Solution() = default;
};

// Funkcja celu - minimalizacja sumy czasów przetwarzania wszystkich zadań na wszystkich maszynach.
__device__ int objectiveFunction(const int* permutation, const int* processingTimes, int n, int m) {
    extern __shared__ int sharedMem[];

    int* completionTimes = sharedMem;
    for (int i = 0; i < m; ++i) {
        completionTimes[i] = 0;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            completionTimes[j] += processingTimes[permutation[i] * m + j];
            if (i > 0) {
                completionTimes[j] = max(completionTimes[j], completionTimes[j - 1]);
            }
        }
    }

    int totalCompletionTime = completionTimes[m - 1];

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


// Funkcja znajduje powtarzające się elementy z listy tabu.
__device__ int* findUnique(int* tabuListPtr, int tabuSize, int n) {
    for (int i = 0; i < n; ++i) {
        int* begin = tabuListPtr + i * tabuSize;
        // Sortowanie listy tabu
        for (int j = 0; j < tabuSize - 1; ++j) {
            for (int k = 0; k < tabuSize - j - 1; ++k) {
                if (begin[k] > begin[k + 1]) {
                    int temp = begin[k];
                    begin[k] = begin[k + 1];
                    begin[k + 1] = temp;
                }
            }
        }
        int newSize = 0;
        for (int j = 0; j < tabuSize; ++j) {
            if (j == 0 || begin[j] != begin[j - 1]) {
                // Kopiowanie unikalnego elementu do docelowej tablicy
                begin[newSize++] = begin[j];
            }
        }
    }
    return tabuListPtr;
}

// Algorytm Tabu Search.
__global__ void tabuSearchKernel(int* bestSolution, int* bestObjective, const int* processingTimes, int n, int m, int maxIterations, int tabuSize) {
    // Inicjalizacja generatora liczb losowych. 
    // clock64 - zwraca liczbę cykli zegara GPU., threadIdx.x - numer wątku w bloku, blockIdx.x - numer bloku, blockDim.x - liczba wątków w bloku
    thrust::default_random_engine rng(clock64() + threadIdx.x + blockIdx.x * blockDim.x);

    // Przygotowanie miejsca na permutację i najlepsze rozwiązanie na urządzeniu.
    int* currentSolution = new int[n];
    int* tabuList = new int[tabuSize * n];
    int* bestSolutionLocal = new int[n];
    int currentObjective, bestObjectiveLocal;

    // Inicjalizacja permutacji.
    for (int i = 0; i < n; ++i) {
        currentSolution[i] = i;
    }

    currentObjective = objectiveFunction(currentSolution, processingTimes, n, m);
    bestObjectiveLocal = currentObjective;

    // Główna pętla algorytmu.
    for (int iter = 0; iter < maxIterations; ++iter) {
        int* neighborSolution = new int[n];
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

        // Dodanie ruchu do listy tabu.
        memcpy(&tabuList[(iter % tabuSize) * n], currentSolution, n * sizeof(int));
        memcpy(&tabuList[((iter % tabuSize) + 1) * n], neighborSolution, n * sizeof(int));


        // Wyszukaj unikalne elementy w liście tabu.
        int* tabuListPtr = tabuList;
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        // Sortowanie listy tabu
        for (size_t i = 0; i < tabuSize - 1; ++i) {
            for (size_t j = i + 1; j < tabuSize; ++j) {
                if (tabuList[tid * tabuSize + i] > tabuList[tid * tabuSize + j]) {
                    // Zamiana miejscami
                    int temp = tabuList[tid * tabuSize + i];
                    tabuList[tid * tabuSize + i] = tabuList[tid * tabuSize + j];
                    tabuList[tid * tabuSize + j] = temp;
                }
            }
        }

        // Wyszukanie elementów powtarzających się
        int* uniqueEnd = findUnique(tabuListPtr, tabuSize, n);

        // Skopiowanie unikalnych elementów do początku listy tabu
        // thrust::distance - odległóść między dwoma elementami
        if (thrust::distance(uniqueEnd, tabuListPtr) > tabuSize * n) {
            int* destIter = tabuListPtr;

            // Iteracja przez źródłową tablicę
            for (int* sourceIter = uniqueEnd - tabuSize * n; sourceIter != uniqueEnd; ++sourceIter) {
                // Sprawdzenie, czy obecny element jest różny od poprzedniego
                if (sourceIter == uniqueEnd - tabuSize * n || *sourceIter != *(sourceIter - 1)) {
                    // Kopiowanie unikalnego elementu do docelowej tablicy
                    *destIter = *sourceIter;
                    ++destIter;
                }
            }
        }

        delete[] neighborSolution;
    }

    // Zapisz najlepsze rozwiązanie do wyniku.

    memcpy(bestSolution, bestSolutionLocal, n * sizeof(int));
    *bestObjective = bestObjectiveLocal;

    delete[] currentSolution;
    delete[] tabuList;
    delete[] bestSolutionLocal;
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    // Przykładowe czasy przetwarzania 
    vector<vector<int>> processingTimes = {
        {8, 6, 9},
        {12, 2, 4},
        {1, 3, 2},
        {7, 3, 5},
        {5, 32, 1},
        {8, 6, 9},
        {12, 32, 4},
        {1, 23, 2},
        {71, 3, 5},
        {5, 31, 1},
        {8, 62, 9},
        {12, 22, 4},
        {1, 33, 2},
        {7, 3, 5},
        {51, 3, 1},
        {8, 62, 9},
        {12, 21, 4},
        {1, 3, 21},
        {37, 3, 15},
        {51, 3, 21}
    };

    int n = processingTimes.size(); // Liczba zadań.
    int m = processingTimes[0].size(); // Liczba maszyn.

    // Parametry algorytmu.
    int maxIterations = 100;
    int tabuSize = 5;

    // Przeniesienie danych czasów przetwarzania na urządzenie.
    int* d_processingTimes;
    cudaMalloc((void**)&d_processingTimes, n * m * sizeof(int));
    cudaMemcpy(d_processingTimes, processingTimes.data(), n * m * sizeof(int), cudaMemcpyHostToDevice);

    // Przygotowanie miejsca na wynik na urządzeniu.
    int* d_bestSolution;
    cudaMalloc((void**)&d_bestSolution, n * sizeof(int));
    int* d_bestObjective;
    cudaMalloc((void**)&d_bestObjective, sizeof(int));

    // Definicja liczby bloków i wątków.
    int blocks = 1;
    int threads = 1;


    // Uruchomienie algorytmu Tabu Search na urządzeniu.
    tabuSearchKernel << <blocks, threads>> > (d_bestSolution, d_bestObjective, d_processingTimes, n, m, maxIterations, tabuSize);

    // Oczekiwanie na zakończenie operacji na GPU
    cudaDeviceSynchronize();

    // Pobranie wyniku z urządzenia.
    int* h_bestSolution = new int[n];
    int h_bestObjective;

    cudaMemcpy(h_bestSolution, d_bestSolution, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_bestObjective, d_bestObjective, sizeof(int), cudaMemcpyDeviceToHost);

    // Wyświetlenie wyniku.
    cout << "Najlepsza permutacja: ";
    for (int i = 0; i < n; ++i) {
        cout << h_bestSolution[i] << " ";
    }
    cout << "\nCzas przetwarzania wszystkich zadań na wszystkich maszynach: " << h_bestObjective << endl;

    // Zwolnienie pamięci na urządzeniu i hostingu.
    cudaFree(d_processingTimes);
    cudaFree(d_bestSolution);
    cudaFree(d_bestObjective);
    delete[] h_bestSolution;

    return 0;
}
