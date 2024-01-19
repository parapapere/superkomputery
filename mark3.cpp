#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <ctime>
#include <cstdlib>
#include <random>  
#include <thread>
#include <mutex>
#include <fstream>
#include <json/json.h>

using namespace std;

mutex tabuListMutex;

// Struktura reprezentująca rozwiązanie (permutację zadań).
struct Solution {
    vector<int> permutation; // Kolejność zadań.

    Solution() = default;

    Solution(int n) {
        for (int i = 0; i < n; ++i) {
            permutation.push_back(i);
        }
        // Permutacja początkowa.
        shuffle(permutation.begin(), permutation.end(), mt19937(random_device()()));
    }
};

// Funkcja celu - minimalizacja sumy czasów przetwarzania wszystkich zadań na wszystkich maszynach.
int objectiveFunction(const Solution& solution, const vector<vector<int>>& processingTimes) {
    int n = solution.permutation.size();
    int m = processingTimes[0].size();

    vector<int> completionTimes(m, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            // Dodajemy czas przetwarzania dla danego zadania i maszyny.
            completionTimes[j] += processingTimes[solution.permutation[i]][j];

            // Jeżeli to nie pierwsza operacja na maszynie i nie pierwsza maszyna, uwzględniamy czas oczekiwania na poprzednie zadanie.
            if (i >= 0 && j > 0) {
                completionTimes[j] = max(completionTimes[j], completionTimes[j - 1]);
            }

            if (i == 0 && j == 0) {
                completionTimes[j] = 0;
            }
        }
    }

    // Sumujemy czasy zakończenia zadań na wszystkich maszynach.
    int totalCompletionTime = completionTimes.back();

    return totalCompletionTime;
}

// Funkcja generująca sąsiedztwo - zamiana dwóch zadań.
Solution generateNeighbor(const Solution& currentSolution) {
    Solution neighbor = currentSolution;
    int n = neighbor.permutation.size();

    int idx1 = rand() % n;
    int idx2 = rand() % n;

    swap(neighbor.permutation[idx1], neighbor.permutation[idx2]);

    return neighbor;
}

// Algorytm Tabu Search.
Solution tabuSearch(const vector<vector<int> >& processingTimes, int maxIterations, int tabuSize) {
    int n = processingTimes.size();
    Solution currentSolution(n);
    Solution bestSolution = currentSolution;
    int currentObjective = objectiveFunction(currentSolution, processingTimes);
    int bestObjective = currentObjective;

    vector<vector<int> > tabuList(tabuSize, vector<int>(n, -1));

    for (int iter = 0; iter < maxIterations; ++iter) {
        Solution neighbor = generateNeighbor(currentSolution);
        int neighborObjective = objectiveFunction(neighbor, processingTimes);

        // Sprawdź, czy ruch jest dozwolony na podstawie listy tabu.
        bool isTabu = false;
        for (const auto& tabuMove : tabuList) {
            if (tabuMove == neighbor.permutation) {
                isTabu = true;
                break;
            }
        }

        // Aktualizacja rozwiązania.
        if (!isTabu && neighborObjective < currentObjective) {
            currentSolution = neighbor;
            currentObjective = neighborObjective;

            // Aktualizacja najlepszego rozwiązania.
            if (currentObjective < bestObjective) {
                bestSolution = currentSolution;
                bestObjective = currentObjective;
            }
        }

        // Dodanie ruchu do listy tabu.
        tabuList.push_back(currentSolution.permutation);
        if (tabuList.size() > tabuSize) {
            tabuList.erase(tabuList.begin());
        }
    }

    return bestSolution;
}

// Funkcja wczytująca dane z pliku JSON.
vector<vector<int>> loadProcessingTimesFromJSON(const string& filePath) {
    vector<vector<int>> processingTimes;

    ifstream file(filePath);
    if (file.is_open()) {
        Json::Value root;
        file >> root;

        for (const auto& row : root) {
            vector<int> rowData;
            for (const auto& value : row) {
                rowData.push_back(value.asInt());
            }
            processingTimes.push_back(rowData);
        }

        file.close();
    } else {
        cerr << "Unable to open file: " << filePath << endl;
    }

    return processingTimes;
}

// Wątek dla algorytmu Tabu Search.
void tabuSearchThread(const vector<vector<int>>& processingTimes, int maxIterations, int tabuSize, Solution& bestSolution, int& bestObjective) {
    Solution localBestSolution;
    int localBestObjective = numeric_limits<int>::max();

    vector<vector<int>> localTabuList(tabuSize, vector<int>(processingTimes.size(), -1));

    for (int iter = 0; iter < maxIterations; ++iter) {
        Solution neighbor = generateNeighbor(bestSolution);
        int neighborObjective = objectiveFunction(neighbor, processingTimes);

        bool isTabu = false;

        // Synchronizacja dostępu do listy tabu.
        {
            lock_guard<mutex> lock(tabuListMutex);
            for (const auto& tabuMove : localTabuList) {
                if (tabuMove == neighbor.permutation) {
                    isTabu = true;
                    break;
                }
            }
        }

        if (!isTabu && neighborObjective < bestObjective) {
            bestSolution = neighbor;
            bestObjective = neighborObjective;

            if (bestObjective < localBestObjective) {
                localBestSolution = bestSolution;
                localBestObjective = bestObjective;
            }
        }

        // Synchronizacja dostępu do listy tabu.
        {
            lock_guard<mutex> lock(tabuListMutex);
            localTabuList.push_back(bestSolution.permutation);
            if (localTabuList.size() > tabuSize) {
                localTabuList.erase(localTabuList.begin());
            }
        }
    }

    // Synchronizacja najlepszego rozwiązania.
    {
        lock_guard<mutex> lock(tabuListMutex);
        if (localBestObjective < bestObjective) {
            bestSolution = localBestSolution;
            bestObjective = localBestObjective;
        }
    }
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    // Ścieżka do pliku JSON
    string filePath = "data/100_10.json";

    // Wczytanie danych z pliku JSON
    vector<vector<int>> processingTimes = loadProcessingTimesFromJSON(filePath);

    // Parametry algorytmu.
    int maxIterations = 100;
    int tabuSize = 10;

    // Uruchomienie algorytmu Tabu Search.
    Solution result = tabuSearch(processingTimes, maxIterations, tabuSize);

    // Wyświetlenie wyniku.
    cout << "Najlepsza permutacja: ";
    for (int task : result.permutation) {
        cout << task << " ";
    }
    cout << "\nCzas przetwarzania wszystkich zadań na wszystkich maszynach: " << objectiveFunction(result, processingTimes) << endl;

    // Parametry dla wielowątkowego rozwiązania.
    int numThreads = 4;
    vector<thread> threads;
    vector<Solution> bestSolutions(numThreads, result);
    vector<int> bestObjectives(numThreads, objectiveFunction(result, processingTimes));

    // Uruchomienie Tabu Search dla każdego wątku.
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back(tabuSearchThread, ref(processingTimes), maxIterations, tabuSize, ref(bestSolutions[t]), ref(bestObjectives[t]));
    }

    // Czekanie na zakończenie wszystkich wątków.
    for (auto& thread : threads) {
        thread.join();
    }

    // Wybieranie najlepszego rozwiązania spośród wszystkich wątków.
    int globalBestIndex = min_element(bestObjectives.begin(), bestObjectives.end()) - bestObjectives.begin();
    Solution globalBestSolution = bestSolutions[globalBestIndex];

    // Wyświetlanie wyniku.
    cout << "Najlepsza permutacja (wielowątkowo): ";
    for (int task : globalBestSolution.permutation) {
        cout << task << " ";
    }
    cout << "\nCzas przetwarzania wszystkich zadań na wszystkich maszynach: " << objectiveFunction(globalBestSolution, processingTimes) << endl;

    return 0;
}
