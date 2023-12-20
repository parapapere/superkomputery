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

struct Solution {
    vector<int> permutation;

    Solution() = default;

    Solution(int n) {
        for (int i = 0; i < n; ++i) {
            permutation.push_back(i);
        }
        shuffle(permutation.begin(), permutation.end(), mt19937(random_device()()));
    }

    Solution& operator=(const Solution& other) {
        if (this != &other) {
            permutation = other.permutation;
        }
        return *this;
    }
};

int objectiveFunction(const Solution& solution, const vector<vector<int>>& processingTimes) {
    int n = solution.permutation.size();
    int m = processingTimes[0].size();
    vector<int> completionTimes(m, 0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            completionTimes[j] += processingTimes[solution.permutation[i]][j];
            if (i > 0 && j > 0) {
                completionTimes[j] = max(completionTimes[j], completionTimes[j - 1]);
            }
            if (i == 0 && j == 0) {
                completionTimes[j] = 0;
            }
        }
    }

    return completionTimes.back();
}

Solution generateNeighbor(const Solution& currentSolution) {
    Solution neighbor = currentSolution;
    int n = neighbor.permutation.size();
    int idx1 = rand() % n;
    int idx2 = rand() % n;
    swap(neighbor.permutation[idx1], neighbor.permutation[idx2]);
    return neighbor;
}

int evaluateNeighbor(const Solution& neighbor, const vector<vector<int>>& processingTimes) {
    return objectiveFunction(neighbor, processingTimes);
}

void generateAndEvaluateNeighbors(const Solution& currentSolution, const vector<vector<int>>& processingTimes,
                                  int maxIterations, int tabuSize, int threadId, int numThreads,
                                  Solution& localBestSolution, int& localBestObjective) {
    int n = currentSolution.permutation.size();
    vector<vector<int>> localTabuList(tabuSize, vector<int>(n, -1));

    for (int iter = 0; iter < maxIterations; ++iter) {
        Solution neighbor = generateNeighbor(currentSolution);
        int neighborObjective = evaluateNeighbor(neighbor, processingTimes);
        bool isTabu = false;

        {
            lock_guard<mutex> lock(tabuListMutex);
            for (const auto& tabuMove : localTabuList) {
                if (tabuMove == neighbor.permutation) {
                    isTabu = true;
                    break;
                }
            }
        }

        if (!isTabu && neighborObjective < localBestObjective) {
            localBestSolution = neighbor;
            localBestObjective = neighborObjective;
        }

        {
            lock_guard<mutex> lock(tabuListMutex);
            localTabuList.push_back(neighbor.permutation);
            if (localTabuList.size() > tabuSize) {
                localTabuList.erase(localTabuList.begin());
            }
        }
    }
}

Solution tabuSearch(const vector<vector<int>>& processingTimes, int maxIterations, int tabuSize) {
    int n = processingTimes.size();
    Solution currentSolution(n);
    Solution bestSolution = currentSolution;
    int currentObjective = objectiveFunction(currentSolution, processingTimes);
    int bestObjective = currentObjective;

    vector<vector<int>> tabuList(tabuSize, vector<int>(n, -1));

    for (int iter = 0; iter < maxIterations; ++iter) {
        Solution neighbor = generateNeighbor(currentSolution);
        int neighborObjective = evaluateNeighbor(neighbor, processingTimes);

        bool isTabu = false;
        for (const auto& tabuMove : tabuList) {
            if (tabuMove == neighbor.permutation) {
                isTabu = true;
                break;
            }
        }

        if (!isTabu && neighborObjective < currentObjective) {
            currentSolution = neighbor;
            currentObjective = neighborObjective;

            if (currentObjective < bestObjective) {
                bestSolution = currentSolution;
                bestObjective = currentObjective;
            }
        }

        tabuList.push_back(currentSolution.permutation);
        if (tabuList.size() > tabuSize) {
            tabuList.erase(tabuList.begin());
        }
    }

    return bestSolution;
}

void parallelTabuSearch(const vector<vector<int>>& processingTimes, int maxIterations, int tabuSize, int numThreads) {
    vector<thread> threads;
    vector<Solution> bestSolutions(numThreads);
    vector<int> bestObjectives(numThreads, numeric_limits<int>::max());

    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            Solution currentSolution(processingTimes.size());
            Solution localBestSolution = currentSolution;
            int currentObjective = objectiveFunction(currentSolution, processingTimes);
            int localBestObjective = currentObjective;

            generateAndEvaluateNeighbors(currentSolution, processingTimes, maxIterations, tabuSize, t, numThreads,
                                         localBestSolution, localBestObjective);

            bestSolutions[t] = localBestSolution;
            bestObjectives[t] = localBestObjective;
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    int globalBestIndex = min_element(bestObjectives.begin(), bestObjectives.end()) - bestObjectives.begin();
    Solution globalBestSolution = bestSolutions[globalBestIndex];

    cout << "Najlepsza permutacja: ";
    for (int task : globalBestSolution.permutation) {
        cout << task << " ";
    }
    cout << "\nCzas przetwarzania wszystkich zadań na wszystkich maszynach: "
         << objectiveFunction(globalBestSolution, processingTimes) << endl;
}

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

int main() {
    srand(static_cast<unsigned>(time(0)));

    string filePath = "data/100_10.json";
    vector<vector<int>> processingTimes = loadProcessingTimesFromJSON(filePath);

    int maxIterations = 100;
    int tabuSize = 10;
    int numThreads = 4;

    Solution result = tabuSearch(processingTimes, maxIterations, tabuSize);

    cout << "Najlepsza permutacja: ";
    for (int task : result.permutation) {
        cout << task << " ";
    }
    cout << "\nCzas przetwarzania wszystkich zadań na wszystkich maszynach: "
         << objectiveFunction(result, processingTimes) << endl;

    parallelTabuSearch(processingTimes, maxIterations, tabuSize, numThreads);

    return 0;
}
