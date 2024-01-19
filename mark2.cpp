#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <ctime>
#include <thread>
#include <mutex>
#include <json/json.h>

using namespace std;

// Function to calculate makespan for a given permutation of jobs
int calculateMakespan(const vector<int>& permutation, const vector<vector<int>>& processingTimes) {
    int numJobs = permutation.size();
    int numMachines = processingTimes[0].size();

    vector<int> completionTimes(numMachines, 0);

    for (int jobIndex : permutation) {
        completionTimes[0] += processingTimes[jobIndex][0];

        for (int machine = 1; machine < numMachines; ++machine) {
            completionTimes[machine] = max(completionTimes[machine - 1], completionTimes[machine]) + processingTimes[jobIndex][machine];
        }
    }

    return completionTimes[numMachines - 1];
}

vector<int> generateRandomSolution(int numJobs) {
    vector<int> solution(numJobs);
    for (int i = 0; i < numJobs; ++i) {
        solution[i] = i;
    }
    random_shuffle(solution.begin(), solution.end());
    return solution;
}

void tabuSearchWorker(int threadID, int maxIterations, int tabuListSize, vector<int>& localBestSolution, int& localBestMakespan, const vector<vector<int>>& processingTimes) {
    int numJobs = processingTimes.size();
    
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        vector<int> currentSolution = generateRandomSolution(numJobs);

        vector<vector<int>> neighbors;
        for (int i = 0; i < numJobs - 1; ++i) {
            for (int j = i + 1; j < numJobs; ++j) {
                vector<int> neighbor = currentSolution;
                swap(neighbor[i], neighbor[j]);
                neighbors.push_back(neighbor);
            }
        }

        int minNeighborMakespan = numeric_limits<int>::max();
        vector<int> bestNeighbor;

        for (const vector<int>& neighbor : neighbors) {
            int neighborMakespan = calculateMakespan(neighbor, processingTimes);

            if (neighborMakespan < minNeighborMakespan) {
                minNeighborMakespan = neighborMakespan;
                bestNeighbor = neighbor;
            }
        }

        currentSolution = bestNeighbor;
        int currentMakespan = minNeighborMakespan;

        if (currentMakespan < localBestMakespan) {
            localBestSolution = currentSolution;
            localBestMakespan = currentMakespan;
        }
    }
}

// Function to load processing times from a JSON file
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

int main(int argc, char* argv[]) {
    srand(time(0));

    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <file_path.json> <num_threads> <max_iterations_per_thread> <tabu_list_size>" << endl;
        return 1;
    }

    string filePath = argv[1];
    vector<vector<int>> processingTimes = loadProcessingTimesFromJSON(filePath);

    int numJobs = processingTimes.size();
    int numMachines = (numJobs > 0) ? processingTimes[0].size() : 0;
    int numThreads = stoi(argv[2]);
    int maxIterationsPerThread = stoi(argv[3]);
    int tabuListSize = stoi(argv[4]);

    vector<thread> threads;
    vector<vector<int>> localBestSolutions(numThreads);
    vector<int> localBestMakespans(numThreads, numeric_limits<int>::max());

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(tabuSearchWorker, i, maxIterationsPerThread, tabuListSize, ref(localBestSolutions[i]), ref(localBestMakespans[i]), processingTimes);
    }

    for (auto& t : threads) {
        t.join();
    }

    int globalBestMakespan = numeric_limits<int>::max();
    vector<int> globalBestSolution;

    for (int i = 0; i < numThreads; ++i) {
        if (localBestMakespans[i] < globalBestMakespan) {
            globalBestMakespan = localBestMakespans[i];
            globalBestSolution = localBestSolutions[i];
        }
    }

    cout << "Global Best Solution - Optimal Permutation: ";
    for (int jobIndex : globalBestSolution) {
        cout << jobIndex << " ";
    }
    cout << endl;

    cout << "Global Best Solution - Optimal Makespan: " << globalBestMakespan << endl;

    return 0;
}
