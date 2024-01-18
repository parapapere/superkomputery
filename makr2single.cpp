#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <ctime>
#include <fstream>
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

// Function to generate a random solution
vector<int> generateRandomSolution(int numJobs) {
    vector<int> solution(numJobs);
    for (int i = 0; i < numJobs; ++i) {
        solution[i] = i;
    }
    random_shuffle(solution.begin(), solution.end());
    return solution;
}

// Function to perform tabu search
void tabuSearch(int maxIterations, int tabuListSize, vector<int>& bestSolution, int& bestMakespan, const vector<vector<int>>& processingTimes) {
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

        if (currentMakespan < bestMakespan) {
            bestSolution = currentSolution;
            bestMakespan = currentMakespan;
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

    // Load processing times from a JSON file dynamically
    vector<vector<int>> processingTimes = loadProcessingTimesFromJSON("your_input_file.json");

    int numJobs = processingTimes.size();
    int numMachines = (numJobs > 0) ? processingTimes[0].size() : 0;

    // Default values
    int nIterations = 10;
    int maxIterations = 100;
    int tabuListSize = 5;

    if (argc >= 4) {
        // If there are enough command-line arguments, use them
        nIterations = atoi(argv[1]);
        maxIterations = atoi(argv[2]);
        tabuListSize = atoi(argv[3]);
    }

    vector<int> bestSolution;
    int bestMakespan = numeric_limits<int>::max();

    for (int i = 0; i < nIterations; ++i) {
        tabuSearch(maxIterations, tabuListSize, bestSolution, bestMakespan, processingTimes);

        cout << "Iteration " << i + 1 << " - Best Solution - Optimal Permutation: ";
        for (int jobIndex : bestSolution) {
            cout << jobIndex << " ";
        }
        cout << endl;

        cout << "Iteration " << i + 1 << " - Best Solution - Optimal Makespan: " << bestMakespan << endl;

        // Reset best solution for the next iteration
        bestSolution.clear();
        bestMakespan = numeric_limits<int>::max();
    }

    return 0;
}
