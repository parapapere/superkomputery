#include <iostream>
#include <fstream>
#include <json/json.h>

using namespace std;

int main() {
    ifstream file("your_input_file.json");
    if (!file.is_open()) {
        cerr << "Unable to open file: your_input_file.json" << endl;
        return 1;
    }

    Json::Value root;
    file >> root;
    file.close();

    // Now you can use 'root' to access your JSON data
    // For example, if your JSON has an array of arrays, you can iterate through them
    for (const auto& row : root) {
        for (const auto& value : row) {
            cout << value.asInt() << " ";
        }
        cout << endl;
    }

    return 0;
}
