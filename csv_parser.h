#ifndef __CSV_PARSER
#define __CSV_PARSER

#include <fstream>
#include <iostream>
#include <string>

#include "Eigen/Dense"

void split(const string &, std::vector<int> &);

void loadFromCSV(const char *filepath, Eigen::MatrixXd &input, Eigen::MatrixXd &output) {
    std::string line;
    std::ifstream csvFile(filepath);

    std::vector<int> buffer;

    int row{0};
    buffer.resize(784 + 1); // label and data
    if (csvFile.is_open()) {
        getline(csvFile, line);
        while (getline(csvFile, line)) {
            split(line, buffer);
            for (int i{1}; i < 785; ++i) {
                input(row, i - 1) = buffer[i] / 255.0;
            }
            for (int i{0}; i < 10; ++i)
                output(row, i) = static_cast<int>(buffer[0] == i);
            row++;
        }
    } else
        std::cout << "CSVParser :: Unable to open file" << std::endl;
}

void loadTestFromCSV(const char *filepath, Eigen::MatrixXd &input) {
    std::string line;
    std::ifstream csvFile(filepath);

    std::vector<int> buffer;

    int row{0};
    buffer.resize(784); // data
    if (csvFile.is_open()) {
        getline(csvFile, line);
        while (getline(csvFile, line)) {
            split(line, buffer);
            for (int i{0}; i < 784; ++i) {
                input(row, i) = buffer[i] / 255.0;
            }
            row++;
        }
    } else
        std::cout << "CSVParser :: Unable to open file" << std::endl;
}

void split(const string &s, std::vector<int> &out) {
    if (out.size() == 0)
        throw std::runtime_error("CSVParser :: receiving vector probably hasn't been resized!");
    int curr{}, pos{};

    for (size_t i{0}; i < s.size(); i++) {
        if (s[i] == ',') {
            out.at(pos++) = curr;
            curr = 0;
        } else {
            curr *= 10;
            curr += s[i] - '0';
        }
    }

    out.at(pos) = curr;
}

#endif