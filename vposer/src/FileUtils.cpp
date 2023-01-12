//
// Created by Burak on 9.01.2023.
//


#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include "../include/FileUtils.h"

Eigen::VectorXd vhop::utility::loadVector(const std::string weightFilePath, int row){
    std::ifstream fin(weightFilePath);

    if(!fin){
        std::cout << "Error: unable to read txt file: " << weightFilePath << std::endl;
    }

    int r = 0; int c = 0;
    std::string line;

    // Read first line which contains dimensions of the matrix
    std::getline(fin, line);
    size_t pos;

    int vector_dim = stoi(line);
    std::cout << "Reading a vector with " <<  vector_dim << " dimension" << std::endl;

    Eigen::MatrixXd out = Eigen::VectorXd::Zero(vector_dim);

    while (getline(fin, line)) {
        std::string token;
        while((pos = line.find('\t')) != std::string::npos){
            token = line.substr(0, pos);

            out(c) = stod(token);
            line.erase(0, pos + 1);
            c += 1;
            if(c== vector_dim) break;
        }
        if (line != ""){
            out(c) = stod(line);
        }
    }
    fin.close();

    if(out.rows() != row){
        throw std::invalid_argument( "loadded weight parameter dimension does not match with the weights" );
    }

    return out;
}

Eigen::MatrixXd vhop::utility::loadDoubleMatrix(const std::string weightFilePath, int row, int col) {

    std::ifstream fin(weightFilePath);

    if(!fin){
        std::cout << "Error: unable to read txt file: " << weightFilePath << std::endl;
    }

    int r = 0; int c = 0;
    std::string line;

    // Read first line which contains dimensions of the matrix
    std::getline(fin, line);
    size_t pos = line.find('\t');

    std::string str_dim1 = line.substr(0, pos);
    int dim1 = std::stoi(str_dim1);
    std::string str_dim2 = line.substr(pos+1);
    int dim2 = std::stoi(str_dim2);
    std::cout << "Reading a vector with the dimension(" <<  dim1 << ", "<< dim2 << ")" << std::endl;

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(dim1, dim2);

    while (getline(fin, line)) {
        std::string token;
        while((pos = line.find('\t')) != std::string::npos){
            token = line.substr(0, pos);

            out(r,c) = stod(token);
            line.erase(0, pos + 1);
            c++;
            if(c == dim2){
                break;
            }
        }
        if (line != ""){
            out(r,c) = stod(token);
        }
        r += 1;
        c = 0;
    }
    fin.close();

    if(out.rows() != row || out.cols() != col){
        throw std::invalid_argument( "loadded weight parameter dimension does not match with the weights" );
    }
    return out;
}
