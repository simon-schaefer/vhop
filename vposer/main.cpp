#include <iostream>
#include "include/VPoser.h"
#include "include/LeakyRelu.h"
#include "include/FileUtils.h"
#include "include/Transformations.h"
#include <fstream>


void testLayerImplementations(){
    // Linear Layer
    // Input [0.4454, 0.8607, 0.2194, 0.6060, 0.6443]
    // Output [ 0.0576, -0.2601,  0.7225,  0.4670, -0.0571,  0.3613, -0.6810, -0.9362, 0.7772, -0.0737]

    Eigen::MatrixXd input = Eigen::MatrixXd::Random(1,5);
    input(0,0) = 0.4454; input(0,1) =  0.8607; input(0,2) = 0.2194; input(0,3) =  0.6060; input(0,4) =  0.6443;
    Linear l1 = Linear(5, 10);
    std::string weightPath = "C:\\Users\\Burak\\Desktop\\Burak\\Projects\\vhop\\data\\test_weights\\linear";
    l1.loadParams(weightPath);
    Eigen::MatrixXd output = l1.forward(input);
    std::cout << " Output for the linear layer: " << std::endl;
    for(int i = 0 ; i < 10 ; i++) std::cout << output(0,i) << "\t";
    std::cout << std::endl;

    // Linear + Leaky Relu
    // Input [-0.24, -24.2, 0.23]
    // Output [ 0.4959, -0.0185, -0.1139, 10.8231, -0.1283]

    input = Eigen::MatrixXd::Random(1,3);
    input(0,0) = -0.24; input(0,1) =  -24.2; input(0,2) =0.23;
    l1 = Linear(3, 5);
    weightPath = "C:\\Users\\Burak\\Desktop\\Burak\\Projects\\vhop\\data\\test_weights\\leaky";
    l1.loadParams(weightPath);
    output = l1.forward(input);

    LeakyRelu leakyRelu = LeakyRelu(0.1);
    Eigen::MatrixXd lout = leakyRelu.forward(output);

    std::cout << " Output for the leaky relu: " << std::endl;
    for(int i = 0 ; i < 5 ; i++) std::cout << lout(0,i) << "\t";
    std::cout << std::endl;
}

int main() {


    std::string mainFilePath = "C:\\Users\\Burak\\Desktop\\Burak\\Projects\\vhop\\data\\weights\\";

    VPoser vposer = VPoser(mainFilePath, 512, 32);
    vposer.printModel();
    vposer.loadParams();


    Eigen::MatrixXd decoder_sampled_input = vhop::utility::loadDoubleMatrix("C:\\Users\\Burak\\Exercises\\FinalProject\\vhop\\vposer\\data\\decoder_testing\\sampled.txt", 500, 32);
    DecoderOut dout = vposer.decode(decoder_sampled_input);


    Eigen::MatrixXd sample_input = vhop::utility::loadDoubleMatrix("C:\\Users\\Burak\\Exercises\\FinalProject\\vhop\\vposer\\data\\sample_input\\amass_body_input.txt", 500, 63);
    VposerOut modelOut = vposer.forward(sample_input);

    return 0;
}