//
// Created by aidan on 11/12/23.
//

#include <iostream>

#include "version.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <cxxopts.hpp>
#include <vector>

#include "Colour_Space_Transformation.h"
#include "Tint_Transformation.h"
#include "version.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    cxxopts::Options options("BARCO Assignment, Version:" + std::to_string(barco_VERSION_MAJOR) + " - " + std::to_string(barco_VERSION_MINOR));

    options.add_options()
            ("h,help", "Print options")
            ("s,source", "Path of the source image to apply the transformation on", cxxopts::value<fs::path>()->default_value("./images/golden_retriever_puppy.png"))
            ("d,destination", "The desired output path", cxxopts::value<fs::path>()->default_value("./output.png"))
            ("t,tint", "Tint to apply to the image", cxxopts::value<std::vector<uint8_t>>())
            ("w,weight", "Tint to apply to the image", cxxopts::value<float>()->default_value("0.5"))
            ("c,colour", "Should transform the image from 709 to 2020", cxxopts::value<bool>());

    auto result = options.parse(argc, argv);

    // If help request was passed
    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if ((result.count("tint") > 0) & (result.count("colour") > 0)) {
        std::cout << "Cannot apply both a tint and colour space conversion, please pick one!" << std::endl;
        return 1;
    }

    if ((result.count("tint") == 0) & (result.count("colour") == 0)) {
        std::cout << "Transformation not selected, please pick one!" << std::endl;
        return 1;
    }

    // ************** IMAGE LOADING **************

    fs::path src_path = result["source"].as<fs::path>().string();
    fs::path dst_path = result["destination"].as<fs::path>().string();

    cv::Mat img = cv::imread(src_path);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << src_path << std::endl;
        return 1;
    }

    // ************** TRANSFORMATIONS **************


    if (result.count("tint") > 0) {
        std::vector<uint8_t> tints = result["tint"].as<std::vector<uint8_t>>();
        float weight = result["weight"].as<float>();

        if (tints.size() != 3) {
            std::cout << "Tint values must have size 3"<< std::endl;
            return 1;
        }

        Tint_Transformation tint_transform(img.rows, img.cols, (uint8_t*) tints.data());

        tint_transform.tint_transformation(img.data, img.step, weight);

    } else {
        // Define the transformation
        Colour_Space_Transformation colour_space_transform(img.rows, img.cols);

        // Execute the transformation in place
        colour_space_transform.r709_to_r2020(img.data, img.step);
    }

    cv::imwrite(dst_path, img);

    return 0;
};
