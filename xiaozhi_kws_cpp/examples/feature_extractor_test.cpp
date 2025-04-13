/**
 * @file feature_extractor_test.cpp
 * @brief C++特征提取测试工具
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

#include "feature_extractor.h"
#include "config_parser.h"
#include "audio_utils.h"

using json = nlohmann::json;

/**
 * 将特征向量转换为JSON格式
 */
json features_to_json(const std::vector<std::vector<float>>& features) {
    json j = json::array();
    for (const auto& frame : features) {
        json frame_json = json::array();
        for (float value : frame) {
            frame_json.push_back(value);
        }
        j.push_back(frame_json);
    }
    return j;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --audio <audio_file> --config <config_file> [--output <output_file>]" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // 解析命令行参数
        std::string audio_path;
        std::string config_path;
        std::string output_path = "features_output.json";
        
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--audio" && i + 1 < argc) {
                audio_path = argv[++i];
            } else if (arg == "--config" && i + 1 < argc) {
                config_path = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                output_path = argv[++i];
            } else if (arg == "--help") {
                print_usage(argv[0]);
                return 0;
            }
        }
        
        if (audio_path.empty() || config_path.empty()) {
            std::cerr << "Error: Required arguments missing." << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        
        std::cout << "加载配置文件: " << config_path << std::endl;
        
        // 加载配置
        ConfigParser config_parser;
        Config config = config_parser.parse_config(config_path);
        
        std::cout << "初始化特征提取器..." << std::endl;
        
        // 创建特征提取器
        FeatureExtractor feature_extractor(config);
        
        std::cout << "加载音频文件: " << audio_path << std::endl;
        
        // 加载音频
        std::vector<int16_t> audio_data;
        size_t sample_rate;
        if (!load_audio_file(audio_path, audio_data, sample_rate)) {
            std::cerr << "无法加载音频文件: " << audio_path << std::endl;
            return 1;
        }
        
        if (sample_rate != config.sample_rate) {
            std::cout << "警告: 音频采样率 (" << sample_rate 
                     << ") 与配置不匹配 (" << config.sample_rate << ")" << std::endl;
            std::cout << "执行重采样..." << std::endl;
            
            // 重采样处理
            std::vector<int16_t> resampled_audio = resample_audio(audio_data, sample_rate, config.sample_rate);
            audio_data = resampled_audio;
        }
        
        std::cout << "提取特征..." << std::endl;
        
        // 提取特征
        std::vector<std::vector<float>> features = 
            feature_extractor.extract_features(audio_data.data(), audio_data.size(), true);
        
        std::cout << "特征提取完成: " << features.size() << " 帧, " 
                 << (features.empty() ? 0 : features[0].size()) << " 维" << std::endl;
        
        // 创建JSON输出
        json output_json;
        output_json["features"] = features_to_json(features);
        output_json["num_frames"] = features.size();
        output_json["feature_dim"] = features.empty() ? 0 : features[0].size();
        
        // 写入JSON文件
        std::ofstream out_file(output_path);
        if (!out_file.is_open()) {
            std::cerr << "无法创建输出文件: " << output_path << std::endl;
            return 1;
        }
        
        out_file << output_json.dump(4);
        out_file.close();
        
        std::cout << "结果已保存到: " << output_path << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
} 