/**
 * @file config_parser.cpp
 * @brief 配置解析器实现
 */

#include "config_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 简单YAML解析函数
void parse_yaml_line(const std::string& line, 
                    std::unordered_map<std::string, std::string>& config_map,
                    std::string& current_section) {
    
    // 跳过空行和注释
    if (line.empty() || line[0] == '#') {
        return;
    }
    
    // 检查缩进
    size_t indent = line.find_first_not_of(" \t");
    if (indent == std::string::npos) {
        return;
    }
    
    // 去除前后空白
    std::string trimmed = line.substr(indent);
    
    // 检查是否是节点（无冒号）
    if (trimmed.find(':') == std::string::npos) {
        return;
    }
    
    // 分割键和值
    size_t colon_pos = trimmed.find(':');
    std::string key = trimmed.substr(0, colon_pos);
    std::string value = (colon_pos + 1 < trimmed.size()) ? 
                       trimmed.substr(colon_pos + 1) : "";
    
    // 去除键和值的前后空白
    key = std::regex_replace(key, std::regex("^\\s+|\\s+$"), "");
    value = std::regex_replace(value, std::regex("^\\s+|\\s+$"), "");
    
    // 检查是否是节点（有值但无引号）
    if (indent == 0 && value.empty()) {
        current_section = key;
        return;
    }
    
    // 添加到配置映射
    std::string full_key = current_section.empty() ? key : current_section + "." + key;
    config_map[full_key] = value;
}

ConfigParser::ConfigParser(const std::string& config_path) {
    // 读取配置文件
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    // 解析YAML
    parse_yaml(content);
}

void ConfigParser::parse_yaml(const std::string& content) {
    std::istringstream stream(content);
    std::string line;
    std::string current_section;
    
    while (std::getline(stream, line)) {
        parse_yaml_line(line, config_map_, current_section);
    }
}

FeatureConfig ConfigParser::get_feature_config() const {
    FeatureConfig config;
    
    config.sample_rate = get_int("feature.sample_rate", 16000);
    config.window_size_ms = get_int("feature.window_size_ms", 30);
    config.window_stride_ms = get_int("feature.window_stride_ms", 10);
    config.n_mfcc = get_int("feature.n_mfcc", 40);
    config.n_fft = get_int("feature.n_fft", 512);
    config.n_mels = get_int("feature.n_mels", 80);
    config.use_delta = get_bool("feature.use_delta", true);
    config.use_delta2 = get_bool("feature.use_delta2", false);
    
    return config;
}

ModelConfig ConfigParser::get_model_config() const {
    ModelConfig config;
    
    config.type = get_string("model.type", "cnn_gru");
    config.input_dim = get_int("model.input_dim", 80);
    config.hidden_dim = get_int("model.hidden_dim", 64);
    config.num_layers = get_int("model.num_layers", 2);
    config.dropout = get_float("model.dropout", 0.1f);
    
    return config;
}

InferenceConfig ConfigParser::get_inference_config() const {
    InferenceConfig config;
    
    config.detection_threshold = get_float("inference.detection_threshold", 0.5f);
    config.smoothing_window = get_int("inference.smoothing_window", 10);
    
    return config;
}

std::vector<std::string> ConfigParser::get_keywords() const {
    std::vector<std::string> keywords;
    
    // 假设关键词在配置中的格式是"data.keywords.0", "data.keywords.1", ...
    for (int i = 0; ; i++) {
        std::string key = "data.keywords." + std::to_string(i);
        auto it = config_map_.find(key);
        if (it == config_map_.end()) {
            break;
        }
        keywords.push_back(it->second);
    }
    
    // 如果未找到任何关键词，尝试查找单个关键词配置
    if (keywords.empty()) {
        auto it = config_map_.find("data.keywords");
        if (it != config_map_.end() && !it->second.empty()) {
            keywords.push_back(it->second);
        }
    }
    
    return keywords;
}

std::string ConfigParser::get_string(const std::string& key, const std::string& default_value) const {
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        return it->second;
    }
    return default_value;
}

int ConfigParser::get_int(const std::string& key, int default_value) const {
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        try {
            return std::stoi(it->second);
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse int value for key: " << key << std::endl;
        }
    }
    return default_value;
}

float ConfigParser::get_float(const std::string& key, float default_value) const {
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        try {
            return std::stof(it->second);
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse float value for key: " << key << std::endl;
        }
    }
    return default_value;
}

bool ConfigParser::get_bool(const std::string& key, bool default_value) const {
    auto it = config_map_.find(key);
    if (it != config_map_.end()) {
        std::string value = it->second;
        // 转换为小写
        std::transform(value.begin(), value.end(), value.begin(), 
                      [](unsigned char c) { return std::tolower(c); });
        
        if (value == "true" || value == "yes" || value == "1") {
            return true;
        } else if (value == "false" || value == "no" || value == "0") {
            return false;
        }
    }
    return default_value;
}

Config ConfigParser::parse_config(const std::string& filepath) {
    // 读取文件内容
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开配置文件: " + filepath);
    }
    
    // 解析JSON
    json config_json;
    try {
        file >> config_json;
    } catch (const std::exception& e) {
        throw std::runtime_error("JSON解析错误: " + std::string(e.what()));
    }
    
    // 获取默认配置
    Config config = get_default_config();
    
    // 处理sample_rate
    if (config_json.contains("sample_rate")) {
        config.sample_rate = config_json["sample_rate"];
    }
    
    // 处理frame_length
    if (config_json.contains("frame_length")) {
        config.frame_length = config_json["frame_length"];
    } else if (config_json.contains("window_size_ms")) {
        // 从窗口大小（毫秒）计算帧长度（样本数）
        config.frame_length = static_cast<int>(config.sample_rate * config_json["window_size_ms"].get<double>() / 1000.0);
    }
    
    // 处理frame_shift
    if (config_json.contains("frame_shift")) {
        config.frame_shift = config_json["frame_shift"];
    } else if (config_json.contains("window_stride_ms")) {
        // 从窗口步长（毫秒）计算帧移（样本数）
        config.frame_shift = static_cast<int>(config.sample_rate * config_json["window_stride_ms"].get<double>() / 1000.0);
    }
    
    // 处理FFT大小
    if (config_json.contains("fft_size")) {
        config.fft_size = config_json["fft_size"];
    } else if (config_json.contains("n_fft")) {
        config.fft_size = config_json["n_fft"];
    }
    
    // 处理Mel滤波器数量
    if (config_json.contains("num_filters")) {
        config.num_filters = config_json["num_filters"];
    } else if (config_json.contains("n_mels")) {
        config.num_filters = config_json["n_mels"];
    }
    
    // 处理MFCC系数数量
    if (config_json.contains("n_mfcc")) {
        config.num_mfcc = config_json["n_mfcc"];
    } else if (config_json.contains("num_mfcc")) {
        config.num_mfcc = config_json["num_mfcc"];
    }
    
    // 处理频率范围
    if (config_json.contains("low_freq")) {
        config.low_freq = config_json["low_freq"];
    }
    if (config_json.contains("high_freq")) {
        config.high_freq = config_json["high_freq"];
    } else {
        // 默认设置为采样率的一半（奈奎斯特频率）
        config.high_freq = config.sample_rate / 2.0;
    }
    
    // 处理差分特征
    if (config_json.contains("use_delta")) {
        config.use_delta = config_json["use_delta"];
    }
    if (config_json.contains("use_delta2")) {
        config.use_delta2 = config_json["use_delta2"];
    }
    
    // 预加重和抖动
    if (config_json.contains("pre_emphasis")) {
        config.pre_emphasis = config_json["pre_emphasis"];
    }
    if (config_json.contains("dither")) {
        config.dither = config_json["dither"];
    }
    
    return config;
}

Config ConfigParser::parse_config_from_string(const std::string& json_str) {
    // 解析JSON
    json config_json;
    try {
        config_json = json::parse(json_str);
    } catch (const std::exception& e) {
        throw std::runtime_error("JSON解析错误: " + std::string(e.what()));
    }
    
    // 获取默认配置
    Config config = get_default_config();
    
    // TODO: 与上面的函数逻辑相同，可以提取一个共用的函数
    
    return config;
}

Config ConfigParser::get_default_config() {
    Config config;
    
    // 默认配置值
    config.sample_rate = 16000;
    config.frame_length = 400;       // 对应25ms窗口
    config.frame_shift = 160;        // 对应10ms步长
    config.fft_size = 512;
    config.num_filters = 40;
    config.num_mfcc = 13;
    config.low_freq = 0;
    config.high_freq = 8000;         // 默认奈奎斯特频率
    config.use_delta = true;
    config.use_delta2 = true;
    config.pre_emphasis = 0.97;      // 典型预加重系数
    config.dither = 0.0;             // 默认不使用抖动
    
    return config;
} 