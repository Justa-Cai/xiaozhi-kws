/**
 * @file kws_folder_inference.cpp
 * @brief 小智唤醒词文件夹推理示例
 * 
 * 本示例展示如何使用小智唤醒词SDK对文件夹中的音频文件进行批量推理
 */

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>

// 引入小智唤醒词SDK头文件
#include "xiaozhi_kws.h"
#include "detector.h"
#include "post_processor.h"
#include "config.h"

namespace fs = std::filesystem;

// 检查文件扩展名是否为音频文件
bool is_audio_file(const std::string& filename) {
    std::string ext = fs::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".wav" || ext == ".mp3";
}

// 递归扫描文件夹
std::vector<std::string> scan_audio_files(const std::string& dir_path) {
    std::vector<std::string> audio_files;
    
    try {
        for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
            if (entry.is_regular_file() && is_audio_file(entry.path().string())) {
                audio_files.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "扫描文件夹时出错：" << e.what() << std::endl;
    }
    
    return audio_files;
}

// 格式化时间
std::string format_time(const std::chrono::system_clock::time_point& time) {
    auto time_t = std::chrono::system_clock::to_time_t(time);
    std::tm tm_buf;
    localtime_r(&time_t, &tm_buf);
    
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return std::string(buffer);
}

// 保存检测结果到CSV文件
void save_results_to_csv(const std::string& output_path, 
                        const std::vector<std::string>& files,
                        const std::vector<bool>& detections,
                        const std::vector<float>& confidences) {
    std::ofstream out(output_path);
    if (!out.is_open()) {
        std::cerr << "无法创建输出文件：" << output_path << std::endl;
        return;
    }
    
    // 写入CSV头
    out << "文件路径,是否检测到唤醒词,置信度\n";
    
    // 写入数据
    for (size_t i = 0; i < files.size(); ++i) {
        out << files[i] << "," 
            << (detections[i] ? "是" : "否") << ","
            << confidences[i] << "\n";
    }
    
    out.close();
    std::cout << "结果已保存到：" << output_path << std::endl;
}

// 使用C API进行文件检测
bool detect_audio_file_c_api(const std::string& model_path,
                            const std::string& config_path,
                            const std::string& audio_path,
                            float threshold,
                            float& confidence) {
    XiaozhiKwsError err = xiaozhi_kws_detect_file(
        model_path.c_str(),
        config_path.c_str(),
        audio_path.c_str(),
        threshold,
        &confidence
    );
    
    if (err != XIAOZHI_KWS_SUCCESS) {
        std::cerr << "处理文件失败：" << audio_path << std::endl;
        return false;
    }
    
    // 默认通过C API的结果判断是否检测到唤醒词
    return confidence > threshold;
}

void print_usage(const char* program_name) {
    std::cout << "用法: " << program_name << " <模型路径> <配置文件路径> <音频文件夹路径> [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  --threshold <阈值>  检测阈值, 默认使用配置文件中的值" << std::endl;
    std::cout << "  --use-c-api         使用C API而非C++ API" << std::endl;
    std::cout << "  --output <文件路径>  结果输出CSV文件, 默认为'detection_results.csv'" << std::endl;
    std::cout << "  --debug             显示详细调试信息，用于与Python版本对比" << std::endl;
    std::cout << "  --help              显示帮助信息" << std::endl;
}

int main(int argc, char** argv) {
    // 解析命令行参数
    std::string model_path;
    std::string config_path;
    std::string audio_dir;
    std::string output_csv = "detection_results.csv";
    float threshold = 0.0f;
    bool use_c_api = false;
    bool debug_mode = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--threshold" && i + 1 < argc) {
            threshold = std::stof(argv[++i]);
        } else if (arg == "--use-c-api") {
            use_c_api = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_csv = argv[++i];
        } else if (arg == "--debug") {
            debug_mode = true;
        } else if (model_path.empty()) {
            model_path = arg;
        } else if (config_path.empty()) {
            config_path = arg;
        } else if (audio_dir.empty()) {
            audio_dir = arg;
        }
    }
    
    if (model_path.empty() || config_path.empty() || audio_dir.empty()) {
        std::cerr << "缺少必要参数！" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // 打印参数
    std::cout << "模型路径: " << model_path << std::endl;
    std::cout << "配置文件: " << config_path << std::endl;
    std::cout << "音频文件夹: " << audio_dir << std::endl;
    std::cout << "检测阈值: " << (threshold > 0.0f ? std::to_string(threshold) : "使用配置文件中的值") << std::endl;
    std::cout << "使用API: " << (use_c_api ? "C API" : "C++ API") << std::endl;
    std::cout << "输出CSV文件: " << output_csv << std::endl;
    std::cout << "调试模式: " << (debug_mode ? "开启" : "关闭") << std::endl;
    
    try {
        // 检查是文件还是目录
        bool is_directory = fs::is_directory(audio_dir);
        std::vector<std::string> audio_files;
        
        if (is_directory) {
            // 扫描音频文件
            std::cout << "扫描音频文件..." << std::endl;
            audio_files = scan_audio_files(audio_dir);
        } else if (is_audio_file(audio_dir)) {
            // 单个文件的情况
            std::cout << "处理单个音频文件..." << std::endl;
            audio_files.push_back(audio_dir);
        } else {
            std::cerr << "提供的路径不是有效的音频文件或目录！" << std::endl;
            return 1;
        }
        
        if (audio_files.empty()) {
            std::cout << "未找到音频文件！" << std::endl;
            return 1;
        }
        
        std::cout << "找到 " << audio_files.size() << " 个音频文件" << std::endl;
        
        // 检测结果
        std::vector<bool> detection_results;
        std::vector<float> confidence_scores;
        
        // 创建C++ API的检测器（如果需要）
        std::unique_ptr<Detector> detector;
        if (!use_c_api) {
            std::cout << "加载模型..." << std::endl;
            detector = std::make_unique<Detector>(model_path, config_path, threshold);
        }
        
        // 开始检测
        std::cout << "开始检测唤醒词..." << std::endl;
        auto start_time = std::chrono::system_clock::now();
        
        for (size_t i = 0; i < audio_files.size(); ++i) {
            std::cout << "[" << (i+1) << "/" << audio_files.size() << "] 处理: " 
                     << fs::path(audio_files[i]).filename().string() << std::endl;
            
            float confidence = 0.0f;
            bool detected = false;
            
            if (use_c_api) {
                // 使用C API
                detected = detect_audio_file_c_api(
                    model_path, config_path, audio_files[i], threshold, confidence);
            } else {
                // 调试模式 - 只有在调试模式下才添加详细输出
                if (debug_mode) {
                    // C++ API的debug_file_detection会添加更详细的输出
                    std::cout << "\n===== 开始Python/C++对比调试 =====" << std::endl;
                    std::cout << "文件: " << audio_files[i] << std::endl;
                    detected = detector->detect_file(audio_files[i], confidence);
                    std::cout << "===== 结束Python/C++对比调试 =====\n" << std::endl;
                } else {
                    // 正常检测流程
                    detected = detector->detect_file(audio_files[i], confidence);
                }
            }
            
            detection_results.push_back(detected);
            confidence_scores.push_back(confidence);
            
            std::cout << "  结果: " << (detected ? "✓ 检测到唤醒词" : "✗ 未检测到唤醒词") 
                     << ", 置信度: " << std::fixed << std::setprecision(4) << confidence << std::endl;
        }
        
        auto end_time = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        
        // 打印统计信息
        int detected_count = std::count(detection_results.begin(), detection_results.end(), true);
        std::cout << "\n检测完成！" << std::endl;
        std::cout << "总文件数: " << audio_files.size() << std::endl;
        std::cout << "检测到唤醒词: " << detected_count << " 个文件" << std::endl;
        std::cout << "未检测到唤醒词: " << (audio_files.size() - detected_count) << " 个文件" << std::endl;
        std::cout << "处理时间: " << duration << " 秒" << std::endl;
        
        // 保存结果
        save_results_to_csv(output_csv, audio_files, detection_results, confidence_scores);
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 