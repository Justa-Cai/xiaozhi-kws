#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "用法: " << argv[0] << " <模型路径> <配置文件路径> <音频文件夹路径> [输出CSV文件]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string config_path = argv[2];
    std::string audio_dir = argv[3];
    std::string output_csv = (argc > 4) ? argv[4] : "detection_results.csv";
    
    // 打印参数
    std::cout << "模型路径: " << model_path << std::endl;
    std::cout << "配置文件: " << config_path << std::endl;
    std::cout << "音频文件夹: " << audio_dir << std::endl;
    std::cout << "输出CSV文件: " << output_csv << std::endl;
    
    try {
        // 创建检测器
        std::cout << "加载模型..." << std::endl;
        Detector detector(model_path, config_path);
        
        // 扫描音频文件
        std::cout << "扫描音频文件..." << std::endl;
        auto audio_files = scan_audio_files(audio_dir);
        
        if (audio_files.empty()) {
            std::cout << "未找到音频文件！" << std::endl;
            return 1;
        }
        
        std::cout << "找到 " << audio_files.size() << " 个音频文件" << std::endl;
        
        // 检测结果
        std::vector<bool> detection_results;
        std::vector<float> confidence_scores;
        
        // 开始检测
        std::cout << "开始检测唤醒词..." << std::endl;
        auto start_time = std::chrono::system_clock::now();
        
        for (size_t i = 0; i < audio_files.size(); ++i) {
            std::cout << "[" << (i+1) << "/" << audio_files.size() << "] 处理: " 
                     << fs::path(audio_files[i]).filename().string() << std::endl;
                     
            float confidence = 0.0f;
            bool detected = detector.detect_file(audio_files[i], confidence);
            
            detection_results.push_back(detected);
            confidence_scores.push_back(confidence);
            
            std::cout << "  结果: " << (detected ? "✓ 检测到唤醒词" : "✗ 未检测到唤醒词") 
                     << ", 置信度: " << confidence << std::endl;
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