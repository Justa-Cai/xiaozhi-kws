#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include "post_processor.h"
#include "config.h"

// 打印检测结果
void print_detection_result(const xiaozhi::DetectionResult& result) {
    std::cout << "检测结果：" << (result.is_detected ? "检测到唤醒词" : "未检测到唤醒词") << std::endl;
    std::cout << "原始置信度：" << result.confidence << std::endl;
    std::cout << "平滑后置信度：" << result.smoothed_confidence << std::endl;
    std::cout << "---------------------------" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "后处理器测试程序" << std::endl;
    std::cout << "==========================" << std::endl;
    
    // 创建配置
    xiaozhi::Config config;
    
    // 设置默认配置
    config.detection_threshold = 0.5f;
    config.smooth_window_size = 5;
    config.min_detection_interval = 1000; // 毫秒
    config.apply_vad = true;
    config.vad_threshold = 0.01f;
    config.vad_window_size = 10;
    
    // 创建后处理器
    xiaozhi::PostProcessor post_processor(config);
    
    // 测试1：连续的低置信度
    std::cout << "测试1：连续的低置信度" << std::endl;
    std::vector<float> test1 = {0.1f, 0.2f, 0.15f, 0.18f, 0.22f};
    for (float conf : test1) {
        auto result = post_processor.process(conf);
        print_detection_result(result);
    }
    
    // 测试2：单个高置信度
    std::cout << "测试2：单个高置信度" << std::endl;
    auto result = post_processor.process(0.8f);
    print_detection_result(result);
    
    // 测试3：连续的高置信度
    std::cout << "测试3：连续的高置信度" << std::endl;
    std::vector<float> test3 = {0.6f, 0.7f, 0.65f, 0.68f, 0.72f};
    for (float conf : test3) {
        auto result = post_processor.process(conf);
        print_detection_result(result);
    }
    
    // 测试4：冷却时间测试
    std::cout << "测试4：冷却时间测试" << std::endl;
    std::cout << "第一次高置信度检测：" << std::endl;
    result = post_processor.process(0.9f);
    print_detection_result(result);
    
    std::cout << "立即再次尝试检测（应被冷却时间阻止）：" << std::endl;
    result = post_processor.process(0.9f);
    print_detection_result(result);
    
    std::cout << "等待500毫秒后再次尝试（应仍被冷却时间阻止）：" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    result = post_processor.process(0.9f);
    print_detection_result(result);
    
    std::cout << "等待1100毫秒后再次尝试（应已超过冷却时间）：" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    result = post_processor.process(0.9f);
    print_detection_result(result);
    
    // 测试5：VAD测试
    std::cout << "测试5：VAD测试" << std::endl;
    std::cout << "低能量测试（能量小于阈值）：" << std::endl;
    bool is_speech = post_processor.apply_vad(0.005f);
    std::cout << "VAD检测结果：" << (is_speech ? "有语音" : "无语音") << std::endl;
    
    std::cout << "高能量测试（能量大于阈值）：" << std::endl;
    is_speech = post_processor.apply_vad(0.02f);
    std::cout << "VAD检测结果：" << (is_speech ? "有语音" : "无语音") << std::endl;
    
    // 测试6：批处理测试
    std::cout << "测试6：批处理测试" << std::endl;
    std::vector<float> test6 = {0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    auto results = post_processor.process_batch(test6);
    
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "批处理结果 #" << (i+1) << ": " << std::endl;
        print_detection_result(results[i]);
    }
    
    // 测试7：重置测试
    std::cout << "测试7：重置测试" << std::endl;
    std::cout << "重置前：" << std::endl;
    result = post_processor.process(0.9f);
    print_detection_result(result);
    
    std::cout << "重置后：" << std::endl;
    post_processor.reset();
    result = post_processor.process(0.9f);
    print_detection_result(result);
    
    std::cout << "测试完成" << std::endl;
    
    return 0;
} 