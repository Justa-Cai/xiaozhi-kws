/**
 * @file detector.h
 * @brief 唤醒词检测器
 */

#ifndef DETECTOR_H
#define DETECTOR_H

#include <memory>
#include <string>
#include <deque>
#include <functional>
#include <chrono>
#include "feature_extractor.h"
#include "model.h"
#include "config_parser.h"
#include "post_processor.h"
#include <onnxruntime/onnxruntime_cxx_api.h>

/**
 * 唤醒回调函数类型
 */
using DetectionCallback = std::function<void(float)>;

/**
 * 唤醒词检测器类
 */
class Detector {
public:
    /**
     * 构造函数
     * 
     * @param model_path 模型文件路径
     * @param config_path 配置文件路径
     * @param threshold 检测阈值，若为0则使用配置文件中的阈值
     */
    Detector(const std::string& model_path, 
             const std::string& config_path, 
             float threshold = 0.0f);
    
    /**
     * 析构函数
     */
    ~Detector();
    
    /**
     * 设置唤醒回调函数
     * 
     * @param callback 回调函数
     */
    void set_callback(DetectionCallback callback);
    
    /**
     * 处理音频数据
     * 
     * @param audio_data 音频数据
     * @param audio_len 音频数据长度（样本数）
     * @return 是否检测到唤醒词
     */
    bool process_audio(const int16_t* audio_data, size_t audio_len);
    
    /**
     * 从音频文件加载并检测
     * 
     * @param audio_path 音频文件路径
     * @param confidence 输出参数，检测置信度
     * @param debug_mode 是否为调试模式
     * @return 是否检测到唤醒词
     */
    bool detect_file(const std::string& audio_path, float& confidence, bool debug_mode = false);
    
    /**
     * 重置检测器状态
     */
    void reset();
    
    /**
     * 获取检测阈值
     * 
     * @return 检测阈值
     */
    float get_threshold() const;
    
    /**
     * 设置检测阈值
     * 
     * @param threshold 检测阈值
     */
    void set_threshold(float threshold);
    
    /**
     * 获取后处理器实例
     * 
     * @return 后处理器的常量引用
     */
    const xiaozhi::PostProcessor& get_post_processor() const;

    /**
     * 对音频数据进行推理
     * 
     * @param audio_data 音频数据
     * @return 置信度得分
     */
    float infer(const std::vector<float>& audio_data);
    
    /**
     * 从音频文件中检测唤醒词
     * 
     * @param audio_path 音频文件路径
     * @param confidence 输出的置信度得分
     * @return 是否检测到唤醒词
     */
    bool detect_from_file(const std::string& audio_path, float& confidence);

    /**
     * 获取音频特征统计信息（调试用）
     * 
     * @param audio_path 音频文件路径
     * @return 特征信息字符串
     */
    std::string get_features_debug_info(const std::string& audio_path);

private:
    // 配置
    std::unique_ptr<ConfigParser> config_parser_;
    InferenceConfig inference_config_;
    FeatureConfig feature_config_; // 添加 FeatureConfig 成员变量
    
    // 模型和特征提取器
    std::unique_ptr<Model> model_;
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    
    // ONNX运行时相关
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    
    // 后处理器
    std::unique_ptr<xiaozhi::PostProcessor> post_processor_;
    
    // 回调函数
    DetectionCallback callback_;
    
    // 状态变量（保留以兼容旧代码，后续可以考虑移除）
    std::deque<float> recent_predictions_;
    std::chrono::time_point<std::chrono::steady_clock> last_trigger_time_;
    const std::chrono::seconds trigger_cooldown_{3}; // 触发冷却时间（秒）
    
    // 处理单个音频片段的函数
    bool process_audio_chunk(const std::vector<std::vector<float>>& features);
    
    // 从PCM数据加载音频
    std::vector<float> load_audio_from_pcm(const int16_t* audio_data, size_t audio_len);
    
    // 从文件加载音频
    std::vector<float> load_audio_from_file(const std::string& filepath, int& source_sample_rate);
    
    // 计算音频能量
    float calculate_audio_energy(const std::vector<float>& audio);
};

#endif /* DETECTOR_H */ 