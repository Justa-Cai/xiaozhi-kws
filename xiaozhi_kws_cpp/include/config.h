#ifndef XIAOZHI_KWS_CONFIG_H
#define XIAOZHI_KWS_CONFIG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>

namespace xiaozhi {

/**
 * @brief 配置类，用于保存系统的各种参数
 */
class Config {
public:
    /**
     * @brief 默认构造函数
     */
    Config() {
        // 设置默认值
        set_default_preprocessing_config();
        set_default_feature_config();
        set_default_model_config();
        set_default_postprocessing_config();
    }

    /**
     * @brief 从JSON文件加载配置
     * @param config_path 配置文件路径
     * @return 成功返回true，失败返回false
     */
    bool load_from_file(const std::string& config_path) {
        // 简化实现，实际项目中这里会解析配置文件
        // 这里仅为演示，返回true表示加载成功
        return true;
    }

    /**
     * @brief 设置预处理配置
     */
    void set_default_preprocessing_config() {
        sample_rate = 16000;
        normalize_audio = true;
        target_level = -25.0f;
        remove_dc = true;
        apply_preemphasis = true;
        preemphasis_coeff = 0.97f;
    }

    /**
     * @brief 设置特征提取配置
     */
    void set_default_feature_config() {
        frame_length = 400;
        frame_shift = 160;
        fft_size = 512;
        num_filters = 40;
        num_mfcc = 13;
        low_freq = 20.0f;
        high_freq = 8000.0f;
        use_energy = true;
        normalize_features = true;
        use_delta = true;
        use_delta2 = true;
    }

    /**
     * @brief 设置模型配置
     */
    void set_default_model_config() {
        model_path = "model.bin";
        feature_dim = 13;
        num_classes = 2;
        detection_threshold = 0.5f;
        keywords = {"小智小智"};
    }

    /**
     * @brief 设置后处理配置
     */
    void set_default_postprocessing_config() {
        smooth_window_size = 5;
        min_detection_interval = 3000;
        apply_vad = true;
        vad_threshold = 0.01f;
        vad_window_size = 10;
        verbose = false;
        save_features = false;
        features_output_dir = "features";
    }

    /**
     * @brief 验证配置有效性
     * @return 配置有效返回true，否则返回false
     */
    bool validate() const {
        // 简化实现，实际项目中这里会进行配置验证
        return true;
    }

    // 音频预处理参数
    int sample_rate;            // 采样率
    bool normalize_audio;       // 是否归一化音频
    float target_level;         // 目标音量级别 (dB)
    bool remove_dc;             // 是否移除直流分量
    bool apply_preemphasis;     // 是否应用预加重
    float preemphasis_coeff;    // 预加重系数

    // 特征提取参数
    int frame_length;           // 帧长度（样本数）
    int frame_shift;            // 帧移（样本数）
    int fft_size;               // FFT大小
    int num_filters;            // 梅尔滤波器数量
    int num_mfcc;               // MFCC系数数量
    float low_freq;             // 最低频率 (Hz)
    float high_freq;            // 最高频率 (Hz)
    bool use_energy;            // 是否使用能量
    bool normalize_features;    // 是否归一化特征
    bool use_delta;             // 是否使用一阶差分
    bool use_delta2;            // 是否使用二阶差分

    // 模型参数
    std::string model_path;     // 模型文件路径
    int feature_dim;            // 特征维度
    int num_classes;            // 类别数量
    float detection_threshold;  // 检测阈值
    std::vector<std::string> keywords;  // 关键词列表

    // 后处理参数
    int smooth_window_size;     // 平滑窗口大小
    int min_detection_interval; // 最小检测间隔 (ms)
    bool apply_vad;             // 是否应用语音活动检测
    float vad_threshold;        // VAD阈值
    int vad_window_size;        // VAD窗口大小 (样本数)
    
    // 调试选项
    bool verbose;               // 详细输出
    bool save_features;         // 保存特征到文件
    std::string features_output_dir; // 特征输出目录
};

} // namespace xiaozhi

#endif // XIAOZHI_KWS_CONFIG_H 