/**
 * @file feature_extractor.h
 * @brief 音频特征提取器
 */

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include <cstdint>
#include "config_parser.h"

/**
 * 特征提取器类
 */
class FeatureExtractor {
public:
    /**
     * 构造函数
     * 
     * @param config 特征提取配置
     */
    FeatureExtractor(const Config& config);
    
    /**
     * 构造函数，从FeatureConfig创建
     * 
     * @param feature_config 特征配置
     */
    FeatureExtractor(const FeatureConfig& feature_config);
    
    /**
     * 析构函数
     */
    ~FeatureExtractor();

    /**
     * 从PCM音频数据提取特征
     * 
     * @param audio PCM音频数据
     * @param audio_len 音频数据长度（样本数）
     * @param normalize 是否归一化
     * @return 提取的特征，每一行代表一帧
     */
    std::vector<std::vector<float>> extract_features(
        const int16_t* audio, 
        size_t audio_len, 
        bool normalize = true);
    
    /**
     * 从滑动窗口提取特征
     * 
     * @param audio PCM音频数据
     * @param audio_len 音频数据长度（样本数）
     * @param window_size_ms 窗口大小（毫秒）
     * @param stride_ms 窗口步长（毫秒）
     * @return 窗口化的特征列表
     */
    std::vector<std::vector<std::vector<float>>> extract_features_sliding_window(
        const int16_t* audio, 
        size_t audio_len, 
        int window_size_ms = 1000, 
        int stride_ms = 500);

    /**
     * 计算MFCC特征
     * 
     * @param audio 音频数据
     * @param audio_len 音频数据长度
     * @return MFCC特征
     */
    std::vector<std::vector<float>> compute_mfcc(
        const float* audio, 
        size_t audio_len);
    
    /**
     * 计算delta特征
     * 
     * @param features 特征数据
     * @param width delta窗口宽度
     * @return delta特征
     */
    std::vector<std::vector<float>> compute_delta(
        const std::vector<std::vector<float>>& features, 
        int width = 3);
    
    /**
     * 获取特征维度
     * 
     * @return 特征维度
     */
    int get_feature_dim() const { return feature_dim_; }
    
    /**
     * 将短音频填充到足够的长度
     * 
     * @param audio 音频数据
     * @param audio_len 音频数据长度
     * @return 填充后的音频
     */
    std::vector<float> pad_audio(const float* audio, size_t audio_len);

private:
    // 配置
    Config config_;
    
    // FFT计算相关
    void* fft_plan_;
    int fft_size_;
    
    // 预计算数据
    std::vector<std::vector<float>> filterbank_;
    std::vector<std::vector<float>> dct_matrix_;
    std::vector<float> hamming_window_;
    
    // 窗口函数
    float* window_func_;
    
    // Mel滤波器组
    float* mel_filters_;
    
    // 特征维度
    int feature_dim_;
    
    // 初始化函数
    void validate_config();
    void init_fft();
    void init_mel_filterbank();
    void init_dct_matrix();
    void init_window_function();
    void init_mel_filters();
    
    // 从FeatureConfig转换为Config
    Config create_config_from_feature_config(const FeatureConfig& feature_config);
};

#endif /* FEATURE_EXTRACTOR_H */ 