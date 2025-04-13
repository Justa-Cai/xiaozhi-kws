/**
 * @file mfcc_features.h
 * @brief MFCC特征提取器
 */

#ifndef MFCC_FEATURES_H
#define MFCC_FEATURES_H

#include <vector>
#include <cstdint>
#include <memory>

/**
 * MFCC特征提取器类
 */
class MFCCFeatures {
public:
    /**
     * 构造函数
     */
    MFCCFeatures();
    
    /**
     * 析构函数
     */
    ~MFCCFeatures();
    
    /**
     * 初始化MFCC提取器
     * 
     * @param sample_rate 采样率
     * @param n_mfcc MFCC特征数量
     * @param n_mels Mel滤波器组数量
     * @param window_size_ms 窗口大小（毫秒）
     * @param window_stride_ms 窗口步长（毫秒）
     */
    void initialize(int sample_rate, int n_mfcc, int n_mels, int window_size_ms, int window_stride_ms);
    
    /**
     * 计算MFCC特征
     * 
     * @param audio 音频数据
     * @return MFCC特征向量
     */
    std::vector<float> compute_mfcc(const std::vector<float>& audio);

private:
    // 配置参数
    int sample_rate_;
    int n_mfcc_;
    int n_mels_;
    int window_size_;
    int window_stride_;
    int fft_size_;
    
    // 是否已初始化
    bool is_initialized_;
    
    // 预计算数据
    std::vector<std::vector<float>> mel_filterbank_;
    std::vector<std::vector<float>> dct_matrix_;
    std::vector<float> window_function_;
    
    // 内部函数
    void init_mel_filterbank();
    void init_dct_matrix();
    void init_window_function();
    
    // 辅助函数
    double hz_to_mel(double hz);
    double mel_to_hz(double mel);
};

#endif /* MFCC_FEATURES_H */ 