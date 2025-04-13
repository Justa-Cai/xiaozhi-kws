/**
 * @file feature_extractor.h
 * @brief 音频特征提取器
 */

#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <vector>
#include <cstdint>
#include "config_parser.h"
#include <complex>

// FFTW 前向声明
struct fftwf_plan_s;
typedef struct fftwf_plan_s* fftwf_plan;

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
    // FeatureExtractor(const Config& config);
    // 修改为使用 FeatureConfig
    explicit FeatureExtractor(const FeatureConfig& feature_config);

    /**
     * 构造函数，从FeatureConfig创建 (移除旧的重载)
     *
     * @param feature_config 特征配置
     */
    // FeatureExtractor(const FeatureConfig& feature_config);

    /**
     * 析构函数
     */
    ~FeatureExtractor();

    /**
     * 从PCM音频数据提取特征
     * 
     * @param audio PCM音频数据 (int16_t)
     * @param audio_len 音频数据长度（样本数）
     * @param normalize 是否归一化 (通常 MFCC 不需要额外归一化)
     * @return 提取的特征 (MFCC)，每一行代表一帧
     */
    std::vector<std::vector<float>> extract_features(
        const int16_t* audio,
        size_t audio_len,
        bool normalize = false); // 默认不进行额外归一化

    /**
     * 从滑动窗口提取特征 (如果需要保留)
     *
     * @param audio PCM音频数据
     * @param audio_len 音频数据长度（样本数）
     * @param window_size_ms 窗口大小（毫秒）
     * @param stride_ms 窗口步长（毫秒）
     * @return 窗口化的特征列表
     */
    // std::vector<std::vector<std::vector<float>>> extract_features_sliding_window(
    //     const int16_t* audio,
    //     size_t audio_len,
    //     int window_size_ms = 1000,
    //     int stride_ms = 500);

    /**
     * 计算MFCC特征 (变为私有方法)
     *
     * @param audio 音频数据 (float)
     * @param audio_len 音频数据长度
     * @return MFCC特征
     */
    // std::vector<std::vector<float>> compute_mfcc(
    //     const float* audio,
    //     size_t audio_len);

    /**
     * 计算delta特征 (如果需要保留)
     *
     * @param features 特征数据
     * @param width delta窗口宽度
     * @return delta特征
     */
    // std::vector<std::vector<float>> compute_delta(
    //     const std::vector<std::vector<float>>& features,
    //     int width = 3);

    /**
     * 获取特征维度
     * 
     * @return 特征维度
     */
    int get_feature_dim() const { return n_mfcc_; } // 直接返回n_mfcc

    /**
     * 将短音频填充到足够的长度 (如果需要保留)
     *
     * @param audio 音频数据
     * @param audio_len 音频数据长度
     * @return 填充后的音频
     */
    // std::vector<float> pad_audio(const float* audio, size_t audio_len);

private:
    // 配置参数
    int sample_rate_;
    int window_size_;       // 样本数
    int window_stride_;     // 样本数
    int n_fft_;
    int n_mels_;
    double preemphasis_coeff_;
    std::vector<double> window_func_;
    std::vector<std::vector<double>> mel_filterbank_;
    std::vector<double> fft_input_;
    std::vector<std::complex<double>> fft_output_;
    std::vector<double> mel_energies_;
    std::vector<double> log_mel_energies_; // Keep this temporary storage
    std::vector<double> mfcc_coeffs_;       // For MFCC calculation result
    std::vector<double> delta_coeffs_;      // For Delta calculation result

    // New parameters for MFCC
    int n_mfcc_;
    bool use_delta_;
    bool use_delta2_; // Keep for completeness, though config is false

    // Precomputed DCT matrix for efficiency
    std::vector<std::vector<double>> dct_matrix_;

    // FFTW plan (Use void* or fftw_plan)
    // fftwf_plan fft_plan_ = nullptr; // Old float plan
    void* fft_plan_ = nullptr; // Use void* for generic plan handle
    // Or include fftw3.h here and use: fftw_plan fft_plan_ = nullptr;

    // 初始化函数
    void initialize(const FeatureConfig& config);
    void init_fft_plan();
    void init_window_function();
    void init_mel_filterbank();
    void init_dct_matrix();

    // 辅助函数
    double hz_to_mel(double hz);
    double mel_to_hz(double mel);

    // 移除 Config 相关
    // Config config_;
    // void validate_config();
    // Config create_config_from_feature_config(const FeatureConfig& feature_config);

    // 移除旧的 FFT 成员
    // void* fft_plan_;
    // int fft_size_;

    // 移除旧的或重复的预计算数据
    // std::vector<float> hamming_window_;
    // float* window_func_;
    // float* mel_filters_;
    // int feature_dim_;

    void initialize_fft(); // Assuming fft setup is handled
    void apply_preemphasis(const std::vector<double>& audio_in, std::vector<double>& audio_out);
    void apply_window(const std::vector<double>& frame, std::vector<double>& windowed_frame);
    void compute_fft(const std::vector<double>& windowed_frame, std::vector<std::complex<double>>& spectrum);
    void compute_mel_filterbank(const std::vector<std::complex<double>>& spectrum, std::vector<double>& mel_energies);
    void compute_log_mel_energies(const std::vector<double>& mel_energies, std::vector<double>& log_mel_energies);

    // New helper functions
    void initialize_dct_matrix();
    void compute_dct(const std::vector<double>& log_mel_energies, std::vector<double>& mfcc_coeffs);
    void compute_deltas(const std::vector<std::vector<double>>& features, int delta_width, std::vector<std::vector<double>>& deltas);

}; 

#endif /* FEATURE_EXTRACTOR_H */ 