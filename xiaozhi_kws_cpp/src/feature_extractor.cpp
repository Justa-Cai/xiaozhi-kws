#include "feature_extractor.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <string>
#include <fstream>
#include <numeric>

// 使用FFTW库进行FFT计算
#include <fftw3.h>

#define _USE_MATH_DEFINES
#include <math.h>

// 常量
constexpr double PI = 3.14159265358979323846;

// Mel频率变换函数
static double hz_to_mel(double hz) {
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

static double mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

FeatureExtractor::FeatureExtractor(const Config& config)
    : config_(config), fft_plan_(nullptr), mel_filters_(nullptr), window_func_(nullptr) {
    
    // 验证配置
    validate_config();
    
    // 初始化FFT和相关参数
    init_fft();
    
    // 初始化梅尔滤波器组
    init_mel_filterbank();
    
    // 初始化DCT矩阵
    init_dct_matrix();
    
    // 初始化汉明窗
    hamming_window_.resize(config_.frame_length);
    for (int i = 0; i < config_.frame_length; i++) {
        hamming_window_[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (config_.frame_length - 1));
    }
    
    // 计算特征维度
    feature_dim_ = config.num_mfcc;
    if (config.use_delta) {
        feature_dim_ += config.num_mfcc;
    }
    if (config.use_delta2) {
        feature_dim_ += config.num_mfcc;
    }
    
    // 初始化窗口函数
    init_window_function();
    
    // 初始化Mel滤波器组
    init_mel_filters();
    
    // 创建FFT计划
    fft_plan_ = fftwf_plan_r2r_1d(config.fft_size, nullptr, nullptr, FFTW_R2HC, FFTW_ESTIMATE);
    
    if (!fft_plan_) {
        throw std::runtime_error("Failed to create FFTW plan");
    }
}

FeatureExtractor::FeatureExtractor(const FeatureConfig& feature_config)
    : FeatureExtractor(create_config_from_feature_config(feature_config)) {
    // 使用委托构造函数，具体实现在主构造函数中
}

Config FeatureExtractor::create_config_from_feature_config(const FeatureConfig& feature_config) {
    Config config;
    
    // 从FeatureConfig转换到Config
    config.sample_rate = feature_config.sample_rate;
    
    // 窗口大小和窗口步长从毫秒转换为样本数
    config.frame_length = static_cast<int>(feature_config.window_size_ms * feature_config.sample_rate / 1000.0);
    config.frame_shift = static_cast<int>(feature_config.window_stride_ms * feature_config.sample_rate / 1000.0);
    
    // FFT相关配置
    config.fft_size = feature_config.n_fft > 0 ? feature_config.n_fft : config.frame_length;
    config.num_filters = feature_config.n_mels;
    config.num_mfcc = feature_config.n_mfcc;
    
    // 差分特征
    config.use_delta = feature_config.use_delta;
    config.use_delta2 = feature_config.use_delta2;
    
    // 其他配置使用默认值
    config.low_freq = 0.0;
    config.high_freq = feature_config.sample_rate / 2.0;
    config.pre_emphasis = 0.97;
    config.dither = 0.0;
    
    return config;
}

FeatureExtractor::~FeatureExtractor() {
    if (fft_plan_) {
        fftwf_destroy_plan(static_cast<fftwf_plan>(fft_plan_));
    }
    
    delete[] mel_filters_;
    delete[] window_func_;
}

void FeatureExtractor::validate_config() {
    if (config_.sample_rate <= 0) {
        throw std::invalid_argument("采样率必须大于0");
    }
    
    if (config_.frame_length <= 0) {
        throw std::invalid_argument("帧长度必须大于0");
    }
    
    if (config_.frame_shift <= 0) {
        throw std::invalid_argument("帧移必须大于0");
    }
    
    if (config_.num_mfcc <= 0) {
        throw std::invalid_argument("MFCC系数数量必须大于0");
    }
    
    if (config_.num_filters <= 0) {
        throw std::invalid_argument("滤波器数量必须大于0");
    }
    
    if (config_.fft_size <= 0) {
        throw std::invalid_argument("FFT大小必须大于0");
    }
    
    if (config_.low_freq < 0) {
        throw std::invalid_argument("最低频率不能为负数");
    }
    
    if (config_.high_freq > config_.sample_rate / 2) {
        throw std::invalid_argument("最高频率不能超过奈奎斯特频率");
    }
}

void FeatureExtractor::init_fft() {
    // 根据配置设置FFT大小
    fft_size_ = config_.fft_size;
    
    // 实际实现中可以使用FFTW或其他FFT库
    // 这里简化处理，不具体实现FFT
}

void FeatureExtractor::init_mel_filterbank() {
    // 创建梅尔滤波器组 - 对齐librosa实现
    filterbank_.resize(config_.num_filters);
    for (auto& filter : filterbank_) {
        filter.resize(fft_size_ / 2 + 1, 0.0f);
    }
    
    // 将赫兹频率转换为梅尔频率 - 使用librosa公式
    float mel_low = 2595.0 * log10(1.0 + config_.low_freq / 700.0);
    float mel_high = 2595.0 * log10(1.0 + config_.high_freq / 700.0);
    
    // 计算梅尔刻度上的等间距点
    std::vector<float> mel_points(config_.num_filters + 2);
    for (int i = 0; i < mel_points.size(); i++) {
        mel_points[i] = mel_low + i * (mel_high - mel_low) / (config_.num_filters + 1);
    }
    
    // 将梅尔刻度点转换回赫兹 - 使用librosa公式
    std::vector<float> hz_points(mel_points.size());
    for (int i = 0; i < mel_points.size(); i++) {
        hz_points[i] = 700.0 * (pow(10.0, mel_points[i] / 2595.0) - 1.0);
    }
    
    // 转换为FFT bin编号 - 使用librosa算法
    std::vector<int> bin_indices(mel_points.size());
    for (int i = 0; i < mel_points.size(); i++) {
        bin_indices[i] = static_cast<int>(floor((fft_size_ + 1) * hz_points[i] / config_.sample_rate));
    }
    
    // 创建三角滤波器 - 匹配librosa实现
    for (int i = 0; i < config_.num_filters; i++) {
        for (int j = bin_indices[i]; j < bin_indices[i + 2]; j++) {
            if (j < 0 || j >= fft_size_ / 2 + 1) continue;
            
            if (j < bin_indices[i + 1]) {
                filterbank_[i][j] = (j - bin_indices[i]) / static_cast<float>(bin_indices[i + 1] - bin_indices[i]);
            } else {
                filterbank_[i][j] = (bin_indices[i + 2] - j) / static_cast<float>(bin_indices[i + 2] - bin_indices[i + 1]);
            }
        }
        
        // 应用能量归一化 - librosa默认行为
        float sum = 0.0f;
        for (int j = 0; j < fft_size_ / 2 + 1; j++) {
            sum += filterbank_[i][j];
        }
        if (sum > 0.0f) {
            for (int j = 0; j < fft_size_ / 2 + 1; j++) {
                filterbank_[i][j] /= sum;
            }
        }
    }
}

void FeatureExtractor::init_dct_matrix() {
    // 初始化DCT矩阵 - 对齐librosa实现
    dct_matrix_.resize(config_.num_mfcc);
    for (auto& row : dct_matrix_) {
        row.resize(config_.num_filters);
    }
    
    // 使用librosa DCT公式
    double scale = sqrt(2.0 / config_.num_filters);
    for (int i = 0; i < config_.num_mfcc; i++) {
        for (int j = 0; j < config_.num_filters; j++) {
            dct_matrix_[i][j] = scale * cos(M_PI * i * (j + 0.5) / config_.num_filters);
        }
    }
    
    // 第一个系数使用不同的缩放 - librosa行为
    for (int j = 0; j < config_.num_filters; j++) {
        dct_matrix_[0][j] *= 1.0 / sqrt(2.0);
    }
}

void FeatureExtractor::init_window_function() {
    // 创建Hamming窗口函数 - 对齐librosa实现
    window_func_ = new float[config_.frame_length];
    
    for (int i = 0; i < config_.frame_length; ++i) {
        window_func_[i] = 0.54f - 0.46f * std::cos(2.0f * PI * i / (config_.frame_length - 1));
    }
    
    // 应用能量归一化 - librosa默认行为
    float sum = 0.0f;
    for (int i = 0; i < config_.frame_length; ++i) {
        sum += window_func_[i];
    }
    for (int i = 0; i < config_.frame_length; ++i) {
        window_func_[i] /= sum;
    }
}

void FeatureExtractor::init_mel_filters() {
    // 计算Mel频率范围
    double min_mel = hz_to_mel(0.0);
    double max_mel = hz_to_mel(config_.sample_rate / 2.0);
    
    // Mel频率均匀间隔
    std::vector<double> mel_points(config_.num_filters + 2);
    for (int i = 0; i < mel_points.size(); ++i) {
        mel_points[i] = min_mel + (max_mel - min_mel) * i / (config_.num_filters + 1);
    }
    
    // 转换回Hz频率
    std::vector<double> hz_points(mel_points.size());
    for (int i = 0; i < hz_points.size(); ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // 转换为FFT bin
    std::vector<int> bins(hz_points.size());
    for (int i = 0; i < bins.size(); ++i) {
        bins[i] = static_cast<int>(std::floor((config_.fft_size + 1) * hz_points[i] / config_.sample_rate));
    }
    
    // 创建Mel滤波器组
    mel_filters_ = new float[config_.num_filters * (config_.fft_size / 2 + 1)];
    std::fill_n(mel_filters_, config_.num_filters * (config_.fft_size / 2 + 1), 0.0f);
    
    for (int i = 0; i < config_.num_filters; ++i) {
        for (int j = bins[i]; j < bins[i + 2]; ++j) {
            if (j < bins[i + 1]) {
                mel_filters_[i * (config_.fft_size / 2 + 1) + j] = 
                    static_cast<float>((j - bins[i]) / static_cast<double>(bins[i + 1] - bins[i]));
            } else {
                mel_filters_[i * (config_.fft_size / 2 + 1) + j] = 
                    static_cast<float>((bins[i + 2] - j) / static_cast<double>(bins[i + 2] - bins[i + 1]));
            }
        }
    }
}

std::vector<std::vector<float>> FeatureExtractor::extract_features(
    const int16_t* audio, 
    size_t audio_len, 
    bool normalize) {
    
    if (!audio || audio_len == 0) {
        throw std::runtime_error("Invalid audio data");
    }
    
    // 转换为float
    std::vector<float> audio_float(audio_len);
    for (size_t i = 0; i < audio_len; ++i) {
        audio_float[i] = static_cast<float>(audio[i]) / 32768.0f;
    }
    
    // 如果音频振幅不在[-1, 1]范围内，进行归一化
    if (normalize) {
        float max_amp = 0.0f;
        for (size_t i = 0; i < audio_len; ++i) {
            max_amp = std::max(max_amp, std::abs(audio_float[i]));
        }
        
        if (max_amp > 1.0f) {
            for (size_t i = 0; i < audio_len; ++i) {
                audio_float[i] /= max_amp;
            }
        }
    }
    
    // 确保音频长度足够
    if (audio_len < config_.frame_length) {
        audio_float = pad_audio(audio_float.data(), audio_len);
        audio_len = audio_float.size();
    }
    
    // 计算MFCC特征
    std::vector<std::vector<float>> mfccs = compute_mfcc(audio_float.data(), audio_len);
    
    // 结果特征
    std::vector<std::vector<float>> features = mfccs;
    
    // 计算delta特征
    if (config_.use_delta) {
        std::vector<std::vector<float>> delta = compute_delta(mfccs);
        
        // 将delta添加到特征中
        for (size_t i = 0; i < mfccs.size(); ++i) {
            features[i].insert(features[i].end(), delta[i].begin(), delta[i].end());
        }
        
        // 计算delta2特征
        if (config_.use_delta2) {
            std::vector<std::vector<float>> delta2 = compute_delta(delta);
            
            // 将delta2添加到特征中
            for (size_t i = 0; i < mfccs.size(); ++i) {
                features[i].insert(features[i].end(), delta2[i].begin(), delta2[i].end());
            }
        }
    }
    
    return features;
}

std::vector<std::vector<std::vector<float>>> FeatureExtractor::extract_features_sliding_window(
    const int16_t* audio, 
    size_t audio_len, 
    int window_size_ms, 
    int stride_ms) {
    
    if (!audio || audio_len == 0) {
        throw std::runtime_error("Invalid audio data");
    }
    
    // 计算窗口大小和步长（样本数）
    int window_samples = static_cast<int>(window_size_ms * config_.sample_rate / 1000);
    int stride_samples = static_cast<int>(stride_ms * config_.sample_rate / 1000);
    
    // 确保音频长度足够
    if (audio_len < window_samples) {
        throw std::runtime_error("Audio length too short for sliding window");
    }
    
    // 结果特征
    std::vector<std::vector<std::vector<float>>> features_windows;
    
    // 滑动窗口提取特征
    for (size_t start = 0; start <= audio_len - window_samples; start += stride_samples) {
        // 提取窗口
        std::vector<int16_t> window(audio + start, audio + start + window_samples);
        
        // 提取特征
        auto features = extract_features(window.data(), window.size());
        
        // 添加到结果
        features_windows.push_back(features);
    }
    
    return features_windows;
}

std::vector<std::vector<float>> FeatureExtractor::compute_mfcc(
    const float* audio, 
    size_t audio_len) {
    
    // 帧数
    int num_frames = 1 + (audio_len - config_.frame_length) / config_.frame_shift;
    
    // 结果MFCC特征
    std::vector<std::vector<float>> mfccs(num_frames, std::vector<float>(config_.num_mfcc));
    
    // 分配FFT缓冲区
    std::vector<float> fft_in(config_.fft_size);
    std::vector<float> fft_out(config_.fft_size);
    std::vector<float> power_spectrum(config_.fft_size / 2 + 1);
    std::vector<float> mel_energies(config_.num_filters);
    
    // 逐帧处理
    for (int frame = 0; frame < num_frames; ++frame) {
        // 当前帧的起始位置
        int start = frame * config_.frame_shift;
        
        // 填充FFT输入缓冲区
        std::fill(fft_in.begin(), fft_in.end(), 0.0f);
        for (int i = 0; i < config_.frame_length && start + i < audio_len; ++i) {
            fft_in[i] = audio[start + i] * window_func_[i];
        }
        
        // 执行FFT
        fftwf_execute_r2r(static_cast<fftwf_plan>(fft_plan_), 
                         fft_in.data(), 
                         fft_out.data());
        
        // 计算功率谱 - 对齐librosa实现
        power_spectrum[0] = fft_out[0] * fft_out[0];  // DC component
        for (int i = 1; i < config_.fft_size / 2; ++i) {
            power_spectrum[i] = fft_out[i] * fft_out[i] + fft_out[config_.fft_size - i] * fft_out[config_.fft_size - i];
        }
        power_spectrum[config_.fft_size / 2] = fft_out[config_.fft_size / 2] * fft_out[config_.fft_size / 2];  // Nyquist
        
        // 应用能量归一化 - librosa默认行为
        for (int i = 0; i < config_.fft_size / 2 + 1; ++i) {
            power_spectrum[i] = power_spectrum[i] / config_.fft_size;
        }
        
        // 应用Mel滤波器组
        std::fill(mel_energies.begin(), mel_energies.end(), 0.0f);
        for (int i = 0; i < config_.num_filters; ++i) {
            for (int j = 0; j < config_.fft_size / 2 + 1; ++j) {
                mel_energies[i] += power_spectrum[j] * filterbank_[i][j];
            }
            // 取对数
            mel_energies[i] = std::log(mel_energies[i] + 1e-10f);
        }
        
        // 计算DCT（离散余弦变换）
        for (int i = 0; i < config_.num_mfcc; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < config_.num_filters; ++j) {
                sum += mel_energies[j] * dct_matrix_[i][j];
            }
            mfccs[frame][i] = sum;
        }
    }
    
    return mfccs;
}

std::vector<std::vector<float>> FeatureExtractor::compute_delta(
    const std::vector<std::vector<float>>& features,
    int width) {
    
    // 对齐librosa delta实现
    int num_frames = features.size();
    int num_coeffs = features[0].size();
    
    std::vector<std::vector<float>> delta(num_frames, std::vector<float>(num_coeffs, 0.0f));
    
    // 计算分母和分子的阶
    int denominator = 0;
    for (int i = 1; i <= width; ++i) {
        denominator += i * i;
    }
    denominator *= 2;
    
    // 计算delta特征
    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < num_coeffs; ++j) {
            float numerator = 0.0f;
            
            for (int k = 1; k <= width; ++k) {
                // 左侧样本
                int left_idx = i - k;
                if (left_idx < 0) {
                    left_idx = 0;  // 使用边界值处理
                }
                
                // 右侧样本
                int right_idx = i + k;
                if (right_idx >= num_frames) {
                    right_idx = num_frames - 1;  // 使用边界值处理
                }
                
                // 累加加权差分
                numerator += k * (features[right_idx][j] - features[left_idx][j]);
            }
            
            // 计算delta
            delta[i][j] = numerator / denominator;
        }
    }
    
    return delta;
}

std::vector<float> FeatureExtractor::pad_audio(const float* audio, size_t audio_len) {
    // 计算所需的最小长度
    int min_samples = config_.frame_length + 4 * config_.frame_shift;
    
    // 创建填充后的音频
    std::vector<float> padded_audio(min_samples);
    
    // 复制原始音频
    std::copy(audio, audio + audio_len, padded_audio.begin());
    
    // 其余部分填充为0
    std::fill(padded_audio.begin() + audio_len, padded_audio.end(), 0.0f);
    
    return padded_audio;
} 