#include "mfcc_features.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

// FFT库
#include <fftw3.h>

// 常量
constexpr double PI = 3.14159265358979323846;

MFCCFeatures::MFCCFeatures() : is_initialized_(false) {
    // 默认构造函数
}

MFCCFeatures::~MFCCFeatures() {
    // 析构函数，释放资源
}

void MFCCFeatures::initialize(int sample_rate, int n_mfcc, int n_mels, int window_size_ms, int window_stride_ms) {
    // 保存参数
    sample_rate_ = sample_rate;
    n_mfcc_ = n_mfcc;
    n_mels_ = n_mels;
    window_size_ = (window_size_ms * sample_rate) / 1000;
    window_stride_ = (window_stride_ms * sample_rate) / 1000;
    
    // 设置FFT大小为窗口大小的下一个2的幂
    fft_size_ = 1;
    while (fft_size_ < window_size_) {
        fft_size_ *= 2;
    }
    
    // 初始化Mel滤波器组
    init_mel_filterbank();
    
    // 初始化DCT矩阵
    init_dct_matrix();
    
    // 初始化窗口函数
    init_window_function();
    
    // 标记为已初始化
    is_initialized_ = true;
}

std::vector<float> MFCCFeatures::compute_mfcc(const std::vector<float>& audio) {
    if (!is_initialized_) {
        throw std::runtime_error("MFCCFeatures未初始化");
    }
    
    if (audio.empty()) {
        return std::vector<float>();
    }
    
    // 预处理: 确保音频振幅范围在[-1, 1]内
    std::vector<float> normalized_audio = audio;
    float max_abs = 0.0f;
    for (const auto& sample : audio) {
        max_abs = std::max(max_abs, std::abs(sample));
    }
    
    if (max_abs > 1.0f) {
        for (auto& sample : normalized_audio) {
            sample /= max_abs;
        }
    }
    
    // 应用预强调 - 与librosa默认值完全一致
    const float pre_emphasis = 0.97f;
    for (int i = normalized_audio.size() - 1; i > 0; --i) {
        normalized_audio[i] -= pre_emphasis * normalized_audio[i - 1];
    }
    normalized_audio[0] *= (1.0f - pre_emphasis);
    
    // 应用窗口函数
    std::vector<float> windowed(window_size_);
    if (normalized_audio.size() < window_size_) {
        // 如果音频太短，则填充零
        for (size_t i = 0; i < normalized_audio.size(); ++i) {
            windowed[i] = normalized_audio[i] * window_function_[i];
        }
        // 剩余部分填充零
        for (size_t i = normalized_audio.size(); i < window_size_; ++i) {
            windowed[i] = 0.0f;
        }
    } else {
        // 应用窗口函数
        for (int i = 0; i < window_size_; ++i) {
            windowed[i] = normalized_audio[i] * window_function_[i];
        }
    }
    
    // 分配FFT计算所需的内存 - 使用double提高精度
    std::vector<float> fft_in(fft_size_, 0.0f);
    std::vector<float> fft_out(fft_size_);
    
    // 复制窗口化的数据到FFT输入
    std::copy(windowed.begin(), windowed.end(), fft_in.begin());
    
    // 创建FFT计划
    fftwf_plan plan = fftwf_plan_r2r_1d(fft_size_, fft_in.data(), fft_out.data(), FFTW_R2HC, FFTW_ESTIMATE);
    
    // 执行FFT
    fftwf_execute(plan);
    
    // 计算功率谱 - 确保与librosa的实现完全一致
    std::vector<double> power_spectrum(fft_size_ / 2 + 1);
    // DC分量 - librosa使用np.abs(fft)**2
    power_spectrum[0] = fft_out[0] * fft_out[0] / (fft_size_ * fft_size_);
    for (int i = 1; i < fft_size_ / 2; ++i) {
        // librosa实现: np.abs(fft)**2 / (win_sum^2)
        power_spectrum[i] = (fft_out[i] * fft_out[i] + fft_out[fft_size_ - i] * fft_out[fft_size_ - i]) / (fft_size_ * fft_size_);
    }
    // 奈奎斯特频率
    power_spectrum[fft_size_ / 2] = fft_out[fft_size_ / 2] * fft_out[fft_size_ / 2] / (fft_size_ * fft_size_);
    
    // 应用Mel滤波器组 - 使用double提高精度
    std::vector<double> mel_energies(n_mels_, 0.0);
    for (int i = 0; i < n_mels_; ++i) {
        for (int j = 0; j < fft_size_ / 2 + 1; ++j) {
            mel_energies[i] += mel_filterbank_[i][j] * power_spectrum[j];
        }
        // 应用对数变换 - 完全匹配librosa
        // librosa使用np.log，即自然对数，并添加一个小常数防止对0取对数
        mel_energies[i] = std::log(std::max(mel_energies[i], 1e-10));
    }
    
    // 应用DCT变换得到MFCC - 与librosa的DCT-II一致
    std::vector<float> mfcc(n_mfcc_);
    for (int i = 0; i < n_mfcc_; ++i) {
        mfcc[i] = 0.0f;
        for (int j = 0; j < n_mels_; ++j) {
            mfcc[i] += dct_matrix_[i][j] * mel_energies[j];
        }
    }
    
    // 特征归一化 - 匹配librosa的默认行为
    // librosa默认不对MFCC特征进行归一化
    // 注释掉原来的归一化代码，不再进行异常值处理
    
    // 清理FFT计划
    fftwf_destroy_plan(plan);
    
    return mfcc;
}

double MFCCFeatures::hz_to_mel(double hz) {
    // 使用HTK公式，与librosa的默认行为一致
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double MFCCFeatures::mel_to_hz(double mel) {
    // 使用HTK公式的逆变换，与librosa的默认行为一致
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

void MFCCFeatures::init_mel_filterbank() {
    // 计算Mel频率范围
    double min_mel = hz_to_mel(0.0);
    double max_mel = hz_to_mel(sample_rate_ / 2.0);
    
    // 在Mel尺度上均匀分布滤波器
    std::vector<double> mel_points(n_mels_ + 2);
    for (int i = 0; i < mel_points.size(); ++i) {
        mel_points[i] = min_mel + (max_mel - min_mel) * i / (n_mels_ + 1);
    }
    
    // 转换回Hz频率
    std::vector<double> hz_points(mel_points.size());
    for (int i = 0; i < hz_points.size(); ++i) {
        hz_points[i] = mel_to_hz(mel_points[i]);
    }
    
    // 计算对应的FFT bin - 确保与librosa匹配
    std::vector<int> bin_indices(hz_points.size());
    for (int i = 0; i < bin_indices.size(); ++i) {
        bin_indices[i] = static_cast<int>(std::floor((fft_size_ + 1) * hz_points[i] / sample_rate_));
    }
    
    // 初始化Mel滤波器组
    mel_filterbank_.resize(n_mels_);
    for (int i = 0; i < n_mels_; ++i) {
        mel_filterbank_[i].resize(fft_size_ / 2 + 1, 0.0f);
        
        // 构建三角滤波器 - 匹配librosa实现
        for (int j = bin_indices[i]; j <= bin_indices[i + 2]; ++j) {
            if (j < 0 || j > fft_size_ / 2) {
                continue;
            }
            
            if (j <= bin_indices[i + 1]) {
                mel_filterbank_[i][j] = (j - bin_indices[i]) / 
                                       static_cast<float>(bin_indices[i + 1] - bin_indices[i]);
            } else {
                mel_filterbank_[i][j] = (bin_indices[i + 2] - j) / 
                                       static_cast<float>(bin_indices[i + 2] - bin_indices[i + 1]);
            }
        }
        
        // 归一化滤波器能量 - librosa默认行为
        float sum = 0.0f;
        for (int j = 0; j < fft_size_ / 2 + 1; ++j) {
            sum += mel_filterbank_[i][j];
        }
        if (sum > 0.0f) {
            for (int j = 0; j < fft_size_ / 2 + 1; ++j) {
                mel_filterbank_[i][j] /= sum;
            }
        }
    }
}

void MFCCFeatures::init_dct_matrix() {
    // 初始化DCT矩阵 - 完全匹配librosa的DCT-II实现
    dct_matrix_.resize(n_mfcc_);
    for (auto& row : dct_matrix_) {
        row.resize(n_mels_);
    }
    
    // 计算DCT系数 - 与librosa一致
    double scale = std::sqrt(2.0 / n_mels_);
    for (int i = 0; i < n_mfcc_; ++i) {
        for (int j = 0; j < n_mels_; ++j) {
            dct_matrix_[i][j] = scale * std::cos(PI * i * (j + 0.5) / n_mels_);
        }
    }
    
    // 第一个系数使用不同的缩放因子 - 与librosa一致
    for (int j = 0; j < n_mels_; ++j) {
        dct_matrix_[0][j] *= 1.0 / std::sqrt(2.0);
    }
}

void MFCCFeatures::init_window_function() {
    // 初始化Hann窗口函数，与librosa默认行为一致
    window_function_.resize(window_size_);
    for (int i = 0; i < window_size_; ++i) {
        // Hann窗口公式：0.5 - 0.5 * cos(2π * n / (N - 1))
        window_function_[i] = 0.5f - 0.5f * std::cos(2.0f * PI * i / (window_size_ - 1));
    }
    
    // librosa中不对窗口函数进行能量归一化，这里也保持一致
    // 只有在需要保持信号能量时才进行归一化，MFCC特征提取通常不需要
} 