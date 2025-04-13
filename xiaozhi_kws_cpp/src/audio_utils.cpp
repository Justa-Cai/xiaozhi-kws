/**
 * @file audio_utils.cpp
 * @brief 音频处理工具函数实现
 */

#include "audio_utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>

// WAV文件头结构
struct WAVHeader {
    char riff[4];                // "RIFF"
    uint32_t chunk_size;         // 文件大小 - 8
    char wave[4];                // "WAVE"
    char fmt[4];                 // "fmt "
    uint32_t fmt_chunk_size;     // 格式区块大小
    uint16_t audio_format;       // 音频格式 (1 = PCM)
    uint16_t num_channels;       // 通道数
    uint32_t sample_rate;        // 采样率
    uint32_t byte_rate;          // 每秒字节数
    uint16_t block_align;        // 数据块对齐
    uint16_t bits_per_sample;    // 采样位深
    // 可能有其他额外字段...
};

bool load_audio_file(const std::string& filepath, std::vector<int16_t>& audio_data, size_t& sample_rate) {
    // 打开文件
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "无法打开音频文件: " << filepath << std::endl;
        return false;
    }
    
    // 检查文件扩展名
    std::string extension = filepath.substr(filepath.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    bool result = false;
    if (extension == "wav") {
        result = load_wav_file(file, audio_data, sample_rate);
    } else if (extension == "mp3") {
        std::cerr << "MP3格式需要额外库支持，本例简化处理" << std::endl;
        result = false;
    } else {
        std::cerr << "不支持的音频格式: " << extension << std::endl;
        result = false;
    }
    
    file.close();
    return result;
}

bool load_wav_file(std::ifstream& file, std::vector<int16_t>& audio_data, size_t& sample_rate) {
    // 读取WAV头
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));
    
    if (file.fail()) {
        std::cerr << "读取WAV头失败" << std::endl;
        return false;
    }
    
    // 验证WAV头
    if (strncmp(header.riff, "RIFF", 4) != 0 ||
        strncmp(header.wave, "WAVE", 4) != 0 ||
        strncmp(header.fmt, "fmt ", 4) != 0) {
        std::cerr << "无效的WAV文件头" << std::endl;
        return false;
    }
    
    // 不支持非PCM格式
    if (header.audio_format != 1) {
        std::cerr << "仅支持PCM格式WAV文件" << std::endl;
        return false;
    }
    
    // 仅支持16位采样
    if (header.bits_per_sample != 16) {
        std::cerr << "仅支持16位采样WAV文件" << std::endl;
        return false;
    }
    
    // 设置采样率
    sample_rate = header.sample_rate;
    
    // 跳到数据部分
    bool found_data = false;
    char chunk_id[4];
    uint32_t chunk_size;
    
    // 查找数据块
    while (!file.eof() && !found_data) {
        file.read(chunk_id, 4);
        if (file.fail()) break;
        
        file.read(reinterpret_cast<char*>(&chunk_size), 4);
        if (file.fail()) break;
        
        if (strncmp(chunk_id, "data", 4) == 0) {
            found_data = true;
        } else {
            // 跳过当前块
            file.seekg(chunk_size, std::ios::cur);
            if (file.fail()) break;
        }
    }
    
    if (!found_data) {
        std::cerr << "未找到WAV数据块" << std::endl;
        return false;
    }
    
    // 计算样本数
    size_t num_samples = chunk_size / (header.bits_per_sample / 8);
    if (header.num_channels > 1) {
        num_samples /= header.num_channels;
    }
    
    // 分配空间
    audio_data.resize(num_samples);
    
    // 读取音频数据（如果是立体声，仅取第一个通道）
    if (header.num_channels == 1) {
        // 单声道，直接读取
        file.read(reinterpret_cast<char*>(audio_data.data()), chunk_size);
    } else {
        // 多声道，只取第一个通道
        std::vector<int16_t> temp_buffer(header.num_channels);
        for (size_t i = 0; i < num_samples; ++i) {
            file.read(reinterpret_cast<char*>(temp_buffer.data()), header.num_channels * sizeof(int16_t));
            audio_data[i] = temp_buffer[0];  // 只取第一个通道
        }
    }
    
    return !file.fail();
}

std::vector<int16_t> resample_audio(const std::vector<int16_t>& audio_data, 
                                   size_t original_sample_rate, 
                                   size_t target_sample_rate) {
    // 简化处理：线性插值重采样
    if (original_sample_rate == target_sample_rate) {
        return audio_data;
    }
    
    double ratio = static_cast<double>(target_sample_rate) / original_sample_rate;
    size_t output_size = static_cast<size_t>(audio_data.size() * ratio);
    
    std::vector<int16_t> resampled(output_size);
    
    for (size_t i = 0; i < output_size; ++i) {
        double src_idx = i / ratio;
        size_t src_idx_floor = static_cast<size_t>(src_idx);
        double frac = src_idx - src_idx_floor;
        
        if (src_idx_floor + 1 < audio_data.size()) {
            // 线性插值
            resampled[i] = static_cast<int16_t>(
                audio_data[src_idx_floor] * (1.0 - frac) + 
                audio_data[src_idx_floor + 1] * frac);
        } else {
            resampled[i] = audio_data[src_idx_floor];
        }
    }
    
    return resampled;
} 