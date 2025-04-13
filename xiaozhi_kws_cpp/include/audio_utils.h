/**
 * @file audio_utils.h
 * @brief 音频处理工具函数
 */

#ifndef AUDIO_UTILS_H
#define AUDIO_UTILS_H

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

/**
 * 加载音频文件
 * 
 * @param filepath 文件路径
 * @param audio_data 音频数据输出
 * @param sample_rate 采样率输出
 * @return 是否成功
 */
bool load_audio_file(const std::string& filepath, std::vector<int16_t>& audio_data, size_t& sample_rate);

/**
 * 加载WAV文件
 * 
 * @param file 已打开的文件流
 * @param audio_data 音频数据输出
 * @param sample_rate 采样率输出
 * @return 是否成功
 */
bool load_wav_file(std::ifstream& file, std::vector<int16_t>& audio_data, size_t& sample_rate);

/**
 * 重采样音频数据
 * 
 * @param audio_data 原始音频数据
 * @param original_sample_rate 原始采样率
 * @param target_sample_rate 目标采样率
 * @return 重采样后的音频数据
 */
std::vector<int16_t> resample_audio(const std::vector<int16_t>& audio_data, 
                                  size_t original_sample_rate, 
                                  size_t target_sample_rate);

#endif /* AUDIO_UTILS_H */ 