#include "detector.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <chrono>
#include <cstring>
#include <memory>
#include <utility>
#include <numeric>
#include <stdexcept>
#include <cassert>
#include <filesystem>
// #include "mfcc_features.h" // 移除

// 音频处理库
// #include <sndfile.h> // 移除旧的sndfile依赖，统一使用FFmpeg
// 添加FFmpeg支持
#include <cstdio>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

// 静态全局变量
// static bool has_registered_custom_ops = false; // 移除未使用变量

Detector::Detector(const std::string& model_path,
                 const std::string& config_path,
                 float threshold)
    : last_trigger_time_(std::chrono::steady_clock::now() - trigger_cooldown_) { // 初始化冷却时间

    try {
        // 加载配置
        config_parser_ = std::make_unique<ConfigParser>(config_path);
        inference_config_ = config_parser_->get_inference_config();
        feature_config_ = config_parser_->get_feature_config(); // 存储特征配置

        // 设置阈值
        if (threshold > 0.0f) {
            inference_config_.detection_threshold = threshold;
        } else {
            // 如果配置文件中没有或为无效值，使用默认值0.5
            if (inference_config_.detection_threshold <= 0.0f || inference_config_.detection_threshold > 1.0f) {
                 inference_config_.detection_threshold = 0.5f;
            }
            // 否则使用配置文件中的值
        }
        std::cout << "[Detector] 使用检测阈值: " << inference_config_.detection_threshold << std::endl;


        // 创建特征提取器 - 使用 FeatureConfig
        feature_extractor_ = std::make_unique<FeatureExtractor>(feature_config_);

        // 移除对 MFCCFeatures 的直接创建，由 FeatureExtractor 管理
        // mfcc_features_ = std::make_unique<MFCCFeatures>();

        // 创建模型
        auto model_config = config_parser_->get_model_config();
        model_ = std::make_unique<Model>(model_path, model_config, inference_config_.detection_threshold); // 将阈值传递给模型

        // 初始化ONNX Runtime环境和会话 (移到Model类内部处理)
        // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "xiaozhi_kws_detector");
        // ... (移除ONNX RT初始化代码) ...

        // 初始化预测队列
        recent_predictions_.resize(inference_config_.smoothing_window, 0.0f);

        // 初始化xiaozhi配置 (如果PostProcessor还需要旧Config)
        // config_.load_from_file(config_path);
        // config_.detection_threshold = inference_config_.detection_threshold;
        // config_.smooth_window_size = inference_config_.smoothing_window; // 使用新配置
        // config_.apply_vad = true;

        // 创建后处理器 - 使用 InferenceConfig
        post_processor_ = std::make_unique<xiaozhi::PostProcessor>(inference_config_);


    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to initialize detector: ") + e.what());
    }
}

Detector::~Detector() = default;

void Detector::set_callback(DetectionCallback callback) {
    callback_ = callback;
}

bool Detector::process_audio(const int16_t* audio_data, size_t audio_len) {
    if (!audio_data || audio_len == 0) {
        return false;
    }
    
    try {
        // 转换音频格式
        std::vector<float> audio_float = load_audio_from_pcm(audio_data, audio_len);
        
        // 计算音频能量（用于VAD）
        float audio_energy = calculate_audio_energy(audio_float);
        
        // 转换为int16_t格式，feature_extractor_的extract_features方法需要int16_t类型
        std::vector<int16_t> audio_int16(audio_float.size());
        for (size_t i = 0; i < audio_float.size(); ++i) {
            float sample = std::max(-1.0f, std::min(1.0f, audio_float[i]));
            audio_int16[i] = static_cast<int16_t>(sample * 32767.0f);
        }
        
        // 提取特征
        std::vector<std::vector<float>> features;
        features = feature_extractor_->extract_features(audio_int16.data(), audio_int16.size(), true);
        
        // 检查特征中是否有NaN值，如果有则替换为0
        for (auto& frame : features) {
            for (auto& value : frame) {
                if (std::isnan(value) || std::isinf(value)) {
                    value = 0.0f;
                }
            }
        }
        
        // 处理音频
        return process_audio_chunk(features);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in processing audio: " << e.what() << std::endl;
        return false;
    }
}

bool Detector::detect_file(
    const std::string& audio_path,
    float& confidence,
    bool debug_mode // 增加 debug_mode 参数
) {
    // 读取音频文件
    std::vector<float> audio_data;
    int source_sample_rate = 0; // 用于存储原始采样率
    try {
        // 使用 load_audio_from_file 加载并重采样
        audio_data = load_audio_from_file(audio_path, source_sample_rate); // 获取原始采样率

        if (audio_data.empty()) {
            std::cerr << "无法加载音频文件或音频数据为空: " << audio_path << std::endl;
            confidence = 0.0f;
            return false;
        }

        // 获取目标采样率
        int target_sample_rate = feature_config_.sample_rate;

        // 调试日志: 打印音频基本信息 (使用debug_mode控制)
        if (debug_mode) {
            float max_amplitude = 0.0f;
            float min_amplitude = 0.0f;
            float avg_amplitude = 0.0f;
            size_t non_zero_count = 0;

            if (!audio_data.empty()) {
                max_amplitude = *std::max_element(audio_data.begin(), audio_data.end());
                min_amplitude = *std::min_element(audio_data.begin(), audio_data.end());
                float sum = std::accumulate(audio_data.begin(), audio_data.end(), 0.0f);
                avg_amplitude = sum / audio_data.size();
                for (const auto& sample : audio_data) {
                    if (std::abs(sample) > 1e-6) {
                        non_zero_count++;
                    }
                }
            }

            std::cout << "[CPP-Debug] 原始采样率: " << source_sample_rate << " Hz" << std::endl;
            std::cout << "[CPP-Debug] 目标采样率: " << target_sample_rate << " Hz" << std::endl;
            std::cout << "[CPP-Debug] 处理后音频长度: " << audio_data.size() << " 样本" << std::endl;
            std::cout << "[CPP-Debug] 处理后音频范围: [" << min_amplitude << ", " << max_amplitude << "]" << std::endl;
            std::cout << "[CPP-Debug] 处理后音频平均值: " << avg_amplitude << std::endl;
            std::cout << "[CPP-Debug] 处理后非零样本数: " << non_zero_count << std::endl;
        }


        // 确保音频长度足够 (例如，至少能提取几帧)
        // int window_size = feature_config_.window_size_ms * target_sample_rate / 1000;
        // int window_stride = feature_config_.window_stride_ms * target_sample_rate / 1000;
        // int min_samples = window_size + 4 * window_stride; // 至少5帧
        // if (audio_data.size() < min_samples) {
        //     if (debug_mode) std::cout << "[CPP-Debug] 音频太短 ( " << audio_data.size() << " < " << min_samples << "), 填充零到 " << min_samples << " 样本" << std::endl;
        //     audio_data.resize(min_samples, 0.0f);
        // }

        // 将 float [-1, 1] 转换为 int16_t [-32767, 32767]
        std::vector<int16_t> audio_data_int16(audio_data.size());
        for (size_t i = 0; i < audio_data.size(); ++i) {
            float scaled = audio_data[i] * 32767.0f;
            scaled = std::max(-32768.0f, std::min(scaled, 32767.0f)); // Clamp
            audio_data_int16[i] = static_cast<int16_t>(scaled);
        }

        // 提取特征
        std::vector<std::vector<float>> features;
        features = feature_extractor_->extract_features(audio_data_int16.data(), audio_data_int16.size(), true); // 传入int16数据

        // 分析特征 (使用debug_mode控制)
        if (debug_mode && !features.empty() && !features[0].empty()) {
            size_t frame_count = features.size();
            size_t feature_dim = features[0].size();

            float features_min = std::numeric_limits<float>::max();
            float features_max = std::numeric_limits<float>::lowest();
            float features_sum = 0.0f;
            size_t features_count = 0;
            size_t nan_count = 0;
            size_t inf_count = 0;

            for (const auto& frame : features) {
                for (const auto& value : frame) {
                    if (std::isnan(value)) {
                        nan_count++;
                        continue;
                    }
                    if (std::isinf(value)) {
                        inf_count++;
                        continue;
                    }
                    features_min = std::min(features_min, value);
                    features_max = std::max(features_max, value);
                    features_sum += value;
                    features_count++;
                }
            }

            float features_avg = features_count > 0 ? features_sum / features_count : 0.0f;

            std::cout << "[CPP-Debug] 特征形状: [" << frame_count << ", " << feature_dim << "]" << std::endl;
            std::cout << "[CPP-Debug] 特征范围: [" << features_min << ", " << features_max << "]" << std::endl;
            std::cout << "[CPP-Debug] 特征平均值: " << features_avg << std::endl;
            std::cout << "[CPP-Debug] 特征NaN数量: " << nan_count << std::endl;
            std::cout << "[CPP-Debug] 特征Inf数量: " << inf_count << std::endl;

            // 打印第一帧的前5个值
            if (frame_count > 0 && feature_dim > 0) {
                 std::cout << "[CPP-Debug] 特征样本(第一帧前5个值): ";
                 for (size_t i = 0; i < std::min(size_t(5), feature_dim); ++i) {
                     std::cout << features[0][i] << (i < std::min(size_t(5), feature_dim) - 1 ? " " : "");
                 }
                 std::cout << std::endl;
            }

            // 检查特征中是否有NaN/Inf值，如果有则替换为0 (模型推理前处理)
             for (auto& frame : features) {
                 for (auto& value : frame) {
                     if (std::isnan(value) || std::isinf(value)) {
                         value = 0.0f;
                     }
                 }
             }
        } else if (features.empty()) {
             std::cerr << "错误: 提取到的特征为空!" << std::endl;
             confidence = 0.0f;
             return false;
        }


        // 使用模型进行推理 (Model类负责处理NaN/Inf)
        float confidence_temp = model_->detect(features); // detect现在只返回置信度
        confidence = confidence_temp; // 直接获取置信度
        bool is_keyword = confidence >= inference_config_.detection_threshold; // 在这里应用阈值

        // 调试日志: 打印模型输出和最终决策 (使用debug_mode控制)
        if (debug_mode) {
             // model_->detect 内部应有打印原始logits和softmax概率的日志
             std::cout << "[CPP-Debug] 模型置信度(关键词): " << confidence << std::endl;
             std::cout << "[CPP-Debug] 检测阈值: " << inference_config_.detection_threshold << std::endl;
             std::cout << "[CPP-Debug] 最终检测结果: " << (is_keyword ? "是" : "不是") << "关键词" << std::endl;
        }

        return is_keyword;
    }
    catch (const std::exception& e) {
        std::cerr << "处理音频文件 " << audio_path << " 时出错: " << e.what() << std::endl;
        confidence = 0.0f;
        return false;
    }
}

void Detector::reset() {
    // 重置老的状态变量
    std::fill(recent_predictions_.begin(), recent_predictions_.end(), 0.0f);
    last_trigger_time_ = std::chrono::steady_clock::now() - trigger_cooldown_;
    
    // 重置后处理器
    post_processor_->reset();
}

float Detector::get_threshold() const {
    return post_processor_->get_threshold();
}

void Detector::set_threshold(float threshold) {
    if (threshold > 0.0f && threshold <= 1.0f) {
        inference_config_.detection_threshold = threshold;
        post_processor_->set_threshold(threshold);
    }
}

const xiaozhi::PostProcessor& Detector::get_post_processor() const {
    return *post_processor_;
}

bool Detector::process_audio_chunk(const std::vector<std::vector<float>>& features) {
    // 创建特征的可修改副本 - Model::detect内部处理NaN/Inf
    // std::vector<std::vector<float>> features_copy = features;

    // 检测
    float confidence = model_->detect(features); // 获取置信度

    // 使用后处理器处理检测结果
    auto result = post_processor_->process(confidence);

    // 保留旧代码的兼容性 - 移到 PostProcessor 内部处理平滑
    // recent_predictions_.pop_front();
    // recent_predictions_.push_back(confidence);

    // 如果检测到唤醒词，调用回调函数
    if (result.is_detected && callback_) {
        callback_(result.smoothed_confidence); // 使用后处理器平滑后的置信度
    }

    return result.is_detected;
}

std::vector<float> Detector::load_audio_from_pcm(const int16_t* audio_data, size_t audio_len) {
    std::vector<float> audio_float(audio_len);
    
    // 将int16_t [-32768, 32767]转换为float [-1.0, 1.0]
    for (size_t i = 0; i < audio_len; ++i) {
        audio_float[i] = static_cast<float>(audio_data[i]) / 32768.0f;
    }
    
    return audio_float;
}

// 从文件加载音频，统一处理格式和重采样
std::vector<float> Detector::load_audio_from_file(const std::string& filepath, int& source_sample_rate) {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    SwrContext* swr_ctx = nullptr;
    int audio_stream_idx = -1;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    std::vector<float> audio_data;
    int target_sample_rate = feature_config_.sample_rate; // 获取目标采样率

    try {
        // 注册所有格式和编解码器 (如果需要)
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
        av_register_all();
#endif

        // 打开输入文件
        if (avformat_open_input(&format_ctx, filepath.c_str(), nullptr, nullptr) != 0) {
            throw std::runtime_error("无法打开音频文件: " + filepath);
        }

        // 获取流信息
        if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
            throw std::runtime_error("无法获取流信息: " + filepath);
        }

        // 找到音频流
        for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
            if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audio_stream_idx = i;
                source_sample_rate = format_ctx->streams[i]->codecpar->sample_rate; // 获取原始采样率
                break;
            }
        }
        if (audio_stream_idx == -1) {
            throw std::runtime_error("找不到音频流: " + filepath);
        }

        // 获取解码器
        AVCodecParameters* codec_params = format_ctx->streams[audio_stream_idx]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
        if (!codec) {
            throw std::runtime_error("不支持的编解码器: " + filepath);
        }

        // 分配和打开解码器上下文
        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx || avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
            throw std::runtime_error("无法创建解码器上下文: " + filepath);
        }
        if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
            throw std::runtime_error("无法打开解码器: " + filepath);
        }

         // 调试日志: 打印原始音频格式
         std::cout << "[CPP-Debug] 原始音频格式: "
                  << "编解码器=" << codec->name
                  << ", 采样率=" << codec_ctx->sample_rate // 使用 codec_ctx->sample_rate
                  << ", 声道=" << codec_ctx->channels
                  << ", 样本格式=" << av_get_sample_fmt_name(codec_ctx->sample_fmt)
                  << std::endl;
         source_sample_rate = codec_ctx->sample_rate; // 更新 source_sample_rate


        // 创建重采样上下文 (如果需要)
        if (source_sample_rate != target_sample_rate || codec_ctx->sample_fmt != AV_SAMPLE_FMT_FLT || codec_ctx->channels != 1) {
            swr_ctx = swr_alloc();
            int64_t in_ch_layout = codec_ctx->channel_layout ? codec_ctx->channel_layout : av_get_default_channel_layout(codec_ctx->channels);
            av_opt_set_int(swr_ctx, "in_channel_layout",    in_ch_layout, 0);
            av_opt_set_int(swr_ctx, "in_sample_rate",       source_sample_rate, 0);
            av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);
            av_opt_set_int(swr_ctx, "out_channel_layout",   AV_CH_LAYOUT_MONO, 0); // 强制单声道
            av_opt_set_int(swr_ctx, "out_sample_rate",      target_sample_rate, 0); // 强制目标采样率
            av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0); // 强制 float 输出
            if (swr_init(swr_ctx) < 0) {
                throw std::runtime_error("无法初始化重采样器");
            }
            std::cout << "[CPP-Debug] 初始化重采样器: " << source_sample_rate << "Hz/" << av_get_sample_fmt_name(codec_ctx->sample_fmt) << "/ch" << codec_ctx->channels
                      << " -> " << target_sample_rate << "Hz/" << av_get_sample_fmt_name(AV_SAMPLE_FMT_FLT) << "/ch1" << std::endl;
        } else {
             std::cout << "[CPP-Debug] 音频格式已满足要求，无需重采样。" << std::endl;
        }


        // 分配帧和包
        frame = av_frame_alloc();
        packet = av_packet_alloc();
        if (!frame || !packet) {
            throw std::runtime_error("无法分配帧或包");
        }

        // 清空 audio_data 以存储新结果
        audio_data.clear();

        // 读取和解码
        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == audio_stream_idx) {
                if (avcodec_send_packet(codec_ctx, packet) >= 0) {
                    while (avcodec_receive_frame(codec_ctx, frame) >= 0) {
                        // 处理解码后的帧
                        const uint8_t **frame_data = (const uint8_t**)frame->data;
                        int frame_samples = frame->nb_samples;
                        std::vector<float> temp_buffer; // 每次循环都创建新的临时缓冲

                        if (swr_ctx) { // 需要重采样
                            // 计算输出样本数
                            int out_samples = av_rescale_rnd(swr_get_delay(swr_ctx, frame->sample_rate) + frame_samples,
                                                        target_sample_rate, frame->sample_rate, AV_ROUND_UP);
                            temp_buffer.resize(out_samples);
                            uint8_t *output_ptr = reinterpret_cast<uint8_t*>(temp_buffer.data());

                            int converted_samples = swr_convert(swr_ctx, &output_ptr, out_samples,
                                                                frame_data, frame_samples);
                            if (converted_samples < 0) {
                                std::cerr << "警告: 重采样失败" << std::endl;
                                temp_buffer.clear(); // 清空以防部分数据被添加
                            } else {
                                temp_buffer.resize(converted_samples); // 调整大小为实际转换的样本数
                            }
                        } else { // 不需要重采样，格式应为 float mono
                            if (codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLT && codec_ctx->channels == 1) {
                                 temp_buffer.resize(frame_samples);
                                 memcpy(temp_buffer.data(), frame_data[0], frame_samples * sizeof(float));
                            } else {
                                 std::cerr << "警告: 音频格式非预期 (需要 float/mono) 且未创建重采样器" << std::endl;
                                 temp_buffer.clear();
                            }
                        }
                         // 将本次处理的有效数据追加到 audio_data
                         if (!temp_buffer.empty()) {
                             std::cout << "[CPP-Debug] swr_convert 输出 " << temp_buffer.size() << " 样本" << std::endl;
                             audio_data.insert(audio_data.end(), temp_buffer.begin(), temp_buffer.end());
                             std::cout << "[CPP-Debug] audio_data 当前大小: " << audio_data.size() << " 样本" << std::endl;
                         }
                    }
                }
            }
            av_packet_unref(packet);
        }

        // 刷新解码器和重采样器
        avcodec_send_packet(codec_ctx, nullptr); // Flush decoder
        while (true) {
             int receive_result = avcodec_receive_frame(codec_ctx, frame);
             if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) {
                 break;
             } else if (receive_result < 0) {
                 std::cerr << "警告: 刷新解码器时接收帧失败" << std::endl;
                 break;
             }
             // 处理解码后的帧 (同上)
             const uint8_t **frame_data = (const uint8_t**)frame->data;
             int frame_samples = frame->nb_samples;
             std::vector<float> temp_buffer;

             if (swr_ctx) {
                 int out_samples = av_rescale_rnd(swr_get_delay(swr_ctx, frame->sample_rate) + frame_samples,
                                             target_sample_rate, frame->sample_rate, AV_ROUND_UP);
                 temp_buffer.resize(out_samples);
                 uint8_t *output_ptr = reinterpret_cast<uint8_t*>(temp_buffer.data());
                 int converted_samples = swr_convert(swr_ctx, &output_ptr, out_samples,
                                                     frame_data, frame_samples);
                 if (converted_samples < 0) {
                     std::cerr << "警告: 重采样失败 (flush decoder)" << std::endl;
                     temp_buffer.clear();
                 } else {
                     temp_buffer.resize(converted_samples);
                 }
             } else {
                  if (codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLT && codec_ctx->channels == 1) {
                      temp_buffer.resize(frame_samples);
                      memcpy(temp_buffer.data(), frame_data[0], frame_samples * sizeof(float));
                  } else {
                      std::cerr << "警告: 音频格式非预期 (flush decoder)" << std::endl;
                      temp_buffer.clear();
                  }
             }
             if (!temp_buffer.empty()) {
                std::cout << "[CPP-Debug] (flush decoder) swr_convert 输出 " << temp_buffer.size() << " 样本" << std::endl;
                audio_data.insert(audio_data.end(), temp_buffer.begin(), temp_buffer.end());
                std::cout << "[CPP-Debug] (flush decoder) audio_data 当前大小: " << audio_data.size() << " 样本" << std::endl;
             }
        }

        if (swr_ctx) { // Flush resampler
            int remaining_samples = 0;
            uint8_t* output_buffer[1]; // 声明为数组
             std::vector<float> temp_buffer;
            do {
                 // 估算一个缓冲区大小，需要足够大以容纳剩余样本
                 const int flush_buffer_size = 4096; // 例如
                 temp_buffer.resize(flush_buffer_size);
                 output_buffer[0] = reinterpret_cast<uint8_t*>(temp_buffer.data());
                 // 调用swr_convert传入nullptr输入来flush
                 remaining_samples = swr_convert(swr_ctx, output_buffer, flush_buffer_size, nullptr, 0);
                 if (remaining_samples < 0) {
                     std::cerr << "警告: 刷新重采样器失败" << std::endl;
                     break;
                 } else if (remaining_samples > 0) {
                     std::cout << "[CPP-Debug] (flush resampler) swr_convert 输出 " << remaining_samples << " 样本" << std::endl;
                     temp_buffer.resize(remaining_samples);
                     audio_data.insert(audio_data.end(), temp_buffer.begin(), temp_buffer.end());
                     std::cout << "[CPP-Debug] (flush resampler) audio_data 当前大小: " << audio_data.size() << " 样本" << std::endl;
                 }
            } while (remaining_samples > 0);
        }


        // 清理资源
        av_frame_free(&frame);
        av_packet_free(&packet);
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&format_ctx);
        if (swr_ctx) swr_free(&swr_ctx);

        if (audio_data.empty()) {
            throw std::runtime_error("解码后未获取到有效音频数据: " + filepath);
        }

        // 确保音频振幅在[-1, 1]范围内 (解码和重采样后可能超出)
        float max_abs = 0.0f;
        for (float sample : audio_data) {
            max_abs = std::max(max_abs, std::abs(sample));
        }
        if (max_abs > 1.0f) {
             std::cout << "[CPP-Debug] 解码/重采样后音频幅度超出 [-1, 1] (max_abs=" << max_abs << ")，进行归一化。" << std::endl;
            float scale = 1.0f / max_abs;
            for (float& sample : audio_data) {
                sample *= scale;
            }
        }

        // 再次检查最终的音频数据长度
        std::cout << "[CPP-Debug] 最终解码/重采样后音频长度: " << audio_data.size() << " 样本" << std::endl;

        return audio_data;

    } catch (const std::exception& e) {
        std::cerr << "加载音频文件时出错: " << e.what() << std::endl;
        // 清理可能已分配的资源
        av_frame_free(&frame);
        av_packet_free(&packet);
        if (codec_ctx) avcodec_free_context(&codec_ctx);
        if (format_ctx) avformat_close_input(&format_ctx);
        if (swr_ctx) swr_free(&swr_ctx);
        return {}; // 返回空向量表示失败
    }
}

float Detector::calculate_audio_energy(const std::vector<float>& audio) {
    if (audio.empty()) {
        return 0.0f;
    }
    
    float sum_squares = 0.0f;
    for (float sample : audio) {
        sum_squares += sample * sample;
    }
    
    return sum_squares / audio.size();
}

float Detector::infer(const std::vector<float>& audio_data) {
    if (audio_data.empty()) {
        std::cerr << "音频数据为空，无法进行推理" << std::endl;
        return 0.0f;
    }

    try {
        // 确保音频振幅在[-1,1]范围内 (通常在加载时已处理)
        // ... (可以移除这里的归一化，或保留作为双重检查) ...

        // 将float[-1,1]转换为int16_t格式
        std::vector<int16_t> audio_int16(audio_data.size());
        for (size_t i = 0; i < audio_data.size(); ++i) {
            float scaled = audio_data[i] * 32767.0f;
            scaled = std::max(-32768.0f, std::min(scaled, 32767.0f));
            audio_int16[i] = static_cast<int16_t>(scaled);
        }

        // 使用特征提取器处理int16_t数据
        std::vector<std::vector<float>> features;
        features = feature_extractor_->extract_features(audio_int16.data(), audio_int16.size(), true);

        // 分析特征 (日志)
        // ... (日志代码不变) ...
        if (features.empty()) {
            std::cerr << "错误: 提取到的特征为空" << std::endl;
            return 0.0f;
        }


        // 使用模型进行检测 - detect返回置信度
        float confidence = model_->detect(features);

        // 打印详细的检测结果 (日志)
        // ... (日志代码不变) ...

        return confidence;
    } catch (const std::exception& e) {
        std::cerr << "推理过程中发生异常: " << e.what() << std::endl;
        return 0.0f;
    }
}

// detect_from_file 函数保持不变或根据需要调整日志
bool Detector::detect_from_file(const std::string& audio_path, float& confidence) {
     int source_sample_rate_ignored = 0; // 忽略原始采样率
    try {
        // 加载音频文件 (使用统一的加载函数)
        std::vector<float> audio = load_audio_from_file(audio_path, source_sample_rate_ignored);

        if (audio.empty()) {
            std::cerr << "无法加载音频文件或音频数据为空: " << audio_path << std::endl;
            confidence = 0.0f; // Reset confidence on error
            return false;      // Return detection failed
        }

        // 分析音频数据 (日志)
        // ... (日志代码不变) ...

        // 使用infer方法进行检测
        confidence = infer(audio);

        // 根据置信度和阈值决定是否检测到关键词
        bool detected = confidence >= inference_config_.detection_threshold;

        // 打印检测结果 (日志)
        // ... (日志代码不变) ...

        return detected;
    } catch (const std::exception& e) {
        std::cerr << "处理音频文件时发生异常: " << e.what() << std::endl;
        confidence = 0.0f;
        return false;
    }
}

std::string Detector::get_features_debug_info(const std::string& audio_path) {
    int source_sample_rate_ignored = 0; // 忽略原始采样率
    try {
        // 加载音频文件
        std::vector<float> audio_data = load_audio_from_file(audio_path, source_sample_rate_ignored);

        // ... (后续代码不变) ...

    } catch (const std::exception& e) {
        return std::string("Error: ") + e.what();
    }
     // ... 确保函数有返回值 ...
     return "Debug info generation failed due to unhandled path."; // 应该不会执行到这里
} 