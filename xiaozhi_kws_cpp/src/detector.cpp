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
#include "mfcc_features.h"

// 音频处理库
#include <sndfile.h>
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
static bool has_registered_custom_ops = false;

Detector::Detector(const std::string& model_path, 
                 const std::string& config_path, 
                 float threshold)
    : last_trigger_time_(std::chrono::steady_clock::now() - trigger_cooldown_) {
    
    try {
        // 加载配置
        config_parser_ = std::make_unique<ConfigParser>(config_path);
        inference_config_ = config_parser_->get_inference_config();
        
        // 设置阈值
        if (threshold > 0.0f) {
            inference_config_.detection_threshold = threshold;
        } else {
            // 如果没有指定阈值，使用训练时的默认值
            inference_config_.detection_threshold = 0.5f;
        }
        
        // 创建特征提取器
        auto feature_config = config_parser_->get_feature_config();
        feature_extractor_ = std::make_unique<FeatureExtractor>(feature_config);
        
        // 创建MFCC特征提取器
        mfcc_features_ = std::make_unique<MFCCFeatures>();
        
        // 创建模型
        auto model_config = config_parser_->get_model_config();
        model_ = std::make_unique<Model>(model_path, model_config);
        
        // 初始化ONNX Runtime环境和会话
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "xiaozhi_kws_detector");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // 创建会话
        session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        
        // 获取输入输出信息
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 获取输入节点名称
        size_t num_input_nodes = session_->GetInputCount();
        if (num_input_nodes > 0) {
            const char* input_name = session_->GetInputNameAllocated(0, allocator).release();
            input_node_names_.push_back(input_name);
        }
        
        // 获取输出节点名称
        size_t num_output_nodes = session_->GetOutputCount();
        if (num_output_nodes > 0) {
            const char* output_name = session_->GetOutputNameAllocated(0, allocator).release();
            output_node_names_.push_back(output_name);
        }
        
        // 初始化预测队列
        recent_predictions_.resize(inference_config_.smoothing_window, 0.0f);
        
        // 初始化xiaozhi配置
        config_.load_from_file(config_path);
        
        // 设置后处理器配置 - 使用与训练模型一致的阈值
        config_.detection_threshold = inference_config_.detection_threshold;
        config_.smooth_window_size = 5;  // 使用默认平滑窗口大小
        config_.apply_vad = true;  // 启用VAD（语音活动检测）
        
        // 创建后处理器
        post_processor_ = std::make_unique<xiaozhi::PostProcessor>(config_);
        
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
        
        // 检查VAD（如果启用）
        if (config_.apply_vad && !post_processor_->apply_vad(audio_energy)) {
            return false; // 非语音区域，不需要继续处理
        }
        
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
    float& confidence
) {
    // 读取音频文件
    std::vector<float> audio_data;
    try {
        // 使用load_audio_from_file代替不存在的load_audio函数
        audio_data = load_audio_from_file(audio_path);
        
        if (audio_data.empty()) {
            std::cerr << "无法加载音频文件或音频数据为空: " << audio_path << std::endl;
            return false;
        }
        
        // 获取采样率
        int sample_rate = config_parser_->get_feature_config().sample_rate;
        
        // 分析音频数据
        float max_amplitude = 0.0f;
        float min_amplitude = 0.0f;
        float avg_amplitude = 0.0f;
        size_t non_zero_count = 0;
        
        max_amplitude = *std::max_element(audio_data.begin(), audio_data.end());
        min_amplitude = *std::min_element(audio_data.begin(), audio_data.end());
        
        float sum = 0.0f;
        for (const auto& sample : audio_data) {
            sum += sample;
            if (std::abs(sample) > 1e-6) {
                non_zero_count++;
            }
        }
        avg_amplitude = audio_data.size() > 0 ? sum / audio_data.size() : 0.0f;
        
        std::cout << "[CPP-Debug] 音频长度: " << audio_data.size() << std::endl;
        std::cout << "[CPP-Debug] 音频范围: [" << min_amplitude << ", " << max_amplitude << "]" << std::endl;
        std::cout << "[CPP-Debug] 音频平均值: " << avg_amplitude << std::endl;
        std::cout << "[CPP-Debug] 非零样本数: " << non_zero_count << std::endl;
        
        // 检查音频长度是否足够，确保有足够的数据进行特征提取
        const int MIN_AUDIO_LENGTH = 16000; // 至少1秒的音频
        if (audio_data.size() < MIN_AUDIO_LENGTH) {
            std::cout << "音频太短，填充至少" << MIN_AUDIO_LENGTH << "样本" << std::endl;
            audio_data.resize(MIN_AUDIO_LENGTH, 0.0f);
        }
        
        // 标准化音频 - 与Python实现一致
        if (max_amplitude > 1e-6f) {
            // 确保音频在[-1,1]范围内
            float max_abs = std::max(std::abs(max_amplitude), std::abs(min_amplitude));
            if (max_abs > 1.0f) {
                for (auto& sample : audio_data) {
                    sample /= max_abs;
                }
                std::cout << "[CPP-Debug] 已标准化音频到[-1,1]范围" << std::endl;
            }
        }
        
        // 转换音频格式从float到int16
        std::vector<int16_t> audio_data_int16(audio_data.size());
        for (size_t i = 0; i < audio_data.size(); ++i) {
            // 将[-1,1]转换为int16范围，避免溢出
            float scaled = audio_data[i] * 32767.0f;
            // 限制在int16范围内
            scaled = std::max(-32768.0f, std::min(scaled, 32767.0f));
            audio_data_int16[i] = static_cast<int16_t>(scaled);
        }
        
        // 提取特征
        std::vector<std::vector<float>> features;
        features = feature_extractor_->extract_features(audio_data_int16.data(), audio_data_int16.size(), true);
        
        // 分析特征
        if (!features.empty() && !features[0].empty()) {
            size_t frame_count = features.size();
            size_t feature_dim = features[0].size();
            
            float features_min = std::numeric_limits<float>::max();
            float features_max = std::numeric_limits<float>::lowest();
            float features_sum = 0.0f;
            size_t features_count = 0;
            size_t nan_count = 0;
            
            for (const auto& frame : features) {
                for (const auto& value : frame) {
                    if (std::isnan(value)) {
                        nan_count++;
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
            
            // 打印第一帧的前5个值
            if (frame_count > 0 && feature_dim >= 5) {
                std::cout << "[CPP-Debug] 特征样本(第一帧前5个值): [";
                for (size_t i = 0; i < 5; ++i) {
                    std::cout << features[0][i];
                    if (i < 4) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            
            // 检查特征中是否有NaN值，如果有则替换为0
            for (auto& frame : features) {
                for (auto& value : frame) {
                    if (std::isnan(value) || std::isinf(value)) {
                        value = 0.0f;
                    }
                }
            }
        }
        
        // 使用模型进行推理
        auto result = model_->detect(features, inference_config_.detection_threshold);
        bool is_keyword = result.first;
        confidence = result.second;
        
        return is_keyword;
    }
    catch (const std::exception& e) {
        std::cerr << "处理音频文件时出错: " << e.what() << std::endl;
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
    // 创建特征的可修改副本
    std::vector<std::vector<float>> features_copy = features;
    
    // 检查特征中是否有NaN值，如果有则替换为0
    for (auto& frame : features_copy) {
        for (auto& value : frame) {
            if (std::isnan(value) || std::isinf(value)) {
                value = 0.0f;
            }
        }
    }
    
    // 检测
    auto [is_keyword, confidence] = model_->detect(features_copy, inference_config_.detection_threshold);
    
    // 使用后处理器处理检测结果
    auto result = post_processor_->process(confidence);
    
    // 保留旧代码的兼容性
    recent_predictions_.pop_front();
    recent_predictions_.push_back(confidence);
    
    // 如果检测到唤醒词，调用回调函数
    if (result.is_detected && callback_) {
        callback_(result.smoothed_confidence);
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

std::vector<float> Detector::load_audio_from_file(const std::string& filepath) {
    std::vector<float> audio_data;
    
    try {
        // 检查文件扩展名
        std::string extension = filepath.substr(filepath.find_last_of(".") + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        // 如果是MP3文件，则使用专用的MP3加载函数
        if (extension == "mp3") {
            std::cout << "[CPP-Debug] 检测到MP3文件，使用FFmpeg处理: " << filepath << std::endl;
            try {
                return load_audio_from_mp3(filepath);
            } catch (const std::exception& mp3_error) {
                std::cerr << "MP3解码失败，尝试用libsndfile打开: " << mp3_error.what() << std::endl;
                // 如果MP3加载失败，继续尝试用libsndfile
            }
        }
        
        // 非MP3文件使用libsndfile处理
        SF_INFO sf_info;
        memset(&sf_info, 0, sizeof(sf_info));
        
        // 打开音频文件
        SNDFILE* file = sf_open(filepath.c_str(), SFM_READ, &sf_info);
        if (!file) {
            std::cerr << "无法打开音频文件: " << filepath << " - " << sf_strerror(NULL) << std::endl;
            return audio_data;
        }
        
        // 确保文件最后关闭
        std::unique_ptr<SNDFILE, decltype(&sf_close)> file_closer(file, &sf_close);
        
        // 读取音频数据
        std::vector<float> temp_buffer(sf_info.frames * sf_info.channels);
        sf_count_t count = sf_readf_float(file, temp_buffer.data(), sf_info.frames);
        
        if (count <= 0) {
            std::cerr << "读取音频数据失败: " << filepath << std::endl;
            return audio_data;
        }
        
        // 采样率
        int target_sample_rate = config_parser_->get_feature_config().sample_rate;
        int source_sample_rate = sf_info.samplerate;
        
        std::cout << "[CPP-Debug] 原始音频采样率: " << source_sample_rate << "Hz, 目标采样率: " << target_sample_rate << "Hz" << std::endl;
        
        // 如果是立体声，转换为单声道
        if (sf_info.channels > 1) {
            std::vector<float> mono_buffer(count);
            for (sf_count_t i = 0; i < count; ++i) {
                float sum = 0.0f;
                for (int ch = 0; ch < sf_info.channels; ++ch) {
                    sum += temp_buffer[i * sf_info.channels + ch];
                }
                mono_buffer[i] = sum / sf_info.channels;
            }
            temp_buffer = std::move(mono_buffer);
        }
        
        // 如果采样率不匹配，执行重采样
        if (source_sample_rate != target_sample_rate) {
            std::cout << "[CPP-Debug] 需要重采样，从 " << source_sample_rate << "Hz 到 " << target_sample_rate << "Hz" << std::endl;
            
            // 创建重采样器
            SwrContext* swr_ctx = swr_alloc_set_opts(
                nullptr,                // 上下文（自动创建）
                AV_CH_LAYOUT_MONO,      // 输出通道布局（单声道）
                AV_SAMPLE_FMT_FLT,      // 输出样本格式（浮点）
                target_sample_rate,     // 输出采样率
                AV_CH_LAYOUT_MONO,      // 输入通道布局（单声道）
                AV_SAMPLE_FMT_FLT,      // 输入样本格式（浮点）
                source_sample_rate,     // 输入采样率
                0,                      // 日志偏移
                nullptr                 // 日志上下文
            );
            
            if (!swr_ctx) {
                std::cerr << "无法创建重采样上下文" << std::endl;
                return audio_data;
            }
            
            // 初始化重采样器
            if (swr_init(swr_ctx) < 0) {
                swr_free(&swr_ctx);
                std::cerr << "无法初始化重采样上下文" << std::endl;
                return audio_data;
            }
            
            // 计算输出采样数量
            int output_samples = static_cast<int>(
                av_rescale_rnd(temp_buffer.size(), target_sample_rate, source_sample_rate, AV_ROUND_UP)
            );
            
            // 分配输出缓冲区
            std::vector<float> resampled_buffer(output_samples);
            
            // 执行重采样 - 修复类型转换问题
            uint8_t* output_ptr = reinterpret_cast<uint8_t*>(resampled_buffer.data());
            const uint8_t* input_ptr = reinterpret_cast<const uint8_t*>(temp_buffer.data());
            
            int ret = swr_convert(
                swr_ctx,
                &output_ptr,
                output_samples,
                &input_ptr,
                temp_buffer.size()
            );
            
            // 释放重采样器
            swr_free(&swr_ctx);
            
            if (ret < 0) {
                std::cerr << "重采样失败" << std::endl;
                return audio_data;
            }
            
            // 移动结果到输出向量
            audio_data.resize(ret);
            std::copy(resampled_buffer.begin(), resampled_buffer.begin() + ret, audio_data.begin());
            
            std::cout << "[CPP-Debug] 重采样完成，实际输出采样数: " << ret << std::endl;
        } else {
            // 采样率相同，直接使用
            audio_data = temp_buffer;
        }
        
        return audio_data;
    } catch (const std::exception& e) {
        std::cerr << "加载音频文件时出错: " << e.what() << std::endl;
        return audio_data;
    }
}

// 使用FFmpeg处理MP3文件
std::vector<float> Detector::load_audio_from_mp3(const std::string& audio_path) {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    SwrContext* swr_ctx = nullptr;
    int audio_stream_idx = -1;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    std::vector<float> audio_data;
    int target_sample_rate = config_parser_->get_feature_config().sample_rate;
    
    try {
        // 注册所有格式和编解码器（对较新版本不需要）
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
        av_register_all();
#endif
        
        // 打开输入文件
        AVDictionary* open_opts = nullptr;
        // 尝试设置更宽松的打开选项，特别是对于MPEG ADTS
        av_dict_set(&open_opts, "analyzeduration", "2000000", 0);  // 增加分析时间
        av_dict_set(&open_opts, "probesize", "1000000", 0);  // 增加探测大小
        
        int ret = avformat_open_input(&format_ctx, audio_path.c_str(), nullptr, &open_opts);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            
            // 释放字典
            av_dict_free(&open_opts);
            
            throw std::runtime_error(std::string("无法打开MP3文件: ") + audio_path + " - " + errbuf);
        }
        
        // 释放打开选项字典
        av_dict_free(&open_opts);
        
        // 输出更多文件信息用于调试
        std::cout << "[CPP-Debug] 音频文件: " << audio_path << std::endl;
        std::cout << "[CPP-Debug] 输入格式: " << (format_ctx->iformat ? format_ctx->iformat->name : "未知") << std::endl;
        std::cout << "[CPP-Debug] 流数量: " << format_ctx->nb_streams << std::endl;
        
        // 获取流信息
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "error_concealment", "1", 0);
        av_dict_set(&opts, "max_delay", "5000000", 0);  // 增加最大延迟时间
        
        ret = avformat_find_stream_info(format_ctx, &opts);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            
            // 释放字典
            av_dict_free(&opts);
            
            throw std::runtime_error(std::string("无法获取流信息: ") + audio_path + " - " + errbuf);
        }
        
        // 释放字典
        av_dict_free(&opts);
        
        // 找到音频流
        for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
            if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audio_stream_idx = i;
                break;
            }
        }
        
        if (audio_stream_idx == -1) {
            throw std::runtime_error("找不到音频流: " + audio_path);
        }
        
        // 获取解码器参数
        AVCodecParameters* codec_params = format_ctx->streams[audio_stream_idx]->codecpar;
        
        // 获取解码器
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(59, 0, 100)
        const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
#else
        const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
#endif
        if (!codec) {
            throw std::runtime_error("不支持的编解码器: " + audio_path);
        }
        
        // 分配解码器上下文
        codec_ctx = avcodec_alloc_context3(codec);
        if (!codec_ctx) {
            throw std::runtime_error("无法分配解码器上下文: " + audio_path);
        }
        
        // 复制解码器参数
        ret = avcodec_parameters_to_context(codec_ctx, codec_params);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            throw std::runtime_error(std::string("无法复制解码器参数: ") + audio_path + " - " + errbuf);
        }
        
        // 打开解码器
        ret = avcodec_open2(codec_ctx, codec, nullptr);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            throw std::runtime_error(std::string("无法打开解码器: ") + audio_path + " - " + errbuf);
        }
        
        // 输出音频格式信息
        std::cout << "[CPP-Debug] 音频格式: "
                 << "编解码器=" << codec->name
                 << ", 采样率=" << codec_ctx->sample_rate
                 << ", 声道=" << codec_ctx->channels
                 << ", 样本格式=" << av_get_sample_fmt_name(codec_ctx->sample_fmt)
                 << std::endl;
        
        // 创建重采样上下文
        swr_ctx = swr_alloc();
        if (!swr_ctx) {
            throw std::runtime_error("无法分配重采样上下文");
        }
        
        // 获取声道布局
        int64_t in_channel_layout = codec_ctx->channel_layout;
        if (in_channel_layout == 0) {
            // 如果没有指定声道布局，则根据通道数估算
            in_channel_layout = av_get_default_channel_layout(codec_ctx->channels);
        }
        
        // 设置重采样参数
        av_opt_set_int(swr_ctx, "in_channel_layout", in_channel_layout, 0);
        av_opt_set_int(swr_ctx, "out_channel_layout", AV_CH_LAYOUT_MONO, 0);  // 输出单声道
        av_opt_set_int(swr_ctx, "in_sample_rate", codec_ctx->sample_rate, 0);
        av_opt_set_int(swr_ctx, "out_sample_rate", target_sample_rate, 0);
        av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);
        av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0);  // 输出为浮点格式
        
        // 初始化重采样
        ret = swr_init(swr_ctx);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            throw std::runtime_error(std::string("无法初始化重采样器: ") + errbuf);
        }
        
        // 分配帧和包
        frame = av_frame_alloc();
        packet = av_packet_alloc();
        
        if (!frame || !packet) {
            throw std::runtime_error("无法分配帧或包: " + audio_path);
        }
        
        // 临时缓冲区用于重采样
        std::vector<float> resampled_buffer;
        
        // 开始解码
        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == audio_stream_idx) {
                // 发送包到解码器
                ret = avcodec_send_packet(codec_ctx, packet);
                if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                    char errbuf[256];
                    av_strerror(ret, errbuf, sizeof(errbuf));
                    std::cerr << "警告: 发送包到解码器失败: " << errbuf << std::endl;
                    av_packet_unref(packet);
                    continue;
                }
                
                // 接收解码后的帧
                while (true) {
                    ret = avcodec_receive_frame(codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        char errbuf[256];
                        av_strerror(ret, errbuf, sizeof(errbuf));
                        std::cerr << "警告: 接收帧失败: " << errbuf << std::endl;
                        break;
                    }
                    
                    // 提取音频数据
                    int samples = frame->nb_samples;
                    
                    // 计算重采样后的样本数
                    int out_samples = av_rescale_rnd(
                        samples, 
                        target_sample_rate, 
                        codec_ctx->sample_rate, 
                        AV_ROUND_UP);
                    
                    // 分配临时缓冲区
                    resampled_buffer.resize(out_samples);
                    
                    // 创建输出缓冲区
                    float* output_buffer[1] = { resampled_buffer.data() };
                    
                    // 执行重采样
                    int resampled = swr_convert(
                        swr_ctx,
                        (uint8_t**)output_buffer, out_samples,
                        (const uint8_t**)frame->data, samples);
                    
                    if (resampled < 0) {
                        char errbuf[256];
                        av_strerror(resampled, errbuf, sizeof(errbuf));
                        std::cerr << "警告: 重采样失败: " << errbuf << std::endl;
                        continue;
                    }
                    
                    // 将重采样后的数据添加到结果
                    size_t old_size = audio_data.size();
                    audio_data.resize(old_size + resampled);
                    std::copy(resampled_buffer.begin(), 
                             resampled_buffer.begin() + resampled, 
                             audio_data.begin() + old_size);
                }
            }
            av_packet_unref(packet);
        }
        
        // 刷新解码器
        avcodec_send_packet(codec_ctx, nullptr);
        while (true) {
            int receive_result = avcodec_receive_frame(codec_ctx, frame);
            if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) {
                break;
            } else if (receive_result < 0) {
                break;
            }
            
            // 提取音频数据
            int samples = frame->nb_samples;
            
            // 计算重采样后的样本数
            int out_samples = av_rescale_rnd(
                samples, 
                target_sample_rate, 
                codec_ctx->sample_rate, 
                AV_ROUND_UP);
            
            // 分配临时缓冲区
            resampled_buffer.resize(out_samples);
            
            // 创建输出缓冲区
            float* output_buffer[1] = { resampled_buffer.data() };
            
            // 执行重采样
            int resampled = swr_convert(
                swr_ctx,
                (uint8_t**)output_buffer, out_samples,
                (const uint8_t**)frame->data, samples);
            
            if (resampled < 0) {
                std::cerr << "Warning: Error resampling audio" << std::endl;
                continue;
            }
            
            // 将重采样后的数据添加到结果
            size_t old_size = audio_data.size();
            audio_data.resize(old_size + resampled);
            std::copy(resampled_buffer.begin(), 
                     resampled_buffer.begin() + resampled, 
                     audio_data.begin() + old_size);
        }
        
        // 清理资源
        if (frame) av_frame_free(&frame);
        if (packet) av_packet_free(&packet);
        if (codec_ctx) avcodec_free_context(&codec_ctx);
        if (format_ctx) avformat_close_input(&format_ctx);
        if (swr_ctx) swr_free(&swr_ctx);
        
        if (audio_data.empty()) {
            throw std::runtime_error("No audio data decoded from file: " + audio_path);
        }
        
        // 确保音频振幅在[-1, 1]范围内
        float max_abs = 0.0f;
        for (float sample : audio_data) {
            max_abs = std::max(max_abs, std::abs(sample));
        }
        
        if (max_abs > 1.0f) {
            float scale = 1.0f / max_abs;
            for (float& sample : audio_data) {
                sample *= scale;
            }
        }
        
        // 确保音频长度足够，Python实现至少需要能提取5帧特征
        auto feature_config = config_parser_->get_feature_config();
        int window_size = feature_config.window_size_ms * target_sample_rate / 1000;
        int window_stride = feature_config.window_stride_ms * target_sample_rate / 1000;
        int min_samples = window_size + 4 * window_stride;
        
        if (audio_data.size() < min_samples) {
            // 填充音频到最小长度
            audio_data.resize(min_samples, 0.0f);
        }
        
        return audio_data;
        
    } catch (const std::exception& e) {
        // 记录错误
        std::cerr << "MP3处理失败: " << e.what() << std::endl;
        
        // 清理资源
        if (frame) av_frame_free(&frame);
        if (packet) av_packet_free(&packet);
        if (swr_ctx) swr_free(&swr_ctx);
        if (codec_ctx) avcodec_free_context(&codec_ctx);
        if (format_ctx) avformat_close_input(&format_ctx);
        
        // 重新抛出异常
        throw;
    }
    
    // 资源清理
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (swr_ctx) swr_free(&swr_ctx);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) avformat_close_input(&format_ctx);
    
    if (audio_data.empty()) {
        throw std::runtime_error("MP3解码后未获取到有效音频数据");
    }
    
    // 输出音频信息（不使用debug_标志）
    {
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        float sum = 0.0f;
        int nonzero_count = 0;
        
        for (const float& sample : audio_data) {
            min_val = std::min(min_val, sample);
            max_val = std::max(max_val, sample);
            sum += sample;
            if (std::abs(sample) > 1e-6) {
                nonzero_count++;
            }
        }
        
        float avg = audio_data.empty() ? 0.0f : sum / audio_data.size();
        
        std::cout << "[CPP-Debug] 音频长度: " << audio_data.size() << " 样本" << std::endl;
        std::cout << "[CPP-Debug] 音频范围: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "[CPP-Debug] 音频平均值: " << avg << std::endl;
        std::cout << "[CPP-Debug] 非零样本数: " << nonzero_count << std::endl;
    }
    
    // 确保足够数据用于特征提取
    if (audio_data.size() < 160) {
        std::cerr << "警告: 音频太短，添加零填充" << std::endl;
        audio_data.resize(160, 0.0f);  // 添加零填充确保至少有160个样本
    }
    
    return audio_data;
}

// 使用libsndfile库加载WAV文件
std::vector<float> Detector::load_audio_from_wav(const std::string& audio_path) {
    SF_INFO sfinfo;
    memset(&sfinfo, 0, sizeof(sfinfo));
    
    SNDFILE* sndfile = sf_open(audio_path.c_str(), SFM_READ, &sfinfo);
    if (!sndfile) {
        throw std::runtime_error("Failed to open WAV file: " + audio_path);
    }
    
    // 检查采样率
    int target_sample_rate = config_parser_->get_feature_config().sample_rate;
    if (sfinfo.samplerate != target_sample_rate) {
        sf_close(sndfile);
        throw std::runtime_error("Sample rate mismatch: " + std::to_string(sfinfo.samplerate) +
                                 " (expected " + std::to_string(target_sample_rate) + ")");
    }
    
    // 分配内存
    std::vector<float> audio(sfinfo.frames * sfinfo.channels);
    
    // 读取音频数据
    sf_count_t frames_read = sf_readf_float(sndfile, audio.data(), sfinfo.frames);
    
    // 关闭文件
    sf_close(sndfile);
    
    if (frames_read != sfinfo.frames) {
        throw std::runtime_error("Failed to read WAV file completely: " + audio_path);
    }
    
    // 如果是多声道，转换为单声道
    if (sfinfo.channels > 1) {
        std::vector<float> mono(sfinfo.frames);
        for (sf_count_t i = 0; i < sfinfo.frames; ++i) {
            float sum = 0.0f;
            for (int c = 0; c < sfinfo.channels; ++c) {
                sum += audio[i * sfinfo.channels + c];
            }
            mono[i] = sum / sfinfo.channels;
        }
        return mono;
    }
    
    return audio;
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
        // 确保音频振幅在[-1,1]范围内
        std::vector<float> normalized_audio = audio_data;
        float max_abs = 0.0f;
        for (const auto& sample : audio_data) {
            max_abs = std::max(max_abs, std::abs(sample));
        }
        
        if (max_abs > 1.0f) {
            for (auto& sample : normalized_audio) {
                sample /= max_abs;
            }
            std::cout << "已标准化音频到[-1,1]范围" << std::endl;
        }
        
        // 将float[-1,1]转换为int16_t格式，与Python实现对齐
        std::vector<int16_t> audio_int16(normalized_audio.size());
        for (size_t i = 0; i < normalized_audio.size(); ++i) {
            // 将[-1,1]转换为int16范围，避免溢出
            float scaled = normalized_audio[i] * 32767.0f;
            // 限制在int16范围内
            scaled = std::max(-32768.0f, std::min(scaled, 32767.0f));
            audio_int16[i] = static_cast<int16_t>(scaled);
        }
        
        // 使用特征提取器处理int16_t数据 - 与Python实现保持一致
        std::vector<std::vector<float>> features;
        features = feature_extractor_->extract_features(audio_int16.data(), audio_int16.size(), true);
        
        // 分析特征
        if (!features.empty() && !features[0].empty()) {
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            float sum = 0.0f;
            size_t count = 0;
            
            for (const auto& frame : features) {
                for (const auto& value : frame) {
                    if (std::isnan(value) || std::isinf(value)) {
                        continue;
                    }
                    min_val = std::min(min_val, value);
                    max_val = std::max(max_val, value);
                    sum += value;
                    count++;
                }
            }
            
            float avg = count > 0 ? sum / count : 0.0f;
            std::cout << "特征统计: " 
                  << "帧数=" << features.size() 
                  << ", 维度=" << features[0].size()
                  << ", 范围=[" << min_val << ", " << max_val << "]"
                  << ", 平均=" << avg << std::endl;
            
            // 打印第一帧的前几个特征值作为示例
            std::cout << "特征样本(第一帧前5个值): ";
            for (size_t i = 0; i < std::min(size_t(5), features[0].size()); ++i) {
                std::cout << features[0][i] << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "错误: 提取到的特征为空" << std::endl;
            return 0.0f;
        }
        
        // 使用模型进行检测 - 确保模型使用正确的概率索引
        auto [is_keyword, confidence] = model_->detect(features, inference_config_.detection_threshold);
        
        // 打印详细的检测结果
        std::cout << "模型检测结果: " << (is_keyword ? "是关键词" : "不是关键词") 
              << ", 置信度 = " << confidence 
              << ", 阈值 = " << inference_config_.detection_threshold << std::endl;
        
        return confidence;
    } catch (const std::exception& e) {
        std::cerr << "推理过程中发生异常: " << e.what() << std::endl;
        return 0.0f;
    }
}

// 从文件推理
bool Detector::detect_from_file(const std::string& audio_path, float& confidence) {
    try {
        // 加载音频文件
        std::vector<float> audio = load_audio_from_file(audio_path);
        
        if (audio.empty()) {
            std::cerr << "无法加载音频文件或音频数据为空: " << audio_path << std::endl;
            confidence = 0.0f;
            return false;
        }
        
        // 分析音频数据
        float max_amplitude = 0.0f;
        float min_amplitude = 0.0f;
        float avg_amplitude = 0.0f;
        size_t non_zero_count = 0;
        
        max_amplitude = *std::max_element(audio.begin(), audio.end());
        min_amplitude = *std::min_element(audio.begin(), audio.end());
        
        float sum = 0.0f;
        for (const auto& sample : audio) {
            sum += sample;
            if (std::abs(sample) > 1e-6) {
                non_zero_count++;
            }
        }
        avg_amplitude = audio.size() > 0 ? sum / audio.size() : 0.0f;
        
        std::cout << "音频统计: " 
                << "长度=" << audio.size() 
                << ", 采样率=" << config_parser_->get_feature_config().sample_rate
                << ", 范围=[" << min_amplitude << ", " << max_amplitude << "]"
                << ", 平均=" << avg_amplitude 
                << ", 非零样本数=" << non_zero_count << std::endl;
        
        // 使用infer方法进行检测
        confidence = infer(audio);
        
        // 根据置信度和阈值决定是否检测到关键词
        bool detected = confidence >= inference_config_.detection_threshold;
        
        std::cout << "检测结果: " << (detected ? "检测到唤醒词" : "未检测到唤醒词") 
                << ", 置信度=" << confidence 
                << std::endl;
        
        return detected;
    } catch (const std::exception& e) {
        std::cerr << "处理音频文件时发生异常: " << e.what() << std::endl;
        confidence = 0.0f;
        return false;
    }
}

std::string Detector::get_features_debug_info(const std::string& audio_path) {
    try {
        // 加载音频文件
        std::vector<float> audio_data = load_audio_from_file(audio_path);
        
        if (audio_data.empty()) {
            return "无法加载音频文件或音频数据为空";
        }
        
        // 获取采样率
        int sample_rate = config_parser_->get_feature_config().sample_rate;
        
        // 分析音频数据
        float max_amplitude = *std::max_element(audio_data.begin(), audio_data.end());
        float min_amplitude = *std::min_element(audio_data.begin(), audio_data.end());
        
        float sum = 0.0f;
        size_t non_zero_count = 0;
        for (const auto& sample : audio_data) {
            sum += sample;
            if (std::abs(sample) > 1e-6) {
                non_zero_count++;
            }
        }
        float avg_amplitude = audio_data.size() > 0 ? sum / audio_data.size() : 0.0f;
        
        std::stringstream ss;
        ss << "音频信息:\n"
           << "  长度: " << audio_data.size() << " 样本\n"
           << "  采样率: " << sample_rate << " Hz\n"
           << "  范围: [" << min_amplitude << ", " << max_amplitude << "]\n"
           << "  平均值: " << avg_amplitude << "\n"
           << "  非零样本: " << non_zero_count << "\n";
        
        // 转换为int16_t格式
        std::vector<int16_t> audio_data_int16(audio_data.size());
        for (size_t i = 0; i < audio_data.size(); ++i) {
            float sample = std::max(-1.0f, std::min(1.0f, audio_data[i]));
            audio_data_int16[i] = static_cast<int16_t>(sample * 32767.0f);
        }
        
        // 提取特征
        std::vector<std::vector<float>> features;
        features = feature_extractor_->extract_features(audio_data_int16.data(), audio_data_int16.size(), true);
        
        if (!features.empty() && !features[0].empty()) {
            size_t frame_count = features.size();
            size_t feature_dim = features[0].size();
            
            ss << "特征信息:\n"
               << "  帧数: " << frame_count << "\n"
               << "  维度: " << feature_dim << "\n";
            
            if (frame_count > 0 && feature_dim > 0) {
                ss << "  第一帧样本 (前5个): ";
                for (size_t i = 0; i < std::min(size_t(5), feature_dim); ++i) {
                    ss << features[0][i] << " ";
                }
                ss << "\n";
            }
        }
        
        return ss.str();
    } catch (const std::exception& e) {
        return std::string("Error: ") + e.what();
    }
} 