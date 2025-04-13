#include "post_processor.h"

namespace xiaozhi {

PostProcessor::PostProcessor(const InferenceConfig& config)
    : detection_threshold_(config.detection_threshold),
      smooth_window_size_(config.smoothing_window),
      min_detection_interval_(std::chrono::milliseconds(1000)),
      last_detection_time_(std::chrono::steady_clock::now() - min_detection_interval_) {
    
    // 初始化置信度历史队列
    confidence_history_.resize(smooth_window_size_, 0.0f);
    
    // 初始化VAD能量历史队列
    // VAD 相关设置可以从InferenceConfig获取，如果存在的话，否则使用默认值
    // apply_vad_(config.apply_vad), // 假设InferenceConfig有 apply_vad
    // vad_threshold_(config.vad_threshold), // 假设InferenceConfig有 vad_threshold
    // vad_window_size_(config.vad_window_size), // 假设InferenceConfig有 vad_window_size
}

DetectionResult PostProcessor::process(float confidence) {
    // 添加到历史队列
    confidence_history_.pop_front();
    confidence_history_.push_back(confidence);
    
    // 平滑置信度
    float smoothed_confidence = smooth_confidence(confidence);
    
    // 检查是否超过阈值
    bool threshold_passed = (smoothed_confidence > detection_threshold_);
    
    // 检查冷却时间
    bool cooldown_passed = check_cooldown();
    
    // 确定是否检测到唤醒词
    bool is_detected = threshold_passed && cooldown_passed;
    
    // 如果检测到，更新最后检测时间
    if (is_detected) {
        last_detection_time_ = std::chrono::steady_clock::now();
    }
    
    // 返回检测结果
    DetectionResult result;
    result.is_detected = is_detected;
    result.confidence = confidence;
    result.smoothed_confidence = smoothed_confidence;
    
    return result;
}

std::vector<DetectionResult> PostProcessor::process_batch(const std::vector<float>& confidences) {
    std::vector<DetectionResult> results;
    results.reserve(confidences.size());
    
    for (float conf : confidences) {
        results.push_back(process(conf));
    }
    
    return results;
}

void PostProcessor::reset() {
    // 重置置信度历史
    std::fill(confidence_history_.begin(), confidence_history_.end(), 0.0f);
    
    // 重置VAD能量历史
    // VAD 相关设置可以从InferenceConfig获取，如果存在的话，否则使用默认值
    // apply_vad_(config.apply_vad), // 假设InferenceConfig有 apply_vad
    // vad_energy_history_.fill(0.0f), // 假设InferenceConfig有 vad_energy_history_
    
    // 重置最后检测时间
    last_detection_time_ = std::chrono::steady_clock::now() - min_detection_interval_;
}

float PostProcessor::get_threshold() const {
    return detection_threshold_;
}

void PostProcessor::set_threshold(float threshold) {
    if (threshold > 0.0f && threshold <= 1.0f) {
        detection_threshold_ = threshold;
    }
}

void PostProcessor::set_smooth_window_size(int window_size) {
    if (window_size > 0) {
        smooth_window_size_ = window_size;
        
        // 调整历史队列大小
        confidence_history_.resize(smooth_window_size_, 0.0f);
    }
}

void PostProcessor::set_min_detection_interval(int interval_ms) {
    if (interval_ms > 0) {
        min_detection_interval_ = std::chrono::milliseconds(interval_ms);
    }
}

bool PostProcessor::apply_vad(float audio_energy) {
    // 如果不启用VAD，直接返回true
    // VAD 相关设置可以从InferenceConfig获取，如果存在的话，否则使用默认值
    // apply_vad_(config.apply_vad), // 假设InferenceConfig有 apply_vad
    // vad_threshold_(config.vad_threshold), // 假设InferenceConfig有 vad_threshold
    // vad_window_size_(config.vad_window_size), // 假设InferenceConfig有 vad_window_size
    // 如果InferenceConfig有 apply_vad，则使用它，否则使用默认值
    bool apply_vad = false; // 假设InferenceConfig没有 apply_vad
    if (apply_vad) {
        // 添加到能量历史队列
        vad_energy_history_.pop_front();
        vad_energy_history_.push_back(audio_energy);
        
        // 计算平均能量
        float avg_energy = std::accumulate(vad_energy_history_.begin(), 
                                           vad_energy_history_.end(), 0.0f) / 
                           vad_energy_history_.size();
        
        // 判断是否为活动语音
        return (avg_energy > vad_threshold_);
    }
    return true; // 如果InferenceConfig没有 apply_vad，则默认返回true
}

float PostProcessor::smooth_confidence(float confidence) {
    // 使用移动平均平滑置信度
    float sum = std::accumulate(confidence_history_.begin(), 
                               confidence_history_.end(), 0.0f);
    
    return sum / confidence_history_.size();
}

bool PostProcessor::check_cooldown() {
    // 检查是否超过冷却时间
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = current_time - last_detection_time_;
    
    return (elapsed >= min_detection_interval_);
}

} // namespace xiaozhi 