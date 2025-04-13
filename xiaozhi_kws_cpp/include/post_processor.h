/**
 * @file post_processor.h
 * @brief 唤醒词检测后处理器
 */

#ifndef POST_PROCESSOR_H
#define POST_PROCESSOR_H

#include <vector>
#include <deque>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "config.h"

namespace xiaozhi {

/**
 * @brief 检测结果结构体
 */
struct DetectionResult {
    bool is_detected;          // 是否检测到唤醒词
    float confidence;          // 检测置信度
    float smoothed_confidence; // 平滑后的置信度
};

/**
 * @brief 后处理器类
 */
class PostProcessor {
public:
    /**
     * @brief 构造函数
     * 
     * @param config 配置对象
     */
    explicit PostProcessor(const Config& config);

    /**
     * @brief 处理单个检测结果
     * 
     * @param confidence 原始检测置信度
     * @return DetectionResult 处理后的检测结果
     */
    DetectionResult process(float confidence);

    /**
     * @brief 批量处理检测结果
     * 
     * @param confidences 原始检测置信度数组
     * @return std::vector<DetectionResult> 处理后的检测结果数组
     */
    std::vector<DetectionResult> process_batch(const std::vector<float>& confidences);

    /**
     * @brief 重置后处理器状态
     */
    void reset();

    /**
     * @brief 获取检测阈值
     * 
     * @return float 检测阈值
     */
    float get_threshold() const;

    /**
     * @brief 设置检测阈值
     * 
     * @param threshold 检测阈值
     */
    void set_threshold(float threshold);

    /**
     * @brief 设置平滑窗口大小
     * 
     * @param window_size 窗口大小
     */
    void set_smooth_window_size(int window_size);

    /**
     * @brief 设置最小检测间隔
     * 
     * @param interval_ms 检测间隔（毫秒）
     */
    void set_min_detection_interval(int interval_ms);

    /**
     * @brief 应用语音活动检测(VAD)
     * 
     * @param audio_energy 音频能量
     * @return bool 是否为活动语音
     */
    bool apply_vad(float audio_energy);

private:
    // 配置参数
    float detection_threshold_;                  // 检测阈值
    int smooth_window_size_;                     // 平滑窗口大小
    std::chrono::milliseconds min_detection_interval_; // 最小检测间隔
    bool apply_vad_;                             // 是否应用VAD
    float vad_threshold_;                        // VAD阈值
    int vad_window_size_;                        // VAD窗口大小

    // 状态变量
    std::deque<float> confidence_history_;       // 置信度历史
    std::chrono::steady_clock::time_point last_detection_time_; // 上次检测时间
    std::deque<float> vad_energy_history_;       // VAD能量历史

    /**
     * @brief 平滑置信度
     * 
     * @param confidence 原始置信度
     * @return float 平滑后的置信度
     */
    float smooth_confidence(float confidence);

    /**
     * @brief 检查冷却时间
     * 
     * @return bool 是否超过冷却时间
     */
    bool check_cooldown();
};

} // namespace xiaozhi

#endif // POST_PROCESSOR_H 