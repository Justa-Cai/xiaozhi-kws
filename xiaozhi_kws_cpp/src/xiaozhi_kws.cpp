#include "xiaozhi_kws.h"
#include "detector.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

// SDK版本
#define XIAOZHI_KWS_VERSION "1.0.0"

// 错误信息
static char last_error_message[512] = {0};

// 全局回调函数
static XiaozhiKwsCallback global_callback = nullptr;
static void* global_user_data = nullptr;

// 互斥锁，保护检测器实例映射表
static std::mutex detector_mutex;

// 实现结构体
struct XiaozhiKwsContext {
    // 只是一个标识符，实际内容在detectors映射表中
};

// 保存所有创建的检测器实例
static std::unordered_map<XiaozhiKwsContext*, std::unique_ptr<Detector>> detectors;

// 设置错误信息
static void set_error_message(const char* message) {
    std::strncpy(last_error_message, message, sizeof(last_error_message) - 1);
    last_error_message[sizeof(last_error_message) - 1] = '\0';
}

// C API实现回调包装函数
static void detection_callback_wrapper(float confidence) {
    if (global_callback) {
        global_callback(confidence, global_user_data);
    }
}

// C API实现
extern "C" {

XiaozhiKwsContext* xiaozhi_kws_create(const char* model_path, const char* config_path, float threshold) {
    try {
        std::lock_guard<std::mutex> lock(detector_mutex);
        
        // 创建新的检测器实例
        auto detector = std::make_unique<Detector>(model_path, config_path, threshold);
        
        // 创建一个新的上下文
        auto* context = new XiaozhiKwsContext();
        
        // 保存到映射表
        detectors[context] = std::move(detector);
        
        return context;
    } catch (const std::exception& e) {
        set_error_message(e.what());
        return nullptr;
    }
}

void xiaozhi_kws_destroy(XiaozhiKwsContext* context) {
    if (!context) return;
    
    try {
        std::lock_guard<std::mutex> lock(detector_mutex);
        detectors.erase(context);
        delete context;
    } catch (const std::exception& e) {
        set_error_message(e.what());
    }
}

XiaozhiKwsError xiaozhi_kws_set_callback(XiaozhiKwsContext* context, 
                                        XiaozhiKwsCallback callback, 
                                        void* user_data) {
    if (!context) {
        set_error_message("Invalid context");
        return XIAOZHI_KWS_ERROR_PARAM;
    }
    
    try {
        std::lock_guard<std::mutex> lock(detector_mutex);
        
        auto it = detectors.find(context);
        if (it == detectors.end()) {
            set_error_message("Invalid context");
            return XIAOZHI_KWS_ERROR_PARAM;
        }
        
        // 保存回调函数和用户数据
        global_callback = callback;
        global_user_data = user_data;
        
        // 设置检测器回调
        if (callback) {
            it->second->set_callback(detection_callback_wrapper);
        }
        
        return XIAOZHI_KWS_SUCCESS;
    } catch (const std::exception& e) {
        set_error_message(e.what());
        return XIAOZHI_KWS_ERROR_STATE;
    }
}

XiaozhiKwsError xiaozhi_kws_process_audio(XiaozhiKwsContext* context, 
                                        const int16_t* audio_data, 
                                        size_t audio_len) {
    if (!context) {
        set_error_message("Invalid context");
        return XIAOZHI_KWS_ERROR_PARAM;
    }
    
    if (!audio_data || audio_len == 0) {
        set_error_message("Invalid audio data");
        return XIAOZHI_KWS_ERROR_AUDIO;
    }
    
    try {
        std::lock_guard<std::mutex> lock(detector_mutex);
        
        auto it = detectors.find(context);
        if (it == detectors.end()) {
            set_error_message("Invalid context");
            return XIAOZHI_KWS_ERROR_STATE;
        }
        
        bool result = it->second->process_audio(audio_data, audio_len);
        return XIAOZHI_KWS_SUCCESS;
    } catch (const std::exception& e) {
        set_error_message(e.what());
        return XIAOZHI_KWS_ERROR_AUDIO;
    }
}

XiaozhiKwsError xiaozhi_kws_reset(XiaozhiKwsContext* context) {
    if (!context) {
        set_error_message("Invalid context");
        return XIAOZHI_KWS_ERROR_PARAM;
    }
    
    try {
        std::lock_guard<std::mutex> lock(detector_mutex);
        
        auto it = detectors.find(context);
        if (it == detectors.end()) {
            set_error_message("Invalid context");
            return XIAOZHI_KWS_ERROR_STATE;
        }
        
        it->second->reset();
        return XIAOZHI_KWS_SUCCESS;
    } catch (const std::exception& e) {
        set_error_message(e.what());
        return XIAOZHI_KWS_ERROR_STATE;
    }
}

XiaozhiKwsError xiaozhi_kws_detect_file(const char* model_path, 
                                        const char* config_path,
                                        const char* audio_path,
                                        float threshold,
                                        float* confidence) {
    if (!model_path || !config_path || !audio_path || !confidence) {
        set_error_message("Invalid parameters");
        return XIAOZHI_KWS_ERROR_PARAM;
    }
    
    try {
        // 临时创建检测器
        Detector detector(model_path, config_path, threshold);
        
        // 检测文件
        bool result = detector.detect_file(audio_path, *confidence);
        
        return XIAOZHI_KWS_SUCCESS;
    } catch (const std::exception& e) {
        set_error_message(e.what());
        return XIAOZHI_KWS_ERROR_AUDIO;
    }
}

const char* xiaozhi_kws_get_version(void) {
    return XIAOZHI_KWS_VERSION;
}

} // extern "C" 