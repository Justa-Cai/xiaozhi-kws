/**
 * @file xiaozhi_kws.h
 * @brief 小智唤醒词识别SDK头文件
 */

#ifndef XIAOZHI_KWS_H
#define XIAOZHI_KWS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/**
 * 错误码定义
 */
typedef enum {
    XIAOZHI_KWS_SUCCESS = 0,        /**< 成功 */
    XIAOZHI_KWS_ERROR_INIT = -1,    /**< 初始化失败 */
    XIAOZHI_KWS_ERROR_AUDIO = -2,   /**< 音频处理错误 */
    XIAOZHI_KWS_ERROR_MODEL = -3,   /**< 模型错误 */
    XIAOZHI_KWS_ERROR_PARAM = -4,   /**< 参数错误 */
    XIAOZHI_KWS_ERROR_STATE = -5,   /**< 状态错误 */
    XIAOZHI_KWS_ERROR_MEMORY = -6,  /**< 内存错误 */
} XiaozhiKwsError;

/**
 * SDK实例句柄（不透明类型）
 */
typedef struct XiaozhiKwsContext XiaozhiKwsContext;

/**
 * 唤醒回调函数类型
 * @param confidence 置信度 0.0-1.0
 * @param user_data 用户数据指针，由xiaozhi_kws_set_callback传入
 */
typedef void (*XiaozhiKwsCallback)(float confidence, void* user_data);

/**
 * 创建唤醒词识别实例
 * 
 * @param model_path 模型文件路径
 * @param config_path 配置文件路径
 * @param threshold 检测阈值，范围0.0-1.0，若为0则使用配置文件中的阈值
 * @return 返回实例句柄，失败返回NULL
 */
XiaozhiKwsContext* xiaozhi_kws_create(const char* model_path, const char* config_path, float threshold);

/**
 * 销毁唤醒词识别实例
 * 
 * @param context 实例句柄
 */
void xiaozhi_kws_destroy(XiaozhiKwsContext* context);

/**
 * 设置唤醒回调函数
 * 
 * @param context 实例句柄
 * @param callback 回调函数
 * @param user_data 用户数据指针，会在回调时传入
 * @return 返回错误码
 */
XiaozhiKwsError xiaozhi_kws_set_callback(XiaozhiKwsContext* context, 
                                         XiaozhiKwsCallback callback, 
                                         void* user_data);

/**
 * 处理PCM音频数据
 * 
 * @param context 实例句柄
 * @param audio_data PCM音频数据，16KHz采样率，16位有符号整数，单声道
 * @param audio_len 音频数据长度（字节数）
 * @return 返回错误码
 */
XiaozhiKwsError xiaozhi_kws_process_audio(XiaozhiKwsContext* context, 
                                          const int16_t* audio_data, 
                                          size_t audio_len);

/**
 * 重置检测器状态
 * 
 * @param context 实例句柄
 * @return 返回错误码
 */
XiaozhiKwsError xiaozhi_kws_reset(XiaozhiKwsContext* context);

/**
 * 获取SDK版本信息
 * 
 * @return 返回版本字符串
 */
const char* xiaozhi_kws_get_version(void);

/**
 * 检测单个音频文件
 * 
 * @param model_path 模型文件路径
 * @param config_path 配置文件路径
 * @param audio_path 音频文件路径
 * @param threshold 检测阈值，范围0.0-1.0，若为0则使用配置文件中的阈值
 * @param confidence 输出参数，检测置信度
 * @return 返回错误码
 */
XiaozhiKwsError xiaozhi_kws_detect_file(const char* model_path, 
                                        const char* config_path,
                                        const char* audio_path,
                                        float threshold,
                                        float* confidence);

#ifdef __cplusplus
}
#endif

#endif /* XIAOZHI_KWS_H */ 