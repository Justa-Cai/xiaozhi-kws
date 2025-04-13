/**
 * @file model.h
 * @brief 模型推理类
 */

#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <memory>
#include "config_parser.h" // 引入配置解析器头文件，包含ModelConfig定义

// 前向声明，避免暴露ONNX运行时的细节
namespace Ort {
    class Session;
    class Env;
    struct MemoryInfo;
}

// 使用config_parser.h中定义的ModelConfig结构体

/**
 * 模型推理类
 */
class Model {
public:
    /**
     * 构造函数
     * 
     * @param model_path 模型文件路径
     * @param config 模型配置
     */
    Model(const std::string& model_path, const ModelConfig& config);
    
    /**
     * 析构函数
     */
    ~Model();

    /**
     * 前向传播
     * 
     * @param features 输入特征，shape=(batch_size, time_steps, features)
     * @return 模型输出，shape=(batch_size, num_classes)
     */
    std::vector<float> forward(const std::vector<std::vector<float>>& features);
    
    /**
     * 检测是否为关键词
     * 
     * @param features 输入特征
     * @param threshold 检测阈值
     * @return pair<是否为关键词, 置信度>
     */
    std::pair<bool, float> detect(
        const std::vector<std::vector<float>>& features, 
        float threshold);

private:
    // 模型配置
    ModelConfig config_;
    
    // ONNX运行时
    std::shared_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    
    // 输入输出信息
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<int64_t> input_shape_;
    std::vector<int64_t> output_shape_;
    
    // 模型加载函数
    void load_model(const std::string& model_path);
    
    // 转换数据格式
    std::vector<float> flatten_features(const std::vector<std::vector<float>>& features);
};

#endif /* MODEL_H */ 