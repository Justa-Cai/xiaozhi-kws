#include "model.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <cstring>

// ONNX Runtime头文件
#include <onnxruntime/onnxruntime_cxx_api.h>

Model::Model(const std::string& model_path, const ModelConfig& config, float threshold)
    : config_(config), detection_threshold_(threshold) {
    
    // 加载模型
    load_model(model_path);
}

Model::~Model() = default;

void Model::load_model(const std::string& model_path) {
    try {
        std::cout << "加载ONNX模型: " << model_path << std::endl;
        
        // 创建ONNX运行时环境
        env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "xiaozhi_kws");
        
        // 会话选项
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // 创建会话
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);
        
        // 创建内存信息
        memory_info_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        
        // 获取输入输出信息
        Ort::AllocatorWithDefaultOptions allocator;
        
        // 获取输入信息 - 适应ONNX Runtime API变化
        size_t num_input_nodes = session_->GetInputCount();
        if (num_input_nodes == 0) {
            throw std::runtime_error("模型没有输入节点");
        }
        
        const char* input_name = session_->GetInputNameAllocated(0, allocator).release();
        input_names_.push_back(input_name);
        
        // 获取输出信息 - 适应ONNX Runtime API变化
        size_t num_output_nodes = session_->GetOutputCount();
        if (num_output_nodes == 0) {
            throw std::runtime_error("模型没有输出节点");
        }
        
        const char* output_name = session_->GetOutputNameAllocated(0, allocator).release();
        output_names_.push_back(output_name);
        
        std::cout << "模型输入节点名称: " << input_names_[0] << std::endl;
        std::cout << "模型输出节点名称: " << output_names_[0] << std::endl;
        
        // 获取输入和输出形状
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_shape_ = input_tensor_info.GetShape();
        
        auto output_type_info = session_->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_shape_ = output_tensor_info.GetShape();
        
        std::cout << "输入形状: [";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i];
            if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "输出形状: [";
        for (size_t i = 0; i < output_shape_.size(); ++i) {
            std::cout << output_shape_[i];
            if (i < output_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "ONNX模型加载成功" << std::endl;
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime error: ") + e.what());
    }
}

std::vector<float> Model::forward(const std::vector<std::vector<float>>& features) {
    try {
        // 打印传入特征的统计信息以便调试
        if (!features.empty() && !features[0].empty()) {
            float min_val = std::numeric_limits<float>::max();
            float max_val = std::numeric_limits<float>::lowest();
            float sum = 0.0f;
            size_t count = 0;
            
            for (const auto& frame : features) {
                for (const auto& value : frame) {
                    min_val = std::min(min_val, value);
                    max_val = std::max(max_val, value);
                    sum += value;
                    count++;
                }
            }
            
            float avg = count > 0 ? sum / count : 0.0f;
            std::cout << "模型输入特征统计: "
                      << "Frames=" << features.size() 
                      << ", Dims=" << features[0].size()
                      << ", Range=[" << min_val << ", " << max_val << "]"
                      << ", Avg=" << avg << std::endl;
            
            // 打印第一帧的前几个特征值作为示例
            std::cout << "特征样本(第一帧前5个值): ";
            for (size_t i = 0; i < std::min(size_t(5), features[0].size()); ++i) {
                std::cout << features[0][i] << " ";
            }
            std::cout << std::endl;
        }
        
        // 获取特征维度
        size_t batch_size = 1;
        size_t time_steps = features.size();
        size_t feature_dim = features[0].size();
        
        // 调整输入形状
        input_shape_[0] = batch_size;
        input_shape_[1] = time_steps;
        input_shape_[2] = feature_dim;
        
        std::cout << "ONNX输入调整后的形状: [" << batch_size << ", " 
                  << time_steps << ", " << feature_dim << "]" << std::endl;
        
        // 展平特征
        std::vector<float> input_tensor_values = flatten_features(features);
        
        // 创建输入tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *memory_info_, 
            input_tensor_values.data(), 
            input_tensor_values.size(),
            input_shape_.data(), 
            input_shape_.size()
        );
        
        std::cout << "执行ONNX模型推理..." << std::endl;
        
        // 运行推理
        std::vector<Ort::Value> output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, 
            input_names_.data(), 
            &input_tensor, 
            1, 
            output_names_.data(), 
            1
        );
        
        // 获取输出
        const float* output_data = output_tensors[0].GetTensorData<float>();
        size_t output_size = output_shape_[1]; // 假设输出形状为[batch_size, num_classes]
        
        std::cout << "ONNX输出大小: " << output_size << std::endl;
        
        // 复制输出数据
        std::vector<float> output(output_data, output_data + output_size);
        
        return output;
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime inference error: ") + e.what());
    }
}

float Model::detect(const std::vector<std::vector<float>>& features) {
    // 检查特征有效性
    if (features.empty() || features[0].empty()) {
        return 0.0f; // 返回0置信度
    }
    
    // 执行前向传播
    std::vector<float> outputs = forward(features);
    
    // 记录原始logits输出
    std::cout << "[CPP-Debug] 原始logits输出: [";
    for (size_t i = 0; i < outputs.size(); ++i) {
        std::cout << outputs[i];
        if (i < outputs.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // 完全按照PyTorch torch.nn.functional.softmax实现
    // 1. 找出最大值用于数值稳定性
    float max_val = *std::max_element(outputs.begin(), outputs.end());
    
    // 2. 计算exp(x - max_val) 并累加
    std::vector<float> exp_outputs(outputs.size());
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < outputs.size(); ++i) {
        // 添加溢出保护
        float val = outputs[i] - max_val;
        // 限制exp的输入范围，避免溢出
        val = std::max(-80.0f, std::min(val, 80.0f));
        exp_outputs[i] = std::exp(val);
        sum_exp += exp_outputs[i];
    }
    
    // 3. 归一化，确保精度，并保护除零
    if (sum_exp < 1e-10f) {
        sum_exp = 1e-10f;  // 避免除零
    }
    
    for (size_t i = 0; i < exp_outputs.size(); ++i) {
        exp_outputs[i] /= sum_exp;
    }
    
    // 获取关键词类别的概率（获取第二个类别的概率）
    // 注意：确保我们使用与Python版本相同的索引逻辑
    // 在Python版本中，索引1对应关键词，索引0对应非关键词
    // 我们需要匹配此逻辑以确保一致的结果
    float confidence = (outputs.size() > 1) ? exp_outputs[1] : 0.0f;
    
    // 调试输出
    std::cout << "[CPP-Debug] Softmax后概率: [";
    for (size_t i = 0; i < exp_outputs.size(); ++i) {
        std::cout << exp_outputs[i];
        if (i < exp_outputs.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "[CPP-Debug] 置信度(索引1/关键词): " << confidence << std::endl; // 不再打印阈值
    
    return confidence; // 只返回置信度
}

std::vector<float> Model::flatten_features(const std::vector<std::vector<float>>& features) {
    // 确保输入有效
    if (features.empty() || features[0].empty()) {
        return {};
    }
    
    // 获取维度
    size_t time_steps = features.size();
    size_t feature_dim = features[0].size();
    
    // PyTorch使用行优先顺序，与C++的嵌套循环布局相同
    // 但为了确保与PyTorch的内存布局完全一致，我们使用显式的展平逻辑
    std::vector<float> flattened(time_steps * feature_dim, 0.0f);
    
    // 按照行优先顺序复制数据
    for (size_t t = 0; t < time_steps; ++t) {
        // 确保当前帧的特征维度正确
        if (features[t].size() != feature_dim) {
            std::cerr << "警告: 特征帧 " << t << " 的维度 " << features[t].size() 
                      << " 与预期的 " << feature_dim << " 不符" << std::endl;
            // 使用零填充或截断
            for (size_t f = 0; f < feature_dim && f < features[t].size(); ++f) {
                flattened[t * feature_dim + f] = features[t][f];
            }
        } else {
            // 正常复制
            std::copy(features[t].begin(), features[t].end(), 
                     flattened.begin() + t * feature_dim);
        }
    }
    
    // 检查NaN和无穷大
    for (size_t i = 0; i < flattened.size(); ++i) {
        if (std::isnan(flattened[i]) || std::isinf(flattened[i])) {
            flattened[i] = 0.0f;  // 替换异常值
        }
    }
    
    return flattened;
} 