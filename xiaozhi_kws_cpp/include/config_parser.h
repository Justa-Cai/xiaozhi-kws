/**
 * @file config_parser.h
 * @brief 配置解析器
 */

#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <map>
#include <vector>
#include <unordered_map>

/**
 * 特征提取配置结构
 */
struct FeatureConfig {
    int sample_rate;
    int window_size_ms;
    int window_stride_ms;
    int n_mfcc;
    int n_fft;
    int n_mels;
    bool use_delta;
    bool use_delta2;
    double preemphasis_coeff = 0.97;
};

/**
 * 模型配置结构
 */
struct ModelConfig {
    std::string type;
    int input_dim;
    int hidden_dim;
    int num_layers;
    float dropout;
};

/**
 * 推理配置结构
 */
struct InferenceConfig {
    float detection_threshold;
    int smoothing_window;
};

/**
 * 配置结构
 */
struct Config {
    int sample_rate;        /**< 采样率 */
    int frame_length;       /**< 帧长度（样本数）*/
    int frame_shift;        /**< 帧移（样本数）*/
    int fft_size;           /**< FFT大小 */
    int num_filters;        /**< Mel滤波器数量 */
    int num_mfcc;           /**< MFCC系数数量 */
    double low_freq;        /**< 最低频率 */
    double high_freq;       /**< 最高频率 */
    bool use_delta;         /**< 是否使用一阶差分 */
    bool use_delta2;        /**< 是否使用二阶差分 */
    double pre_emphasis;    /**< 预加重系数 */
    double dither;          /**< 抖动系数 */
};

/**
 * 配置解析器类
 */
class ConfigParser {
public:
    /**
     * 默认构造函数
     */
    ConfigParser() = default;
    
    /**
     * 构造函数
     * 
     * @param config_path 配置文件路径
     */
    ConfigParser(const std::string& config_path);
    
    /**
     * 获取特征配置
     * 
     * @return 特征配置
     */
    FeatureConfig get_feature_config() const;
    
    /**
     * 获取模型配置
     * 
     * @return 模型配置
     */
    ModelConfig get_model_config() const;
    
    /**
     * 获取推理配置
     * 
     * @return 推理配置
     */
    InferenceConfig get_inference_config() const;
    
    /**
     * 获取关键词列表
     * 
     * @return 关键词列表
     */
    std::vector<std::string> get_keywords() const;
    
    /**
     * 从文件解析配置
     * 
     * @param filepath 配置文件路径
     * @return 解析后的配置
     */
    Config parse_config(const std::string& filepath);
    
    /**
     * 从JSON字符串解析配置
     * 
     * @param json_str JSON字符串
     * @return 解析后的配置
     */
    Config parse_config_from_string(const std::string& json_str);

private:
    /**
     * 解析YAML内容
     * 
     * @param content YAML内容
     */
    void parse_yaml(const std::string& content);
    
    /**
     * 获取字符串配置项
     * 
     * @param key 键
     * @param default_value 默认值
     * @return 配置值
     */
    std::string get_string(const std::string& key, const std::string& default_value) const;
    
    /**
     * 获取整数配置项
     * 
     * @param key 键
     * @param default_value 默认值
     * @return 配置值
     */
    int get_int(const std::string& key, int default_value) const;
    
    /**
     * 获取浮点数配置项
     * 
     * @param key 键
     * @param default_value 默认值
     * @return 配置值
     */
    float get_float(const std::string& key, float default_value) const;
    
    /**
     * 获取布尔配置项
     * 
     * @param key 键
     * @param default_value 默认值
     * @return 配置值
     */
    bool get_bool(const std::string& key, bool default_value) const;
    
    /**
     * 填充默认配置
     * 
     * @return 默认配置
     */
    Config get_default_config();
    
    /**
     * 配置映射
     */
    std::unordered_map<std::string, std::string> config_map_;
};

#endif /* CONFIG_PARSER_H */ 