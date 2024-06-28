#pragma once

#include <vector>
#include <unordered_map>
#include <numeric>
#include <functional>
#include <string>
#include <sstream>
#include <iostream>

#include <cuda_fp16.h>

#include <src/utils/macro.h>
#include <src/utils/string_utils.h>


enum class Device
{
    CPU,
    GPU,
};

enum class DataType
{
    FP32,
    FP16,
    INT8,
    INT32,
    BOOL,
    BYTES,
    UNSUPPORTED
};



template <typename T>
DataType getTensorType()
{
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return DataType::FP32;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return DataType::FP16;
    }
    else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        return DataType::INT32;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return DataType::INT8;
    }
    else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return DataType::BOOL;
    }
    else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
        return DataType::BYTES;
    }
    else {
        return DataType::UNSUPPORTED;
    }
};

template<typename T>
class TensorWrapper;

// 为什么data不能放在tensor里而要放在wrapper里，因为后需要放在map里，不同类型不好操作，模版化在子类好。
struct Tensor
{
    Device device;
    DataType dtype;
    std::vector<int> shape;

    Tensor() = default;

    Tensor(const Device device,
            const DataType dtype,
            const std::vector<int> shape):
            device(device),
            dtype(dtype),
            shape(shape){}
    virtual int size() const {
        if (shape.size() > 0)
        {
            return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
        }
        else
            return 0;
    }

    template<typename T>
    TensorWrapper<T>* as()
    {
        // dynamic_cast
        return static_cast<TensorWrapper<T>*>(this);
    }

    std::string DeviceString() const
    {
        static const std::unordered_map<Device, std::string> devicetring{
            {Device::CPU, "CPU"}, {Device::GPU, "GPU"}};
        return devicetring.at(device);
    }

    virtual std::string toString() const
    {
        std::string device_str = DeviceString();

        static const std::unordered_map<DataType, std::string> type_to_string{
            {DataType::INT8, "INT8"},
            {DataType::INT32,"INT32"},
            {DataType::FP16, "FP16"},
            {DataType::FP32, "FP32"},

        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s]",
                    device_str.c_str(),
                    type_to_string.at(dtype).c_str(),
                    vec2str(shape).c_str());
    }  

};

template <typename T>
class TensorWrapper: public Tensor
{
public:
    T* data;
    TensorWrapper(Device device, DataType dtype, std::vector<int> shape):
    	Tensor(device, dtype, shape){}
    
    TensorWrapper(Device device, DataType dtype, std::vector<int> shape, T* data):
    	Tensor(device, dtype, shape),
    	data(data){
            DataType in_dtype = getTensorType<T>();
            LLM_CHECK_WITH_INFO(in_dtype == dtype, "when build TensorWrapper, the passed in data type should be same as dtype in params");
        }
    
    virtual int size() const {
        if (data == nullptr || shape.size() == 0) {
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
    }

    inline T getVal(int id) const {
        LLM_CHECK(device == Device::CPU);
        return data[id];
    } // only available on CPU by []
    
    inline T getVal() const
    {
        // TODO: add type check, this is very important, because we often naturally access GPU data, which is wrong
        // for example, I am in transpose kernel to use layer_id->getVal<int>(), which is wrong
        LLM_CHECK(device == Device::CPU);
        return getVal(0);
    }

    inline T* getPtr() const {
        //TODO: need some boundry check
        return (T*)data;
    }

    inline T* getPtrByOffset(int offset) const {
        //TODO: need some boundry check
        return (T*)data + offset;
    }

    virtual std::string toString() const
    {
        std::string device_str = DeviceString();

        static const std::unordered_map<DataType, std::string> type_to_string{
            {DataType::INT8, "INT8"},
            {DataType::FP16, "FP16"},
            {DataType::FP32, "FP32"},

        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
                    device_str.c_str(),
                    type_to_string.at(dtype).c_str(),
                    vec2str(shape).c_str(),
                    data);
    }    
};



struct TensorMap
{
    std::unordered_map<std::string, Tensor*> tensor_map_;

    TensorMap() = default;
    TensorMap(std::initializer_list<std::pair<std::string, Tensor*>> tensor_map){
        for (auto& pair : tensor_map) {
            if (isValid(pair.second)) {
                insert(pair.first, pair.second);
            }
            else {
                // std::cout << "this is not a valid tensor, skip to insert into tensormap" << std::endl;
                LLM_CHECK_WITH_INFO(isValid(pair.second),fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
            }
        }
    }
    TensorMap(const std::unordered_map<std::string, Tensor*>& tensor_map) {
        
        for(auto it = tensor_map_.begin(); it != tensor_map_.end(); it++) {
            if (isValid(it->second)) {
                insert(it->first, it->second);
            }
            else {
                // TODO: add a reminder info
            }
        }        
    };

    ~TensorMap(){
        tensor_map_.clear();
    }
    inline size_t size() const
    {
        return tensor_map_.size();
    }
    inline bool isExist(const std::string& key) const
    {
        return tensor_map_.find(key) != tensor_map_.end();
    }

    inline bool isValid(const Tensor* tensor)
    {
        return tensor->size() > 0;
    }
    
    inline void insert(const std::string& key, Tensor* value)
    {
        tensor_map_[key] = value;
    }
    inline Tensor* at(const std::string& key)
    {
         // TODO: add a check to check key is existed
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);
        
    }
    inline Tensor* operator[](const std::string& key)
    {
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map    (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);

    }

    std::vector<std::string> keys() const
    {
        std::vector<std::string> key_names;
        for (auto& kv : tensor_map_) {
            key_names.push_back(kv.first);
        }
        return key_names;
    }
    // 打印出tensormap中的所有key
    std::string toString()
    {
        std::stringstream ss;
        ss << "{";
        std::vector<std::string> key_names = keys();
        for (size_t i = 0; i < tensor_map_.size(); ++i) {
            ss << key_names[i] << ": " << at(key_names[i])->toString();
            if (i < tensor_map_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }

};