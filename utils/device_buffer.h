//
// Created by andiw on 24/10/2023.
//

#ifndef TRAYRACING_DEVICE_BUFFER_H
#define TRAYRACING_DEVICE_BUFFER_H

#include "optix_types.h"
#include <cuda_runtime.h>
#include <optix_lib.h>
#include <stdexcept>
#include <vector>

#ifdef NDEBUG
#define ASSERT(expr, msg)
#else
#define ASSERT(expr, msg) if (expr) {} else { throw std::runtime_error(std::format("{} | {}:{}", msg, __FILE__, __LINE__));}
#endif

class DeviceData {

public:

    template<typename T>
    T* get_device_ptr() { return reinterpret_cast<T*>(device_ptr); }

    CUdeviceptr get_CUdevice_ptr() { return reinterpret_cast<CUdeviceptr>(device_ptr); }

    template<typename T>
    void to_device(T* data, size_t count) {
        size_t size_in_bytes = sizeof(T) * count;
        spdlog::info("size_in_bytes: {}", size_in_bytes);
        spdlog::info("allocated: {}", allocated_size);
        ASSERT(size_in_bytes == allocated_size, "device data size differs from host data size");

        CU_CHECK(cudaMemcpy(device_ptr, data, size_in_bytes, cudaMemcpyHostToDevice));
    }

    template<typename T>
    void to_host(T* data) {
        CU_CHECK(cudaMemcpy(data, device_ptr, allocated_size, cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void resize(size_t size) {
        if (size == allocated_size)
            return;
        free();
        alloc<T>(size);
    }

    size_t size() const { return allocated_size; }

    template<typename T>
    void alloc(size_t size) {
        allocated_size = size * sizeof(T);
        CU_CHECK(cudaMalloc(&device_ptr, allocated_size));
    }

    void free() {
        CU_CHECK(cudaFree(device_ptr));
        device_ptr = nullptr;
        allocated_size = 0;
    }

    ~DeviceData() {
        if (device_ptr)
            cudaFree(device_ptr);

    }
private:
    void* device_ptr = nullptr;
    size_t allocated_size = 0;
};

class DeviceBuffer {
public:
    template<typename T>
    void alloc_and_upload(std::vector<T> const& cont) {
        data.alloc<T>(cont.size());
        data.to_device(cont.data(), cont.size());
    }


    // fall through
    //
    template<typename T>
    void alloc(size_t size) {
        data.alloc<T>(size);
    }

    template<typename T>
    void to_device(T* data, size_t count) {
        this->data.to_device(data, count);
    }

    template<typename T>
    void download(std::vector<T>& data) {
        data.resize(this->data.size() / sizeof(T));
        this->data.to_host(data.data());
    }

    template<typename T>
    T* get_device_ptr() { return data.get_device_ptr<T>(); }

    CUdeviceptr get_CUdevice_ptr() { return data.get_CUdevice_ptr(); }

public:
    DeviceData data;
};


#endif //TRAYRACING_DEVICE_BUFFER_H
