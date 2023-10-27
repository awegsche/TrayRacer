//
// Created by andiw on 24/10/2023.
//

#ifndef TRAYRACING_DEVICE_BUFFER_H
#define TRAYRACING_DEVICE_BUFFER_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#ifdef NDEBUG
#define ASSERT(expr, msg)
#else
#define ASSERT(expr, msg) if (expr) {} else { throw std::runtime_error(msg);}
#endif

class DeviceData {

public:

    template<typename T>
    T* get_device_ptr() { return reinterpret_cast<T*>(device_ptr); }

    template<typename T>
    void to_device(T* data, size_t count) {
        size_t size_in_bytes = sizeof(T) * count;
        ASSERT(size_in_bytes != allocated_size, "device data size differs from host data size");

        cudaMemcpy(device_ptr, data, size_in_bytes, cudaMemcpyHostToDevice);
    }

    template<typename T>
    void to_host(T* data, size_t count) {
        size_t size_in_bytes = sizeof(T) * count;
        ASSERT(size_in_bytes != allocated_size, "device data size differs from host data size");

        cudaMemcpy(data, device_ptr, size_in_bytes, cudaMemcpyDeviceToHost);
    }

    void alloc(size_t size) {
        cudaMalloc(&device_ptr, size);
        allocated_size = size;
    }

    void free() {
        cudaFree(device_ptr);
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
    template<typename T>
    void alloc_and_upload(std::vector<T> const& cont) {
        data.alloc(sizeof(T) * cont.size());
        data.to_device(cont.data(), cont.size());
    }

    template<typename Cont>
    void upload(Cont const& cont) {

    }
private:
    DeviceData data;
};


#endif //TRAYRACING_DEVICE_BUFFER_H
