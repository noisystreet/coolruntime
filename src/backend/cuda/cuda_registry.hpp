// SPDX-FileCopyrightText: 2026 COOLRuntime contributors
// SPDX-License-Identifier: MIT

/*
 * CUDA 后端的资源注册表（内部头）。
 *
 * 记录本后端分配的设备/主机指针，以及创建的流/事件/模块/内核函数句柄，用于
 * 识别未知指针、重复释放、失效句柄，并在 finalize / deviceReset 时统一清理。
 *
 * 为什么注册表在后端而不是 R 层：宿主并发原语（std::mutex）只允许出现在
 * src/backend 下（见 AGENTS.md、ADR 0003 / ADR 0004）。
 */
#ifndef COOL_BACKEND_CUDA_REGISTRY_HPP
#define COOL_BACKEND_CUDA_REGISTRY_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace cool {
namespace cuda {

/* 设备/主机内存指针 → 字节数。 */
class AllocationRegistry {
   public:
    void addDevice(std::uintptr_t ptr, std::size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        device_[ptr] = size;
    }

    bool hasDevice(std::uintptr_t ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        return device_.count(ptr) != 0;
    }

    void removeDevice(std::uintptr_t ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        device_.erase(ptr);
    }

    void addHost(std::uintptr_t ptr, std::size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        host_[ptr] = size;
    }

    bool hasHost(std::uintptr_t ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        return host_.count(ptr) != 0;
    }

    void removeHost(std::uintptr_t ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        host_.erase(ptr);
    }

    /*
     * 页锁定注册表（coolHostRegister）：与 host_ 分开记录，因为「注册的既有内存」
     * 与「mallocHost 分配的内存」在 unregister / freeHost 上必须是两条不同的路径。
     * 判重与「未注册」由驱动权威判定，这里只用于最终清理。
     */
    void addRegisteredHost(std::uintptr_t ptr, std::size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        registeredHost_[ptr] = size;
    }

    void removeRegisteredHost(std::uintptr_t ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        registeredHost_.erase(ptr);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        device_.clear();
        host_.clear();
        registeredHost_.clear();
    }

   private:
    std::mutex mutex_;
    std::unordered_map<std::uintptr_t, std::size_t> device_;
    std::unordered_map<std::uintptr_t, std::size_t> host_;
    std::unordered_map<std::uintptr_t, std::size_t> registeredHost_;
};

/*
 * 流/事件/模块/函数句柄登记。
 *
 * NULL 表示默认流，不登记。函数句柄归属其模块：卸载模块时级联移除该模块的
 * 所有函数，避免模块卸载后仍能启动已失效的内核。
 */
class HandleRegistry {
   public:
    void addStream(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        streams_.insert(handle);
    }

    bool hasStream(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        return streams_.count(handle) != 0;
    }

    void removeStream(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        streams_.erase(handle);
    }

    void addEvent(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.insert(handle);
    }

    bool hasEvent(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.count(handle) != 0;
    }

    void removeEvent(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.erase(handle);
    }

    void addModule(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        modules_.insert(handle);
    }

    bool hasModule(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        return modules_.count(handle) != 0;
    }

    /* 移除模块并级联移除其拥有的所有函数句柄。 */
    void removeModule(std::uintptr_t handle) {
        std::lock_guard<std::mutex> lock(mutex_);
        modules_.erase(handle);
        for (auto it = functions_.begin(); it != functions_.end();) {
            if (it->second == handle) {
                it = functions_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void addFunction(std::uintptr_t func, std::uintptr_t module) {
        std::lock_guard<std::mutex> lock(mutex_);
        functions_[func] = module;
    }

    bool hasFunction(std::uintptr_t func) {
        std::lock_guard<std::mutex> lock(mutex_);
        return functions_.count(func) != 0;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        streams_.clear();
        events_.clear();
        modules_.clear();
        functions_.clear();
    }

   private:
    std::mutex mutex_;
    std::unordered_set<std::uintptr_t> streams_;
    std::unordered_set<std::uintptr_t> events_;
    std::unordered_set<std::uintptr_t> modules_;
    std::unordered_map<std::uintptr_t, std::uintptr_t> functions_;
};

/*
 * 进程级单例。cuda::init() 会显式预构造二者，以保证析构顺序晚于
 * bootstrap 的 atexit 清理。
 */
AllocationRegistry& allocations();
HandleRegistry& handles();

/* CUstream / CUevent 等不透明句柄 → 可比较的整数地址。 */
std::uintptr_t handleAddr(const void* handle);

}  // namespace cuda
}  // namespace cool

#endif /* COOL_BACKEND_CUDA_REGISTRY_HPP */
