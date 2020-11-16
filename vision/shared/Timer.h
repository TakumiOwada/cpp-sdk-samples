#pragma once
#include <chrono>

struct Timer{
    int count_ = 0;
    long total_elapsed_time_ = 0;
    std::chrono::steady_clock::time_point begin_;
    std::chrono::steady_clock::time_point end_;

    void startTimer() {
        begin_ = std::chrono::steady_clock::now();
        ++count_;
    }

    void stopTimer() {
        end_ = std::chrono::steady_clock::now();
        total_elapsed_time_ += std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count();
    }
};