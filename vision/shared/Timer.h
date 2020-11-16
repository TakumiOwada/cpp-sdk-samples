#pragma once
#include <chrono>

struct Timer{
    int count = 0;
    long total_elapsed_time = 0;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    void startTimer() {
        begin = std::chrono::steady_clock::now();
        ++count;
    }

    void stopTimer() {
        end = std::chrono::steady_clock::now();
        total_elapsed_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
};