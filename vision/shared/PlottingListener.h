
#pragma once

#include "Visualizer.h"
#include "Frame.h"

#include <deque>
#include <mutex>
#include <fstream>
#include <condition_variable>
#include <iostream>
#include <iomanip>
#include <chrono>


using namespace affdex;

template<typename T> class PlottingListener {

public:
    PlottingListener(std::ofstream& csv, bool draw_display, bool enable_logging) :
        out_stream_(csv),
        image_data_(),
        process_last_ts_(0),
        draw_display_(draw_display),
        processed_frames_(0),
        logging_enabled_(enable_logging),
        process_fps_(0) {
    }

    int getProcessedFrames() {
        return processed_frames_;
    }

    //Needed to get Image data to create output video
    cv::Mat getImageData() { return image_data_; }

    virtual void outputToFile(const std::map<vision::Id, T>& id_type_map, double time_stamp) = 0;

    virtual void draw(const std::map<vision::Id, T>& id_type_map, const vision::Frame& image) = 0;

    virtual void reset() = 0;

    void drawRecentFrame() {
        if (draw_display_) {
            if (most_recent_frame_.getTimestamp() - time_callback_received_ <= timeout_) {
                draw(latest_data_.second, most_recent_frame_);
            }
            else {
                draw({}, most_recent_frame_);
            }
        }
    }

    void processResults(const vision::Frame& frame) {
        most_recent_frame_ = frame;
        std::lock_guard<std::mutex> lg(mtx);
        if (!results_.empty()) {
            time_callback_received_ = most_recent_frame_.getTimestamp();
            latest_data_ = results_.front();
            results_.pop_front();
            const vision::Frame old_frame = latest_data_.first;
            const auto items = latest_data_.second;
            outputToFile(items, old_frame.getTimestamp());
        }
        drawRecentFrame();
    }

    void startTimer() {
        std::lock_guard<std::mutex> lg(mtx);
        begin_ = std::chrono::steady_clock::now();
    }

    void stopTimer(const Timestamp& timestamp_ms) {
        end_ = std::chrono::steady_clock::now();
        const int diff = timestamp_ms - process_last_ts_;
        if (diff > 0) {
            instantaneous_fps_ = 1000.0 / diff;
        }
        else {
            instantaneous_fps_ = 0;
        }
        process_fps_ = 1000.0 /std::chrono::duration_cast<std::chrono::milliseconds>(end_ - begin_).count();
        process_last_ts_ = timestamp_ms;

    }

    float getProcessingFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return process_fps_;
    }

    float getInstantaneousFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return instantaneous_fps_;
    }

protected:
    using frame_type_id_pair = std::pair<vision::Frame, std::map<vision::Id, T>>;
    std::ofstream& out_stream_;
    Visualizer viz_;
    cv::Mat image_data_;

    std::mutex mtx;

    std::deque<frame_type_id_pair> results_;
    Timestamp process_last_ts_;
    bool draw_display_;
    int processed_frames_;
    bool logging_enabled_;
    frame_type_id_pair latest_data_;
    vision::Frame most_recent_frame_;
    Timestamp time_callback_received_ = 0;
    Duration timeout_ = 500;
    float process_fps_;
    float instantaneous_fps_;
    std::chrono::steady_clock::time_point begin_;
    std::chrono::steady_clock::time_point end_;

};
