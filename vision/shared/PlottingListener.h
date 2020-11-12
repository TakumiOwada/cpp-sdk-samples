
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
    PlottingListener(std::ofstream& csv, bool draw_display, bool enable_logging, bool write_video) :
        out_stream_(csv),
        image_data_(),
        draw_display_(draw_display),
        processed_frames_(0),
        logging_enabled_(enable_logging),
        write_video_(write_video){
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
        if (most_recent_frame_.getTimestamp() - time_callback_received_ <= timeout_) {
            draw(latest_data_.second, most_recent_frame_);
        }
        else {
            draw({}, most_recent_frame_);
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
        if(write_video_ || draw_display_) {
            drawRecentFrame();
        }
    }

protected:
    using frame_type_id_pair = std::pair<vision::Frame, std::map<vision::Id, T>>;
    std::ofstream& out_stream_;
    Visualizer viz_;
    cv::Mat image_data_;

    std::mutex mtx;

    std::deque<frame_type_id_pair> results_;
    bool draw_display_;
    int processed_frames_;
    bool logging_enabled_;
    frame_type_id_pair latest_data_;
    vision::Frame most_recent_frame_;
    Timestamp time_callback_received_ = 0;
    Duration timeout_ = 500;
    bool write_video_;

};
