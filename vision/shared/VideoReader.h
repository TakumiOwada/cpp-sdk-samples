#pragma once

#include <Core.h>
#include "ProgressBar.h"

#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>

class VideoReader {
public:
    explicit VideoReader(const boost::filesystem::path& file_path, float sampling_frame_rate = 0);

    bool GetFrame(cv::Mat& bgr_frame, affdex::Timestamp& timestamp_ms);
    bool GetFrameData(cv::Mat& bgr_frame, affdex::Timestamp& timestamp_ms);

    uint64_t TotalFrames() const;

private:
    cv::VideoCapture cap_;
    affdex::Timestamp last_timestamp_ms_;
    float sampling_frame_rate_;

    uint64_t total_frames_ = 0;
    uint64_t current_frame_ = 0;
    std::unique_ptr<ProgressBar> frame_progress_;
};