#include "VideoReader.h"

#include <set>
#include <stdexcept>

using namespace std;

VideoReader::VideoReader(const boost::filesystem::path& file_path, int sampling_frame_rate)
    : sampling_frame_rate_(sampling_frame_rate) {

    // Initialize so that with sampling, we always process the first frame.
    last_timestamp_ms_ = sampling_frame_rate == 0 ? -1 :(0 - 1000 / sampling_frame_rate);

    set<boost::filesystem::path> SUPPORTED_EXTS = {
        // Videos
        boost::filesystem::path(".avi"), boost::filesystem::path(".mov"), boost::filesystem::path(".flv"),
        boost::filesystem::path(".webm"), boost::filesystem::path(".wmv"), boost::filesystem::path(".mp4"),
    };

    boost::filesystem::path ext = file_path.extension();
    if (SUPPORTED_EXTS.find(ext) == SUPPORTED_EXTS.end()) {
        throw runtime_error("Unsupported file extension: " + ext.string());
    }

    cap_.open(file_path.string());
    if (!cap_.isOpened()) {
        throw runtime_error("Error opening video/image file: " + file_path.string());
    }

    total_frames_ = cap_.get(CV_CAP_PROP_FRAME_COUNT);
    frame_progress_ = std::unique_ptr<ProgressBar>(new ProgressBar(total_frames_, "Video processed:"));
}

uint64_t VideoReader::TotalFrames() const {
    return total_frames_;
}

bool VideoReader::GetFrame(cv::Mat& bgr_frame, affdex::Timestamp& timestamp_ms) {
    bool frame_data_loaded;

    do {
        frame_data_loaded = GetFrameData(bgr_frame, timestamp_ms);
        if (frame_data_loaded) {
            current_frame_++;
        }
    } while ((sampling_frame_rate_ > 0) && (timestamp_ms > 0) &&
        ((timestamp_ms - last_timestamp_ms_) < 1000 / sampling_frame_rate_) && frame_data_loaded);

    last_timestamp_ms_ = timestamp_ms;
    frame_progress_->progressed(current_frame_);
    return frame_data_loaded;
}

bool VideoReader::GetFrameData(cv::Mat& bgr_frame, affdex::Timestamp& timestamp_ms) {
    static const int MAX_ATTEMPTS = 2;
    affdex::Timestamp prev_timestamp_ms = cap_.get(::CV_CAP_PROP_POS_MSEC);
    bool frame_found = cap_.grab();
    bool frame_retrieved = cap_.retrieve(bgr_frame);
    timestamp_ms = cap_.get(::CV_CAP_PROP_POS_MSEC);

    // Two conditions result in failure to decode (grab/retrieve) a video frame (timestamp reports 0):
    // (1) error on a particular frame
    // (2) end of the video file
    //
    // This workaround double-checks that a subsequent frame can't be decoded, in the absence
    // of better reporting on which case has been encountered.
    //
    // In the case of reading an image, first attempt will not return a new frame, but the second one will
    // succeed. So as a workaround, the new timestamp must be greater than the previous one.
    int n_attempts = 0;
    while (!(frame_found && frame_retrieved) && n_attempts++ < MAX_ATTEMPTS) {
        frame_found = cap_.grab();
        frame_retrieved = cap_.retrieve(bgr_frame);
        timestamp_ms = cap_.get(::CV_CAP_PROP_POS_MSEC);
    }

    if (frame_found && frame_retrieved && n_attempts > 0) {
        if (timestamp_ms <= prev_timestamp_ms) {
            frame_found = false;
        }
    }

    return frame_found && frame_retrieved;
}
