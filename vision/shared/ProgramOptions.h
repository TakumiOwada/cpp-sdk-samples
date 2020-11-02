#pragma once

#include <Core.h>
#include "vector"
#include <opencv2/highgui/highgui.hpp>


struct ProgramOptionsVideo {
    enum DetectionType {
        FACE,
        OBJECT,
        OCCUPANT,
        BODY
    };
    // cmd line args
    affdex::Path data_dir;
    affdex::Path input_video_path;
    affdex::Path output_video_path;
    float sampling_frame_rate;
    bool draw_display;
    unsigned int num_faces;
    bool loop = false;
    bool draw_id = false;
    bool disable_logging = false;
    bool write_video = false;
    bool show_drowsiness = false;
    cv::VideoWriter output_video;
    DetectionType detection_type = FACE;
};

struct ProgramOptionsWebcam {

    enum DetectionType {
        FACE,
        OBJECT,
        OCCUPANT,
        BODY
    };
    // cmd line args
    affdex::Path data_dir;
    affdex::Path output_file_path;
    affdex::Path output_video_path;
    std::vector<int> resolution;
    int process_framerate;
    int camera_framerate;
    int camera_id;
    unsigned int num_faces;
    bool draw_display = true;
    bool sync = false;
    bool draw_id = true;
    bool disable_logging = false;
    bool write_video = false;
    bool show_drowsiness = false;
    cv::VideoWriter output_video;
    DetectionType detection_type = FACE;
};