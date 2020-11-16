#pragma once

#include <Core.h>
#include "vector"
#include <opencv2/highgui/highgui.hpp>


struct ProgramOptionsCommon {
    enum DetectionType {
        FACE,
        OBJECT,
        OCCUPANT,
        BODY
    };
    // cmd line args
    affdex::Path data_dir;
    affdex::Path output_video_path;
    cv::VideoWriter output_video;
    DetectionType detection_type = FACE;

    unsigned int num_faces;

    bool disable_logging = false;
    bool write_video = false;
    bool show_drowsiness = false;
};

struct ProgramOptionsVideo : ProgramOptionsCommon {

    // cmd line args
    affdex::Path input_video_path;
    int sampling_frame_rate;
    bool draw_display;
    bool loop = false;
    bool draw_id = false;
};

struct ProgramOptionsWebcam : ProgramOptionsCommon{
    // cmd line args
    affdex::Path output_file_path;
    std::vector<int> resolution;
    float process_framerate;
    int camera_framerate;
    int camera_id;
    bool draw_display = true;
    bool sync = false;
    bool draw_id = true;
};