#pragma once

#include "BodyListener.h"
#include "PlottingListener.h"

using namespace affdex;

static std::map<BodyPoint, str> BODY_POINT_TO_STRING = {
    {BodyPoint::NOSE, "nose"},
    {BodyPoint::RIGHT_SHOULDER, "right_shoulder"},
    {BodyPoint::RIGHT_ELBOW, "right_elbow"},
    {BodyPoint::RIGHT_WRIST, "right_wrist"},
    {BodyPoint::LEFT_SHOULDER, "left_shoulder"},
    {BodyPoint::LEFT_ELBOW, "left_elbow"},
    {BodyPoint::LEFT_WRIST, "left_wrist"},
    {BodyPoint::RIGHT_HIP, "right_hip"},
    {BodyPoint::LEFT_HIP, "left_hip"},
    {BodyPoint::RIGHT_EYE, "right_eye"},
    {BodyPoint::LEFT_EYE, "left_eye"},
    {BodyPoint::RIGHT_EAR, "right_ear"},
    {BodyPoint::LEFT_EAR, "left_ear"},
    {BodyPoint::NECK, "neck"}
};

static const std::vector<BodyPoint> BODY_POINTS = {
    BodyPoint::NOSE,
    BodyPoint::RIGHT_SHOULDER,
    BodyPoint::RIGHT_ELBOW,
    BodyPoint::RIGHT_WRIST,
    BodyPoint::LEFT_SHOULDER,
    BodyPoint::LEFT_ELBOW,
    BodyPoint::LEFT_WRIST,
    BodyPoint::RIGHT_HIP,
    BodyPoint::LEFT_HIP,
    BodyPoint::RIGHT_EYE,
    BodyPoint::LEFT_EYE,
    BodyPoint::RIGHT_EAR,
    BodyPoint::LEFT_EAR,
    BodyPoint::NECK
};

static const int HEADER_SIZE = 38;

class PlottingBodyListener : public vision::BodyListener, public PlottingListener<vision::Body> {

public:

    PlottingBodyListener(std::ofstream& csv, bool draw_display, bool enable_logging, bool draw_body_id, const
    Duration callback_interval) :
        PlottingListener(csv, draw_display, enable_logging), callback_interval_(callback_interval),
        draw_body_id_(draw_body_id), frames_with_bodies_(0) {

        out_stream_ << "TimeStamp, bodyId";

        for (auto bodyPoint : BODY_POINTS) {
            if (BODY_POINT_TO_STRING.find(bodyPoint) != BODY_POINT_TO_STRING.end()) {
                auto text = BODY_POINT_TO_STRING[bodyPoint];
                out_stream_ << ", " + text + "_x" << ", " + text + "_y";
            }
            else {
                std::cout << "ERROR\n";
            }
        }

        out_stream_ << std::endl;
        out_stream_.precision(2);
        out_stream_ << std::fixed;
    }

    Duration getCallbackInterval() const override {
        return callback_interval_;
    }

    void onBodyResults(const std::map<vision::BodyId, vision::Body>& bodies,
                       vision::Frame frame) override {
        std::lock_guard<std::mutex> lg(mtx);
        results_.emplace_back(frame, bodies);
        process_last_ts_ = frame.getTimestamp();

        ++processed_frames_;
        if (!bodies.empty()) {
            ++frames_with_bodies_;
        }
    };

    void outputToFile(const std::map<vision::BodyId, vision::Body>& bodies, double time_stamp) override {
        if (bodies.empty()) {

            out_stream_ << time_stamp;
            for (int i = 0; i < HEADER_SIZE - 1; ++i) {
                out_stream_ << ",nan";
            }
            out_stream_ << std::endl;
        }

        for (const auto& body_id_pair : bodies) {
            std::map<BodyPoint, Point> body_point_pt = body_id_pair.second.body_points;

            out_stream_ << time_stamp << ","
                        << body_id_pair.first << std::setprecision(0);
            for (auto bodyPoint : BODY_POINTS) {
                if (body_point_pt.find(bodyPoint) != body_point_pt.end()) {
                    Point pt = body_point_pt[bodyPoint];
                    out_stream_ << ", " << pt.x << ", " << pt.y;
                }
                else {
                    out_stream_ << ", nan, nan ";
                }
            }
            out_stream_ << std::setprecision(4) << std::endl;
        }
    }

    void draw(const std::map<vision::BodyId, vision::Body>& bodies, const vision::Frame& image) override {
        const cv::Mat img = *(image.getImage());
        viz_.updateImage(img);

        for (const auto& id_body_pair : bodies) {
            auto body_points = id_body_pair.second.body_points;
            viz_.drawBodyMetrics(body_points);
        }

        if (draw_display_) {
            viz_.showImage();
        }
        image_data_ = viz_.getImageData();
    }

    int getSamplesWithBodiesPercent() const {
        return (static_cast<float>(frames_with_bodies_) / processed_frames_) * 100;
    }

    void reset() override {
        std::lock_guard<std::mutex> lg(mtx);
        process_last_ts_ = 0;
        start_ = std::chrono::system_clock::now();
        processed_frames_ = 0;
        frames_with_bodies_ = 0;
        results_.clear();
    }

private:
    Duration callback_interval_;
    std::vector<int> body_regions_;
    bool draw_body_id_;
    int frames_with_bodies_;
};
