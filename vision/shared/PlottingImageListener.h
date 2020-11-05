#pragma once

#include "ImageListener.h"
#include "PlottingListener.h"

#include "ProgramOptions.h"

using namespace affdex;

class PlottingImageListener : public vision::ImageListener, public PlottingListener<vision::Face> {

public:

    PlottingImageListener(std::ofstream& csv, ProgramOptionsWebcam& program_options ) :
        PlottingListener(csv, program_options.draw_display, !program_options.disable_logging), capture_last_ts_(0),
        capture_fps_(0), draw_face_id_(program_options.draw_id), frames_with_faces_(0),
        show_drowsiness_(program_options.show_drowsiness) {
        setCSVHeaders();
    }

    PlottingImageListener(std::ofstream& csv, ProgramOptionsVideo& program_options ) :
        PlottingListener(csv, program_options.draw_display, !program_options.disable_logging), capture_last_ts_(0),
        capture_fps_(0), draw_face_id_(program_options.draw_id), frames_with_faces_(0),
        show_drowsiness_(program_options.show_drowsiness) {
        setCSVHeaders();
    }

    void setCSVHeaders() {
        out_stream_ << "TimeStamp,faceId,upperLeftX,upperLeftY,lowerRightX,lowerRightY,confidence,interocularDistance,";
        for (const auto& angle : viz_.HEAD_ANGLES) {
            out_stream_ << angle.second << ",";
        }
        for (const auto& emotion : viz_.EMOTIONS) {
            out_stream_ << emotion.second << ",";
        }
        for (const auto& expression : viz_.EXPRESSIONS) {
            out_stream_ << expression.second << ",";
        }
        out_stream_ << "mood,dominantEmotion,dominantEmotionConfidence,gazeRegion, gazeConfidence,";
        out_stream_ << "identity,identityConfidence,age,ageConfidence,ageCategory,glasses";

        if(show_drowsiness_) {
            out_stream_ << ",drowsinessLevel,drowsinessConfidence";
        }

        out_stream_ << std::endl;
        out_stream_.precision(2);
        out_stream_ << std::fixed;
    }

    int getCaptureFrameRate() {
        std::lock_guard<std::mutex> lg(mtx);
        return capture_fps_;
    }

    void onImageResults(std::map<vision::FaceId, vision::Face> faces, vision::Frame frame) override {
        std::lock_guard<std::mutex> lg(mtx);
        stopTimer();
        results_.emplace_back(frame, faces);
        ++processed_frames_;
        if (!faces.empty()) {
            ++frames_with_faces_;
        }
    }

    void onImageCapture(vision::Frame image) override {
        std::lock_guard<std::mutex> lg(mtx);
        const int diff = image.getTimestamp() - capture_last_ts_;
        if (diff > 0) {
            capture_fps_ = 1000.0f / diff;
            capture_last_ts_ = image.getTimestamp();
        }
    };

    void outputToFile(const std::map<vision::FaceId, vision::Face>& faces, double time_stamp) override {
        if (faces.empty()) {
            out_stream_ << time_stamp
                        << ",nan,nan,nan,nan,nan,nan,nan,"; // face ID, bbox UL X, UL Y, BR X, BR Y, confidence, interocular distance
            for (const auto& angle : viz_.HEAD_ANGLES) {
                out_stream_ << "nan,";
            }
            for (const auto& emotion : viz_.EMOTIONS) {
                out_stream_ << "nan,";
            }
            for (const auto& expression : viz_.EXPRESSIONS) {
                out_stream_ << "nan,";
            }
            // mood,dominantEmotion,dominantEmotionConfidence,gazeRegion, gazeConfidence,identity,identityConfidence,age,ageConfidence,ageCategory,glasses
            out_stream_<< "nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan";
            if(show_drowsiness_) {
                out_stream_ << ",nan,nan";
            }
            out_stream_ << std::endl;
        }

        for (const auto& face_id_pair : faces) {
            vision::Face f = face_id_pair.second;
            std::vector<vision::Point> bbox(f.getBoundingBox());

            out_stream_ << time_stamp << ","
                        << f.getId() << ","
                        << std::setprecision(0) << bbox[0].x << "," << bbox[0].y << "," << bbox[1].x << "," << bbox[1].y
                        << "," << std::setprecision(4)
                        << f.getConfidence() << ","
                        << f.getMeasurements().at(vision::Measurement::INTEROCULAR_DISTANCE) << ",";

            auto measurements = f.getMeasurements();
            for (const auto& m : viz_.HEAD_ANGLES) {
                out_stream_ << measurements.at(m.first) << ",";
            }

            auto emotions = f.getEmotions();
            for (const auto& emo : viz_.EMOTIONS) {
                out_stream_ << emotions.at(emo.first) << ",";
            }

            auto expressions = f.getExpressions();
            for (const auto& exp : viz_.EXPRESSIONS) {
                out_stream_ << expressions.at(exp.first) << ",";
            }

            vision::Mood mood = f.getMood();
            out_stream_ << viz_.MOODS[mood] << ",";

            vision::DominantEmotionMetric dominant_emotion_metric = f.getDominantEmotion();
            out_stream_ << viz_.DOMINANT_EMOTIONS[dominant_emotion_metric.dominantEmotion] << ","
                        << dominant_emotion_metric.confidence << ",";

            vision::GazeMetric gaze_metric = f.getGazeMetric();

            out_stream_ << viz_.GAZE_REGIONS[gaze_metric.gazeRegion] << "," << gaze_metric.confidence << ",";

            auto identity_metric = f.getIdentityMetric();
            std::string id_content;
            identity_metric.id == -1 ? id_content = "UNKNOWN" : id_content = std::to_string(identity_metric.id);
            out_stream_ << id_content << "," << identity_metric.confidence << ",";

            auto age_metric = f.getAgeMetric();
            std::string age_content;
            age_metric.years == -1 ? age_content = "UNKNOWN" : age_content = std::to_string(age_metric.years);
            out_stream_ << age_content << "," << age_metric.confidence << ",";

            auto age_category = f.getAgeCategory();
            out_stream_ << viz_.AGE_CATEGORIES.at(age_category);

            out_stream_ << "," << f.getGlasses();

            if(show_drowsiness_) {
                const auto drowsiness_metric = f.getDrowsinessMetric();
                out_stream_ << "," << viz_.DROWSINESS_LEVELS[drowsiness_metric.drowsiness]
                            << "," << drowsiness_metric.confidence;
            }

            out_stream_ << std::endl;
        }
    }

    void draw(const std::map<vision::FaceId, vision::Face>& faces, const vision::Frame& image) override {
        const cv::Mat img = *(image.getImage());
        viz_.updateImage(img);

        for (const auto& face_id_pair : faces) {
            vision::Face f = face_id_pair.second;

            std::map<vision::FacePoint, vision::Point> points = f.getFacePoints();

            // Draw bounding box
            auto bbox = f.getBoundingBox();
            const float valence = f.getEmotions().at(vision::Emotion::VALENCE);
            viz_.drawBoundingBox(bbox, valence);

            // Draw Facial Landmarks Points
            viz_.drawPoints(f.getFacePoints());

            // Draw a face on screen
            viz_.drawFaceMetrics(f, bbox, draw_face_id_, show_drowsiness_);
        }
        if (draw_display_) {
            viz_.showImage();
        }
        image_data_ = viz_.getImageData();
    }

    int getFramesWithFaces() const {
        return frames_with_faces_;
    }

    int getFramesWithFacesPercent() {
        return (static_cast<float>(frames_with_faces_) / processed_frames_) * 100;
    }


    void reset() override {
        std::lock_guard<std::mutex> lg(mtx);
        capture_last_ts_ = 0;
        capture_fps_ = 0;
        processed_frames_ = 0;
        frames_with_faces_ = 0;
        process_fps_ = 0.0f;
        results_.clear();
        total_frames_count_ = 0;
    }

private:
    Timestamp capture_last_ts_;
    int capture_fps_;
    bool draw_face_id_;
    int frames_with_faces_;
    bool show_drowsiness_;
};
