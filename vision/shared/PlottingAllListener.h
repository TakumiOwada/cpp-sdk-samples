#pragma once

#include "Visualizer.h"
#include "Frame.h"
#include "OccupantListener.h"
#include "ObjectListener.h"

#include <deque>
#include <mutex>
#include <fstream>
#include <condition_variable>
#include <iostream>
#include <iomanip>
#include <utility>


using namespace affdex;

class PlottingAllListener : public vision::OccupantListener, public vision::ObjectListener, public vision::ImageListener{

public:

    PlottingAllListener(std::ofstream& csv, bool draw_display, bool enable_logging, const Duration callback_interval,
                        std::vector<vision::CabinRegion> cabin_regions,
                        std::map<vision::Feature, Duration> callback_intervals) :
        out_stream_(csv), image_data_(),  process_last_ts_(0),
        draw_display_(draw_display),  processed_frames_(0),logging_enabled_(enable_logging),
        callback_interval_(callback_interval), cabin_regions_(std::move(cabin_regions)),
        frames_with_occupants_(0), callback_intervals_(callback_intervals) {


        out_stream_ << "TimeStamp, occupantId, bodyId, confidence, regionId,  upperLeftX, upperLeftY, lowerRightX, "
                       "lowerRightY";

        for (const auto& cr :cabin_regions_) {
            out_stream_ << "," << "Region " << cr.id;
        }

        out_stream_ << std::endl;
        out_stream_.precision(2);
        out_stream_ << std::fixed;

        // set the timeout to the max callback interval
        for (const auto& pair : callback_intervals_) {
            if (pair.second > timeout_) {
                timeout_ = pair.second;
            }
        }
    }

    std::map<vision::Feature, Duration> getCallbackIntervals() const override {
        return callback_intervals_;
    }

    Duration getCallbackInterval() const override {
        return callback_interval_;
    }

    void onObjectResults(const std::map<ObjectId, Object>& objects, vision::Frame frame) override {
        std::lock_guard<std::mutex> lg(mtx);
        results_object_.emplace_back(frame, objects);
        process_last_ts_ = frame.getTimestamp();

        processed_frames_++;
        if (!objects.empty()) {
            frames_with_objects_++;
        }
    }

    void onOccupantResults(const std::map<vision::OccupantId, vision::Occupant>& occupants,
                           vision::Frame frame) override {
        std::lock_guard<std::mutex> lg(mtx);
        results_occupant_.emplace_back(frame, occupants);
        process_last_ts_ = frame.getTimestamp();

        processed_frames_++;
        if (!occupants.empty()) {
            frames_with_occupants_++;
        }
    }

    void onImageResults(std::map<vision::FaceId, vision::Face> faces, vision::Frame image) override {
        std::lock_guard<std::mutex> lg(mtx);
        const int diff = image.getTimestamp() - process_last_ts_;
        results_face_.emplace_back(image, faces);
        process_last_ts_ = image.getTimestamp();

        processed_frames_++;
        if (!faces.empty()) {
            frames_with_faces_++;
        }
    }

    void onImageCapture(vision::Frame image) override {
        //nothing to do here
    }


    void outputToFile(const std::map<vision::OccupantId, vision::Occupant>& occupants, double time_stamp) {
        if (occupants.empty()) {
            // TimeStamp occupantId confidence regionId upperLeftX upperLeftY lowerRightX lowerRightY"
            out_stream_ << time_stamp << ",nan,nan,nan,nan,nan,nan,nan,nan";
            for (const auto& cr :cabin_regions_) {
                out_stream_ << ",nan";
            }
            out_stream_ << std::endl;
        }

        for (const auto& occupant_id_pair : occupants) {
            const vision::Occupant occup = occupant_id_pair.second;
            std::vector<vision::Point> bbox({occup.boundingBox.getTopLeft(), occup.boundingBox.getBottomRight()});

            out_stream_ << time_stamp << ","
                        << occupant_id_pair.first << "," << (occup.body ? std::to_string(occup.body->id) : "Nan") << ","
                        << occup.matchedSeat.matchConfidence << "," << occup.matchedSeat.cabinRegion.id << ","
                        << std::setprecision(0) << bbox[0].x << "," << bbox[0].y << "," << bbox[1].x << "," << bbox[1].y
                        << std::setprecision(4);

            for (const auto& cr :cabin_regions_) {
                if (cr.id == occup.matchedSeat.cabinRegion.id) {
                    out_stream_ << "," << occup.matchedSeat.matchConfidence;
                }
                else {
                    out_stream_ << "," << 0;
                }
            }

            out_stream_ << std::endl;
        }
    }

    void draw() {
        const cv::Mat img = *(most_recent_frame_.getImage());
        viz_.updateImage(img);

        //draw occupant
        for (const auto& id_occupant_pair : latest_data_occupant_.second) {
            const auto occupant =  id_occupant_pair.second;
            viz_.drawOccupantMetrics(occupant);
            //add occupant region detected
            const auto id = occupant.matchedSeat.cabinRegion.id;
            if(std::find(occupant_regions_.begin(), occupant_regions_.end(), id) == occupant_regions_.end()) {
                occupant_regions_.emplace_back(id);
            }
        }

        //draw object
        for (const auto& id_object_pair : latest_data_object_.second) {
            const auto obj = id_object_pair.second;
            viz_.drawObjectMetrics(obj);
            //add object region detected
            for(const auto& o : obj.matchedRegions){
                const auto id = o.cabinRegion.id;
                if(std::find(object_regions_.begin(), object_regions_.end(), id) == object_regions_.end()) {
                    object_regions_.emplace_back(id);
                }
            }

            //add object type detected
            if(std::find(object_types_.begin(), object_types_.end(), obj.type) == object_types_.end()) {
                object_types_.emplace_back(obj.type);
            }
        }

        //draw face
        for (const auto& face_id_pair : latest_data_face_.second) {
            vision::Face f = face_id_pair.second;

            std::map<vision::FacePoint, vision::Point> points = f.getFacePoints();

            // Draw bounding box
            auto bbox = f.getBoundingBox();
            const float valence = f.getEmotions().at(vision::Emotion::VALENCE);
            viz_.drawBoundingBox(bbox, valence);

            // Draw Facial Landmarks Points
            viz_.drawPoints(f.getFacePoints());

            // Draw a face on screen
            viz_.drawFaceMetrics(f, bbox, true, false);
        }

        if (draw_display_) {
            viz_.showImage();
        }
        image_data_ = viz_.getImageData();
    }



    int getSamplesWithOccupantsPercent() const {
        return (static_cast<float>(frames_with_occupants_) / processed_frames_) * 100;
    }

    std::string getOccupantRegionsDetected() const {
        std::string occupant_regions;
        for(int i = 0; i<occupant_regions_.size(); ++i){
            if(i>0){
                occupant_regions += ", ";
            }
            occupant_regions += std::to_string(occupant_regions_[i]);
        }
        return occupant_regions;
    }

    void reset() {
        std::lock_guard<std::mutex> lg(mtx);
        process_last_ts_ = 0;
        processed_frames_ = 0;
        process_last_ts_ = 0;
        processed_frames_ = 0;

        frames_with_objects_ = 0;
        frames_with_occupants_ = 0;

        results_object_.clear();
        results_face_.clear();
        results_occupant_.clear();
    }

    int getProcessedFrames() {
        return processed_frames_;
    }

    //Needed to get Image data to create output video
    cv::Mat getImageData() { return image_data_; }

    void processResults(const vision::Frame& frame) {
        most_recent_frame_ = frame;
        std::lock_guard<std::mutex> lg(mtx);
        if (!results_object_.empty()) {
            time_callback_received_ = most_recent_frame_.getTimestamp();
            latest_data_object_ = results_object_.front();
            results_object_.pop_front();
            const vision::Frame old_frame = latest_data_object_.first;
            const auto items = latest_data_object_.second;
            //outputToFile(items, old_frame.getTimestamp());
        }
        if (!results_occupant_.empty()) {
            time_callback_received_ = most_recent_frame_.getTimestamp();
            latest_data_occupant_ = results_occupant_.front();
            results_occupant_.pop_front();
            const vision::Frame old_frame = latest_data_occupant_.first;
            const auto items = latest_data_occupant_.second;
            //outputToFile(items, old_frame.getTimestamp());
        }
        if (!results_face_.empty()) {
            time_callback_received_ = most_recent_frame_.getTimestamp();
            latest_data_face_ = results_face_.front();
            results_face_.pop_front();
            const vision::Frame old_frame = latest_data_face_.first;
            const auto items = latest_data_face_.second;
            //outputToFile(items, old_frame.getTimestamp());
        }
        draw();
    }

    std::string getObjectTypesDetected() const {
        std::string obj_types;
        for(int i = 0; i<object_types_.size(); ++i){
            if(i>0){
                obj_types += ", ";
            }
            obj_types += PlottingObjectListener::typeToString(object_types_[i]);
        }
        return obj_types;
    }

    std::string getObjectRegionsDetected() const {
        std::string object_regions;
        for(int i = 0; i<object_regions_.size(); ++i){
            if(i>0){
                object_regions += ", ";
            }
            object_regions += std::to_string(object_regions_[i]);
        }
        return object_regions;
    }

private:

    using frame_type_id_object_pair = std::pair<vision::Frame, std::map<vision::Id, vision::Object>>;
    using frame_type_id_occupant_pair = std::pair<vision::Frame, std::map<vision::Id, vision::Occupant>>;
    using frame_type_id_face_pair = std::pair<vision::Frame, std::map<vision::Id, vision::Face>>;

    std::deque<frame_type_id_object_pair> results_object_;
    std::deque<frame_type_id_occupant_pair> results_occupant_;
    std::deque<frame_type_id_face_pair> results_face_;

    frame_type_id_object_pair latest_data_object_;
    frame_type_id_occupant_pair latest_data_occupant_;
    frame_type_id_face_pair latest_data_face_;

    std::ofstream& out_stream_;
    Visualizer viz_;
    cv::Mat image_data_;

    std::mutex mtx;

    Timestamp process_last_ts_;
    bool draw_display_;
    int processed_frames_;
    bool logging_enabled_;
    vision::Frame most_recent_frame_;
    Timestamp time_callback_received_ = 0;
    Duration timeout_ = 500;
    Duration callback_interval_;
    std::vector<vision::CabinRegion> cabin_regions_;
    std::vector<int> occupant_regions_;
    int frames_with_occupants_;


    std::map<vision::Feature, Duration> callback_intervals_;
    std::vector<vision::Object::Type> object_types_;
    std::vector<int> object_regions_;
    int frames_with_objects_;
    int frames_with_faces_;
};
