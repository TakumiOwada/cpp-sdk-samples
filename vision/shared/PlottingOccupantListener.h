#pragma once

#include "OccupantListener.h"
#include "PlottingListener.h"

using namespace affdex;

class PlottingOccupantListener : public vision::OccupantListener, public PlottingListener<vision::Occupant> {

public:

    template<typename T>
    PlottingOccupantListener(std::ofstream& csv, T& program_options, const
    Duration callback_interval, std::vector<CabinRegion> cabin_regions) :
        PlottingListener(csv, program_options.draw_display, !program_options.disable_logging, program_options.write_video), callback_interval_(callback_interval),
        cabin_regions_(std::move(cabin_regions)), draw_occupant_id_(program_options.draw_id), frames_with_occupants_(0) {
        out_stream_ << "TimeStamp, occupantId, bodyId, confidence, regionId,  upperLeftX, upperLeftY, lowerRightX, "
                       "lowerRightY";

        for (const auto& cr :cabin_regions_) {
            out_stream_ << "," << "Region " << cr.id;
        }

        out_stream_ << std::endl;
        out_stream_.precision(2);
        out_stream_ << std::fixed;
    }

    Duration getCallbackInterval() const override {
        return callback_interval_;
    }

    void onOccupantResults(const std::map<vision::OccupantId, vision::Occupant>& occupants,
                           vision::Frame frame) override {
        std::lock_guard<std::mutex> lg(mtx);
        results_.emplace_back(frame, occupants);
        ++processed_frames_;
        if (!occupants.empty()) {
            ++frames_with_occupants_;
        }
    };

    void outputToFile(const std::map<vision::OccupantId, vision::Occupant>& occupants, double time_stamp) override {
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
            std::vector<vision::Point> bbox({occup.getBoundingBox().getTopLeft(), occup.getBoundingBox().getBottomRight()});

            out_stream_ << time_stamp << ","
                        << occupant_id_pair.first << "," << (occup.getBody() ? std::to_string(occup.getBody()->getId()) : "Nan") << ","
                        << occup.getMatchedSeat().matchConfidence << "," << occup.getMatchedSeat().cabinRegion.id << ","
                        << std::setprecision(0) << bbox[0].x << "," << bbox[0].y << "," << bbox[1].x << "," << bbox[1].y
                        << std::setprecision(4);

            for (const auto& cr :cabin_regions_) {
                if (cr.id == occup.getMatchedSeat().cabinRegion.id) {
                    out_stream_ << "," << occup.getMatchedSeat().matchConfidence;
                }
                else {
                    out_stream_ << "," << 0;
                }
            }

            out_stream_ << std::endl;
        }
    }

    void draw(const std::map<vision::OccupantId, vision::Occupant>& occupants, const vision::Frame& image) override {
        const cv::Mat img = *(image.getImage());
        viz_.updateImage(img);

        for (const auto& id_occupant_pair : occupants) {
            const auto occupant =  id_occupant_pair.second;
            viz_.drawOccupantMetrics(occupant);
            //add occupant region detected
            const auto id = occupant.getMatchedSeat().cabinRegion.id;
            if(std::find(occupant_regions_.begin(), occupant_regions_.end(), id) == occupant_regions_.end()) {
                occupant_regions_.emplace_back(id);
            }
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

    void reset() override {
        std::lock_guard<std::mutex> lg(mtx);
        processed_frames_ = 0;
        frames_with_occupants_ = 0;
        results_.clear();
    }

private:
    Duration callback_interval_;
    std::vector<CabinRegion> cabin_regions_;
    std::vector<int> occupant_regions_;
    bool draw_occupant_id_;
    int frames_with_occupants_;
};
