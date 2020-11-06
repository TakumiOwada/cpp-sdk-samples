#pragma once

#include <opencv2/imgproc/imgproc.hpp>
#include <Face.h>
#include "Occupant.h"
#include "Object.h"
#include <set>

struct ColorEdges {
    ColorEdges(const cv::Scalar& color, const affdex::vision::BodyPoint& start, const affdex::vision::BodyPoint& end) :
        color_(color), start_(start), end_(end) {}

    cv::Scalar color_;
    affdex::vision::BodyPoint start_;
    affdex::vision::BodyPoint end_;
};

//Plot the face metrics using opencv highgui
class Visualizer {
public:

    Visualizer();

    //UpdateImage refreshes the output_img that will be update
    void updateImage(const cv::Mat& output_img);

    //DrawPoints displays the landmark points on the image
    void drawPoints(const std::map<affdex::vision::FacePoint, affdex::vision::Point>& points);

    //DrawBoundingBox displays the bounding box with the valence value
    void drawBoundingBox(const std::vector<affdex::vision::Point>& bounding_box, float valence);

    //DrawBoundingBox displays the bounding box with the color
    void drawBoundingBox(const std::vector<affdex::vision::Point>& bounding_box, const cv::Scalar& color);

    //drawPolygon displays the polygon with the points and the color passed
    void drawPolygon(const std::vector<affdex::vision::Point>& points, const cv::Scalar& color);

    //DrawHeadOrientation Displays head orientation and associated value
    void drawHeadOrientation(std::map<affdex::vision::Measurement, float> headAngles, const int x, int& padding,
                             bool align_right = true, const cv::Scalar& color = cv::Scalar(255, 255, 255));

    //DrawFaceMetrics Displays all facial metrics and associated value
    void drawFaceMetrics(affdex::vision::Face face,
                         std::vector<affdex::vision::Point> bounding_box,
                         bool draw_face_id = false, bool draw_drowsiness = false);

    // Draw body related metrics
    void drawBodyMetrics(const std::map<affdex::vision::BodyPoint, affdex::vision::Point>& body_points);

    //Draw occupant related metrics
    void drawOccupantMetrics(const affdex::vision::Occupant& occupant);

    //Draw object related metrics
    void drawObjectMetrics(const affdex::vision::Object& object);

    //ShowImage displays image on screen for specified interval
    void showImage(int interval = 5);

    //Image data used to write the annotated video
    cv::Mat getImageData();

    std::vector<std::pair<affdex::vision::Expression, std::string>> EXPRESSIONS;
    std::vector<std::pair<affdex::vision::Emotion, std::string>> EMOTIONS;
    std::vector<std::pair<affdex::vision::Measurement, std::string>> HEAD_ANGLES;
    std::map<affdex::vision::DominantEmotion, std::string> DOMINANT_EMOTIONS;
    std::map<affdex::vision::Mood, std::string> MOODS;
    std::map<affdex::vision::AgeCategory, std::string> AGE_CATEGORIES;
    std::vector<ColorEdges> COLOR_EDGES_PAIR; //contains body points with its respective color
    std::map<affdex::vision::GazeRegion, std::string> GAZE_REGIONS; //Region of gaze, from the person’s perspective -- e.g. LEFT means the person is looking to THEIR left”
    std::map<affdex::vision::Drowsiness, std::string> DROWSINESS_LEVELS; //Drowsiness level with its respective string

private:

    /* Overlay an image with an Alpha (foreground) channel over background at a specified location
    * Adapted from : http://jepsonsblog.blogspot.com/2012/10/overlay-transparent-image-in-opencv.html*/
    void overlayImage(const cv::Mat& foreground, cv::Mat& background, const cv::Point2i& location);

    //DrawClassifierOutput Displays a classifier and associated value with passed location and alignment
    void drawClassifierOutput(const std::string& classifier, const float value,
                              const cv::Point2f& loc, bool align_right = false);

    //DrawEqualizer displays an equalizer on screen either right or left justified at the anchor location (loc)
    void drawEqualizer(const std::string& name, const float value, const cv::Point2f& loc,
                       bool align_right, const cv::Scalar& color);

    //DrawText displays an text on screen either right or left justified at the anchor location (loc)
    void drawText(const std::string& name, const std::string& value,
                  const cv::Point2f& loc, bool align_right = false, cv::Scalar color = cv::Scalar(255, 255, 255),
                  cv::Scalar bg_color = cv::Scalar(50, 50, 50));

    std::set<std::string> GREEN_COLOR_CLASSIFIERS;
    std::set<std::string> RED_COLOR_CLASSIFIERS;

    cv::Mat img;
    cv::Mat logo;
    bool logo_resized;
    const int spacing = 20;
    const int LOGO_PADDING = 20;
};

//Color generator (linear) for red-to-green values
class ColorgenRedGreen {
public:
    //Constructor
    ColorgenRedGreen(const float red_val, const float green_val)
        : red_val_(red_val),
          green_val_(green_val) {}

    //Generate accessor from val
    cv::Scalar operator()(const float val) const;

private:
    const float red_val_;
    const float green_val_;
};

//Color generator (linear) between any two colors
class ColorgenLinear {
public:
    //Constructor
    ColorgenLinear(const float val1, const float val2, cv::Scalar color1, cv::Scalar color2)
        :
        val1_(val1),
        val2_(val2),
        color1_(color1),
        color2_(color2) {}

    // Generate accessor from val
    cv::Scalar operator()(const float val) const;

private:
    const float val1_;
    const float val2_;

    const cv::Scalar color1_;
    const cv::Scalar color2_;
};


