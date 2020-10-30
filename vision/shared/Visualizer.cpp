#include "Visualizer.h"
#include "AffectivaLogo.h"
#include "PlottingObjectListener.h"

#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include <utility>
#include <Face.h>

using namespace affdex::vision;

Visualizer::Visualizer() :
    GREEN_COLOR_CLASSIFIERS({
                                "joy"
                            }),
    RED_COLOR_CLASSIFIERS({
                              "anger"
                          }) {
    logo_resized = false;
    logo = cv::imdecode(cv::InputArray(small_logo), CV_LOAD_IMAGE_UNCHANGED);

    EXPRESSIONS = {
        {Expression::SMILE, "smile"},
        {Expression::BROW_RAISE, "browRaise"},
        {Expression::BROW_FURROW, "browFurrow"},
        {Expression::NOSE_WRINKLE, "noseWrinkle"},
        {Expression::UPPER_LIP_RAISE, "upperLipRaise"},
        {Expression::MOUTH_OPEN, "mouthOpen"},
        {Expression::EYE_CLOSURE, "eyeClosure"},
        {Expression::CHEEK_RAISE, "cheekRaise"},
        {Expression::YAWN, "yawn"},
        {Expression::BLINK, "blink"},
        {Expression::BLINK_RATE, "blinkRate"},
        {Expression::EYE_WIDEN, "eyeWiden"},
        {Expression::INNER_BROW_RAISE, "innerBrowRaise"},
        {Expression::LIP_CORNER_DEPRESSOR, "lipCornerDepressor"},
        {Expression::LID_TIGHTEN, "lidTighten"}
    };

    EMOTIONS = {
        {Emotion::JOY, "joy"},
        {Emotion::ANGER, "anger"},
        {Emotion::SURPRISE, "surprise"},
        {Emotion::VALENCE, "valence"},
        {Emotion::FEAR, "fear"},
        {Emotion::SADNESS, "sadness"},
        {Emotion::DISGUST, "disgust"},
        {Emotion::NEUTRAL, "neutral"},
        {Emotion::CONTEMPT, "contempt"}
    };

    HEAD_ANGLES = {
        {Measurement::PITCH, "pitch"},
        {Measurement::YAW, "yaw"},
        {Measurement::ROLL, "roll"}
    };

    DOMINANT_EMOTIONS = {
        {DominantEmotion::UNKNOWN, "unknown"},
        {DominantEmotion::NEUTRAL, "neutral"},
        {DominantEmotion::JOY, "joy"},
        {DominantEmotion::ANGER, "anger"},
        {DominantEmotion::SURPRISE, "surprise"},
        {DominantEmotion::SADNESS, "sadness"},
        {DominantEmotion::DISGUST, "disgust"},
        {DominantEmotion::FEAR, "fear"},
    };

    MOODS = {
        {Mood::UNKNOWN, "UNKNOWN"},
        {Mood::NEUTRAL, "NEUTRAL"},
        {Mood::NEGATIVE, "NEGATIVE"},
        {Mood::POSITIVE, "POSITIVE"},
    };

    AGE_CATEGORIES = {
        {AgeCategory::UNKNOWN, "UNKNOWN"},
        {AgeCategory::BABY, "BABY"},
        {AgeCategory::CHILD, "CHILD"},
        {AgeCategory::TEEN, "TEEN"},
        {AgeCategory::ADULT, "ADULT"}
    };

    COLOR_EDGES_PAIR = {
        {cv::Scalar(255, 0, 0), BodyPoint::NECK, BodyPoint::RIGHT_SHOULDER},
        {cv::Scalar(255, 85, 0), BodyPoint::NECK, BodyPoint::LEFT_SHOULDER},
        {cv::Scalar(255, 170, 0), BodyPoint::RIGHT_SHOULDER, BodyPoint::RIGHT_ELBOW},
        {cv::Scalar(255, 255, 0), BodyPoint::RIGHT_ELBOW, BodyPoint::RIGHT_WRIST},
        {cv::Scalar(170, 255, 0), BodyPoint::LEFT_SHOULDER, BodyPoint::LEFT_ELBOW},
        {cv::Scalar(85, 255, 0), BodyPoint::LEFT_ELBOW, BodyPoint::LEFT_WRIST},
        {cv::Scalar(0, 255, 0), BodyPoint::NECK, BodyPoint::RIGHT_HIP},
        {cv::Scalar(0, 255, 255), BodyPoint::NECK, BodyPoint::LEFT_HIP},
        {cv::Scalar(0, 0, 255), BodyPoint::NECK, BodyPoint::NOSE},
        {cv::Scalar(85, 0, 255), BodyPoint::NOSE, BodyPoint::RIGHT_EYE},
        {cv::Scalar(170, 0, 255), BodyPoint::RIGHT_EYE, BodyPoint::RIGHT_EAR},
        {cv::Scalar(255, 0, 255), BodyPoint::NOSE, BodyPoint::LEFT_EYE},
        {cv::Scalar(255, 0, 170), BodyPoint::LEFT_EYE, BodyPoint::LEFT_EAR}
    };

    GAZE_REGIONS = {
        {GazeRegion::UNKNOWN, "UNKNOWN"},
        {GazeRegion::LEFT, "LEFT"},
        {GazeRegion::RIGHT, "RIGHT"},
        {GazeRegion::UP_RIGHT, "UP_RIGHT"},
        {GazeRegion::FORWARD, "FORWARD"},
        {GazeRegion::FORWARD_DOWN, "FORWARD_DOWN"},
        {GazeRegion::DOWN, "DOWN"}
    };

    DROWSINESS_LEVELS = {
        {Drowsiness::UNKNOWN, "UNKNOWN"},
        {Drowsiness::AWAKE, "AWAKE"},
        {Drowsiness::MODERATE, "MODERATE"},
        {Drowsiness::SEVERE, "SEVERE"},
        {Drowsiness::ASLEEP, "ASLEEP"}
    };

}

void Visualizer::drawFaceMetrics(affdex::vision::Face face, std::vector<Point> bounding_box, bool draw_face_id, bool show_drowsiness) {
    //Draw Right side metrics
    int padding = bounding_box[0].y; //Top left Y

    const auto expressions = face.getExpressions();
    for (auto& exp : EXPRESSIONS) {
        // special case: display blink rate as number instead of bar
        if (exp.first == Expression::BLINK_RATE) {
            std::stringstream ss;
            ss << std::fixed << std::setw(3) << std::setprecision(1);
            ss << expressions.at(exp.first);
            drawText(exp.second,
                     ss.str(),
                     cv::Point(bounding_box[1].x, padding += spacing),
                     false,
                     cv::Scalar(255, 255, 255));
        }
        else {
            float val = expressions.at(exp.first);
            if (exp.first == Expression::BLINK) {
                val *= 100;
            } // blink is 0 or 1, so translate to 0 or 100 so it shows up in the UI
            drawClassifierOutput(exp.second, val, cv::Point(bounding_box[1].x, padding += spacing), false);
        }
    }

    //Draw Head Angles
    drawHeadOrientation(face.getMeasurements(), bounding_box[1].x, padding, false);

    padding = bounding_box[0].y;  //Top right Y
    if (draw_face_id) {
        drawText("ID",
                 std::to_string(face.getId()),
                 cv::Point(bounding_box[0].x, padding + spacing),
                 false,
                 cv::Scalar(255, 255, 255));
    }

    //Draw Left side metrics
    const auto emotions = face.getEmotions();
    for (auto& emo : EMOTIONS) {
        drawClassifierOutput(emo.second,
                             emotions.at(emo.first),
                             cv::Point(bounding_box[0].x, padding += spacing),
                             true);
    }

    //Draw identity
    const auto identity = face.getIdentityMetric();
    std::string id_content;
    identity.id == -1 ? id_content = "UNKNOWN" : id_content = std::to_string(identity.id);
    drawText("identity", id_content, cv::Point(bounding_box[0].x, padding += spacing), true);
    drawClassifierOutput("identity_confidence",
                         identity.confidence,
                         cv::Point(bounding_box[0].x, padding += spacing),
                         true);

    //Draw age
    const auto age = face.getAgeMetric();
    std::string age_content;
    age.years == -1 ? age_content = "UNKNOWN" : age_content = std::to_string(age.years);
    drawText("age", age_content, cv::Point(bounding_box[0].x, padding += spacing), true);
    drawClassifierOutput("age_confidence", age.confidence, cv::Point(bounding_box[0].x, padding += spacing), true);

    //Draw age category
    const auto age_category = face.getAgeCategory();
    drawText("age_category", AGE_CATEGORIES[age_category], cv::Point(bounding_box[0].x, padding += spacing),
             true);

    //Draw gaze
    const auto gaze = face.getGazeMetric();

    drawText("gaze_region", GAZE_REGIONS[gaze.gazeRegion], cv::Point(bounding_box[0].x, padding += spacing), true);
    drawClassifierOutput("gaze_confidence", gaze.confidence, cv::Point(bounding_box[0].x, padding += spacing), true);

    //Draw glasses confidence
    drawClassifierOutput("glasses", face.getGlasses(), cv::Point(bounding_box[0].x, padding += spacing), true);

    if(show_drowsiness) {
        // draw drowsiness metric
        const auto drowsiness_metric = face.getDrowsinessMetric();
        drawText("drowsiness_level", DROWSINESS_LEVELS[drowsiness_metric.drowsiness], cv::Point(bounding_box[0].x,
                                                                                                padding += spacing), true);
        drawText("drowsiness_confidence", std::to_string(drowsiness_metric.confidence), cv::Point(bounding_box[0].x,
                                                                                                  padding += spacing), true);
    }
}

void Visualizer::updateImage(const cv::Mat& output_img) {
    img = output_img;

    if (!logo_resized) {
        const double logo_width = (logo.size().width > img.size().width * 0.25 ? img.size().width * 0.25 : logo.size().width);
        const double logo_height = ((double)logo_width) * ((double)logo.size().height / logo.size().width);
        cv::resize(logo, logo, cv::Size(logo_width, logo_height));
        logo_resized = true;
    }
    cv::Mat roi = img(cv::Rect(img.cols - logo.cols - 10, 10, logo.cols, logo.rows));
    overlayImage(logo, roi, cv::Point(0, 0));
}

void Visualizer::drawPoints(const std::map<FacePoint, Point>& points) {
    for (const auto& point : points)    //Draw face feature points.
    {
        cv::circle(img, cv::Point(point.second.x, point.second.y), 2.0f, cv::Scalar(255, 255, 255));
    }
}

void Visualizer::drawBoundingBox(const std::vector<Point>& bounding_box, float valence) {
    if (!bounding_box.empty()) {
        //Draw bounding box
        const ColorgenRedGreen valence_color_generator(-100, 100);
        const cv::Point top_left(bounding_box[0].x, bounding_box[0].y);
        const cv::Point bottom_right(bounding_box[1].x, bounding_box[1].y);
        cv::rectangle(img, top_left, bottom_right,
                      valence_color_generator(valence), 3);
    }
}

void Visualizer::drawBoundingBox(const std::vector<Point>& bounding_box, const cv::Scalar& color) {
    if (!bounding_box.empty()) {
        //Draw bounding box
        const cv::Point top_left(bounding_box[0].x, bounding_box[0].y);
        const cv::Point bottom_right(bounding_box[1].x, bounding_box[1].y);
        cv::rectangle(img, top_left, bottom_right, color, 3);
    }
}

void Visualizer::drawPolygon(const std::vector<Point>& points, const cv::Scalar& color) {
    if (!points.empty()) {
        //Draw polygon
        std::vector<cv::Point> pts;
        pts.reserve(points.size());
        for (const auto& p: points) {
            pts.emplace_back(cv::Point(p.x, p.y));
        }
        cv::polylines(img, pts, true, color, 3);
    }
}

void Visualizer::drawText(const std::string& name, const std::string& value,
                          const cv::Point2f& loc, bool align_right, cv::Scalar color, cv::Scalar bg_color) {
    const int block_width = 8;
    const int margin = 2;
    const int block_size = 10;
    const int max_blocks = 100 / block_size;

    cv::Point2f display_loc = loc;
    const std::string label = name + ": ";

    if (align_right) {
        display_loc.x -= (margin + block_width) * max_blocks;
        int baseline = 0;
        cv::Size txtSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5f, 5, &baseline);
        display_loc.x -= txtSize.width;
    }
    cv::putText(img, label + value, display_loc, cv::FONT_HERSHEY_SIMPLEX, 0.5f, std::move(bg_color), 5);
    cv::putText(img, label + value, display_loc, cv::FONT_HERSHEY_SIMPLEX, 0.5f, std::move(color), 1);
}

void Visualizer::drawBodyMetrics(const std::map<BodyPoint, Point>& body_points) {
    //draw lines between two decided body points
    for (const auto& color_edges : COLOR_EDGES_PAIR) {
        if (body_points.find(color_edges.start_) != body_points.end() &&
            body_points.find(color_edges.end_) != body_points.end()) {
            const Point pt1 = body_points.at(color_edges.start_);
            const Point pt2 = body_points.at(color_edges.end_);

            cv::line(img, cv::Point(pt1.x, pt1.y), cv::Point(pt2.x, pt2.y), color_edges.color_, 3);
        }
    }
}

void Visualizer::drawOccupantMetrics(const affdex::vision::Occupant& occupant) {

    // Draw occupant bounding box
    auto bbox = {occupant.getBoundingBox().getTopLeft(), occupant.getBoundingBox().getBottomRight()};
    drawBoundingBox(bbox, {199, 110, 255});

    //Do not draw if polygon's ID is Unknown
    if (occupant.getMatchedSeat().cabinRegion.id != REGION_UNKNOWN) {
        drawPolygon(occupant.getMatchedSeat().cabinRegion.vertices, {255, 255, 255});
    }

    int padding = occupant.getBoundingBox().getTopLeft().y; //Top left Y

    const std::string id(std::to_string(occupant.getMatchedSeat().cabinRegion.id));
    const std::string region_type(affdex::vision::CabinRegion::typeToString(occupant.getMatchedSeat().cabinRegion.type));
    const std::string match_confidence(std::to_string(occupant.getMatchedSeat().matchConfidence));

    drawText("Region Confidence", match_confidence, cv::Point(occupant.getBoundingBox().getTopLeft().x, padding -= spacing),
             false);
    drawText("Region " + id, region_type, cv::Point(occupant.getBoundingBox().getTopLeft().x, padding -= spacing), false);
    drawText("ID", std::to_string(occupant.getId()), cv::Point(occupant.getBoundingBox().getTopLeft().x, padding -= spacing), false);
    if (occupant.getFace()) {
        drawText("FaceID", std::to_string(occupant.getFace()->getId()), cv::Point(occupant.getBoundingBox().getTopLeft().x, padding -= spacing), false);
//        drawBoundingBox(occupant.face->getBoundingBox(), {0, 0, 255});
    }
    if (occupant.getBody()) {
        drawText("BodyID", std::to_string(occupant.getBody()->getId()), cv::Point(occupant.getBoundingBox().getTopLeft().x, padding -= spacing), false);
        drawBodyMetrics(occupant.getBody()->getBodyPoints());
    }
}

void Visualizer::drawObjectMetrics(const affdex::vision::Object& object) {

    // Draw object bounding box
    auto bbox = {object.getBoundingBox().getTopLeft(), object.getBoundingBox().getBottomRight()};

    //default color ==GRAY
    cv::Scalar color(128, 128, 128);
    if (object.getType() == Object::Type::PHONE) {
        //phone color == YELLOW
        color = {0, 255, 255};
    }
    else if (object.getType() == Object::Type::CHILD_SEAT) {
        //child seat color == RED
        color = {0, 0, 255};
    }

    //Configured area region;
    drawBoundingBox(bbox, color);

    const auto matched_region = object.getMatchedRegions()[0];
    //Do not draw if polygon's ID is Unknown
    if (matched_region.cabinRegion.id != REGION_UNKNOWN) {
        drawPolygon(matched_region.cabinRegion.vertices, {255, 255, 255});
    }

    int padding = object.getBoundingBox().getTopLeft().y; //Top left Y

    drawText("Type", PlottingObjectListener::typeToString(object.getType()), cv::Point(object.getBoundingBox().getTopLeft().x,
                                                                                  padding -=
                                                                                      spacing),
             false);

    const std::string id(std::to_string(matched_region.cabinRegion.id));
    const std::string
        region_type(affdex::vision::CabinRegion::typeToString(matched_region.cabinRegion.type));

    const std::string confidence(std::to_string(object.getConfidence()));
    const std::string regions_confidence(std::to_string(matched_region.matchConfidence));

    drawText("Object Confidence",
             confidence,
             cv::Point(object.getBoundingBox().getTopLeft().x, padding -= spacing),
             false);

    drawText("Region Confidence",
             regions_confidence,
             cv::Point(object.getBoundingBox().getTopLeft().x, padding -= spacing),
             false);

    drawText("Region " + id, region_type, cv::Point(object.getBoundingBox().getTopLeft().x, padding -= spacing), false);
}

void Visualizer::drawClassifierOutput(const std::string& classifier,
                                      const float value, const cv::Point2f& loc, bool align_right) {

    static const ColorgenLinear white_yellow_generator(0, 100, cv::Scalar(255, 255, 255), cv::Scalar(0, 255, 255));
    static const ColorgenRedGreen valence_color_generator(-100, 100);

    // Determine the display color
    cv::Scalar color = cv::Scalar(255, 255, 255);
    if (classifier == "valence") {
        color = valence_color_generator(value);
    }
    else if (RED_COLOR_CLASSIFIERS.count(classifier)) {
        color = cv::Scalar(0, 0, 255);
    }
    else if (GREEN_COLOR_CLASSIFIERS.count(classifier)) {
        color = cv::Scalar(0, 255, 0);
    }

    float equalizer_magnitude = value;
    if (classifier == "valence") {
        equalizer_magnitude = std::fabs(value);
    }
    drawEqualizer(classifier, equalizer_magnitude, loc, align_right, color);
}

void Visualizer::drawEqualizer(const std::string& name, const float value, const cv::Point2f& loc,
                               bool align_right, const cv::Scalar& color) {
    const int block_width = 8;
    const int block_height = 10;
    const int margin = 2;
    const int block_size = 10;
    const int max_blocks = 100 / block_size;
    int blocks = round(value / block_size);
    int i = loc.x, j = loc.y - 10;

    cv::Point2f display_loc = loc;
    const std::string label = align_right ? name + ": " : " :" + name;

    for (int x = 0; x < (100 / block_size); x++) {
        cv::Scalar scalar_clr = color;
        float alpha = 0.8;
        const int ii = (std::max)(float(i), 0.0f);
        const int jj = (std::max)(float(j), 0.0f);
        const int width = (std::min)(float(block_width), float(img.size().width - ii));
        const int height = (std::min)(float(block_height), float(img.size().height - jj));
        if (height < 0 || width < 0) {
            continue;
        }
        cv::Mat roi = img(cv::Rect(ii, jj, width, height));
        if (x >= blocks) {
            alpha = 0.3;
            scalar_clr = cv::Scalar(186, 186, 186);
        }
        cv::Mat color1(roi.size(), CV_8UC3, scalar_clr);
        cv::addWeighted(color1, alpha, roi, 1.0 - alpha, 0.0, roi);

        i += align_right ? -(margin + block_width) : (margin + block_width);
    }
    display_loc.x += align_right ? -(margin + block_width) * max_blocks : (margin + block_width) * max_blocks;
    if (align_right) {
        int baseline = 0;
        cv::Size txtSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5f, 5, &baseline);
        display_loc.x -= txtSize.width;
    }
    cv::putText(img, label, display_loc, cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(50, 50, 50), 5);
    cv::putText(img, label, display_loc, cv::FONT_HERSHEY_SIMPLEX, 0.5f, cv::Scalar(255, 255, 255), 1);
}

void Visualizer::drawHeadOrientation(std::map<Measurement, float> headAngles, const int x, int& padding,
                                     bool align_right, const cv::Scalar& color) {
    std::stringstream ss;
    ss << std::fixed << std::setw(3) << std::setprecision(1);
    for (auto& h: HEAD_ANGLES) {
        ss << headAngles.at(h.first);
        drawText(h.second, ss.str(), cv::Point(x, padding += spacing), align_right, color);
        ss.str(""); // clear the string.
    }
}

void Visualizer::showImage(int interval) {
    cv::imshow("analyze video", img);
    cv::waitKey(interval);
}

cv::Mat Visualizer::getImageData() {
    return img;
}

void Visualizer::overlayImage(const cv::Mat& foreground, cv::Mat& background, const cv::Point2i& location) {

    // start at the row indicated by location, or at row 0 if location.y is negative.
    for (int y = (std::max)(location.y, 0); y < background.rows; ++y) {
        int fY = y - location.y; // because of the translation

        // we are done of we have processed all rows of the foreground image.
        if (fY >= foreground.rows) {
            break;
        }

        // start at the column indicated by location,

        // or at column 0 if location.x is negative.
        for (int x = (std::max)(location.x, 0); x < background.cols; ++x) {
            int fX = x - location.x; // because of the translation.

            // we are done with this row if the column is outside of the foreground image.
            if (fX >= foreground.cols) {
                break;
            }

            // determine the opacity of the foregrond pixel, using its fourth (alpha) channel.
            double opacity =
                ((double)foreground.data[fY * foreground.step + fX * foreground.channels()
                    + (foreground.channels() - 1)])

                    / 255.;


            // and now combine the background and foreground pixel, using the opacity,

            // but only if opacity > 0.
            for (int c = 0; opacity > 0 && c < background.channels(); ++c) {
                unsigned char foregroundPx =
                    foreground.data[fY * foreground.step + fX * foreground.channels() + c];
                unsigned char backgroundPx =
                    background.data[y * background.step + x * background.channels() + c];
                background.data[y * background.step + background.channels() * x + c] =
                    backgroundPx * (1. - opacity) + foregroundPx * opacity;
            }
        }
    }
}

cv::Scalar ColorgenRedGreen::operator()(const float val) const {
    float norm_val = (val - red_val_) / (green_val_ - red_val_);
    norm_val = norm_val < 0.0 ? 0.0 : norm_val;
    norm_val = norm_val > 1.0 ? 1.0 : norm_val;
    const int B = 0;
    const int G = norm_val * 255;
    const int R = (1.0 - norm_val) * 255;
    return cv::Scalar(B, G, R);
}

cv::Scalar ColorgenLinear::operator()(const float val) const {
    float norm_val = (val - val1_) / (val2_ - val1_);
    const int B = color1_.val[0] * (1.0f - norm_val) + color2_.val[0] * norm_val;
    const int G = color1_.val[1] * (1.0f - norm_val) + color2_.val[1] * norm_val;
    const int R = color1_.val[2] * (1.0f - norm_val) + color2_.val[2] * norm_val;
    return cv::Scalar(B, G, R);
}