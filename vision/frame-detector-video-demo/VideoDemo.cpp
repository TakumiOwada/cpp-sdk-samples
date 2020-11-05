#include "PlottingImageListener.h"
#include "PlottingObjectListener.h"
#include "PlottingOccupantListener.h"
#include "PlottingBodyListener.h"
#include "StatusListener.h"
#include "VideoReader.h"
#include "FileUtils.h"

#include <Core.h>
#include <FrameDetector.h>
#include <SyncFrameDetector.h>

#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


#include <iostream>
#include <iomanip>

static const std::string DISPLAY_DATA_DIR_ENV_VAR = "AFFECTIVA_VISION_DATA_DIR";
static const affdex::Str DATA_DIR_ENV_VAR = STR(DISPLAY_DATA_DIR_ENV_VAR);

using namespace std;
using namespace affdex;
namespace po = boost::program_options; // abbreviate namespace

void assembleProgramOptions(po::options_description& description, ProgramOptionsVideo& program_options) {

    description.add_options()
        ("help,h", po::bool_switch()->default_value(false), "Display this help message.")
#ifdef _WIN32
        ("data,d", po::wvalue<affdex::Path>(&program_options.data_dir),
            std::string("Path to the data folder. Alternatively, specify the path via the environment variable "
                + DISPLAY_DATA_DIR_ENV_VAR + R"(=\path\to\data)").c_str())
        ("input,i", po::wvalue<affdex::Path>(&program_options.input_video_path)->required(), "Video file to processs")
        ("output,o", po::wvalue<affdex::Path>(&program_options.output_video_path), "Output video path.")
#else // _WIN32
        ("data,d", po::value<affdex::Path>(&program_options.data_dir),
         (std::string("Path to the data folder. Alternatively, specify the path via the environment variable ")
             + DATA_DIR_ENV_VAR + "=/path/to/data").c_str())
        ("input,i", po::value<affdex::Path>(&program_options.input_video_path)->required(), "Video file to processs")
        ("output,o", po::value<affdex::Path>(&program_options.output_video_path), "Output video path.")
#endif // _WIN32
        ("sfps",
         po::value<float>(&program_options.sampling_frame_rate)->default_value(-1.0f),
         "Input sampling frame rate. Default is 0, which means the app will respect the video's FPS and read all frames")
        ("draw", po::value<bool>(&program_options.draw_display)->default_value(true), "Draw video on screen.")
        ("numFaces", po::value<unsigned int>(&program_options.num_faces)->default_value(5), "Number of faces to be "
                                                                                            "tracked.")
        ("loop", po::bool_switch(&program_options.loop)->default_value(false), "Loop over the video being processed.")
        ("face_id",
         po::bool_switch(&program_options.draw_id)->default_value(false),
         "Draw face id on screen. Note: Drawing to screen should be enabled.")
        ("quiet,q",
         po::bool_switch(&program_options.disable_logging)->default_value(false),
         "Disable logging to console")
        ("object", "Enable object detection")
        ("occupant", "Enable occupant detection, also enables body and face detection")
        ("body", "Enable body detection")
        ("drowsiness", "Enable drowsiness detection, will be used only if no other detection types are enabled");

}

void processObjectVideo(vision::SyncFrameDetector& detector, std::ofstream& csv_file_stream,
                        ProgramOptionsVideo& program_options) {

    // configure the Detector by enabling features
    detector.enable({vision::Feature::CHILD_SEATS, vision::Feature::PHONES});

    // prepare listeners
    PlottingObjectListener object_listener(csv_file_stream,
                                           program_options.draw_display,
                                           !program_options.disable_logging,
                                           program_options.draw_id,
                                           {{vision::Feature::CHILD_SEATS, 1000}, {vision::Feature::PHONES, 1000}},
                                           detector.getCabinRegionConfig().getRegions());
    StatusListener status_listener;

    // configure the Detector by assigning listeners
    detector.setObjectListener(&object_listener);
    detector.setProcessStatusListener(&status_listener);

    // start the detector
    detector.start();

    do {
        // the VideoReader will handle decoding frames from the input video file
        VideoReader video_reader(program_options.input_video_path);

        cv::Mat mat;
        Timestamp timestamp_ms;
        while (video_reader.GetFrame(mat, timestamp_ms)) {
            // create a Frame from the video input and process it with the Detector
            vision::Frame
                f(mat.size().width, mat.size().height, mat.data, vision::Frame::ColorFormat::BGR, timestamp_ms);
            object_listener.startTimer();
            detector.process(f);
            object_listener.processResults(f);

            //To save output video file
            if (program_options.write_video) {
                program_options.output_video << object_listener.getImageData();
            }
        }

        cout << "******************************************************************\n"
             << "Percent of samples w/objects present: " << object_listener.getSamplesWithObjectsPercent() << "%"
             << endl
             << "Object types detected: " << object_listener.getObjectTypesDetected() << endl
             << "Objects detected in regions " << object_listener.getObjectRegionsDetected() << endl
             << "Object callback interval: " << object_listener.getCallBackInterval() << endl
             << "Average processed fps: "
             << object_listener.totalFramesCount() * 1.0f / object_listener.getTotalTimeToProcessFrames()
             << " (detector ran every " << object_listener.getCallBackInterval()<< ")\n"
             << "******************************************************************\n";

        detector.reset();
        object_listener.reset();
    } while (program_options.loop);
}

void processOccupantVideo(vision::SyncFrameDetector& detector, std::ofstream& csv_file_stream,
                          ProgramOptionsVideo& program_options) {

    // configure the Detector by enabling features
    detector.enable(vision::Feature::FACES);
    detector.enable(vision::Feature::BODIES);
    detector.enable(vision::Feature::OCCUPANTS);

    // prepare listeners
    PlottingOccupantListener occupant_listener(csv_file_stream, program_options.draw_display, !program_options
        .disable_logging, program_options.draw_id, 500, detector.getCabinRegionConfig().getRegions());
    StatusListener status_listener;

    // configure the Detector by assigning listeners
    detector.setOccupantListener(&occupant_listener);
    detector.setProcessStatusListener(&status_listener);

    // start the detector
    detector.start();

    do {
        // the VideoReader will handle decoding frames from the input video file
        VideoReader video_reader(program_options.input_video_path);

        cv::Mat mat;
        Timestamp timestamp_ms;
        while (video_reader.GetFrame(mat, timestamp_ms)) {
            // create a Frame from the video input and process it with the Detector
            vision::Frame
                f(mat.size().width, mat.size().height, mat.data, vision::Frame::ColorFormat::BGR, timestamp_ms);
            occupant_listener.startTimer();
            detector.process(f);
            occupant_listener.processResults(f);

            //To save output video file
            if (program_options.write_video) {
                program_options.output_video << occupant_listener.getImageData();
            }
        }

        cout << "******************************************************************\n"
             << "Percent of samples w/occupants present: " << occupant_listener.getSamplesWithOccupantsPercent()
             << "%\n"
             << "Occupants detected in regions:  " << occupant_listener.getOccupantRegionsDetected() << endl
             << "Occupant callback interval: " << occupant_listener.getCallbackInterval() << "ms\n"
             << "Average processed fps: "
             << occupant_listener.totalFramesCount() * 1.0f / occupant_listener.getTotalTimeToProcessFrames()
             << " (detector ran every " << occupant_listener.getCallbackInterval()<< "ms)\n"
             << "******************************************************************\n";

        detector.reset();
        occupant_listener.reset();
    } while (program_options.loop);
}

void processBodyVideo(vision::SyncFrameDetector& detector, std::ofstream& csv_file_stream,
                      ProgramOptionsVideo& program_options) {

    // configure the Detector by enabling features
    detector.enable(vision::Feature::BODIES);

    // prepare listeners
    PlottingBodyListener body_listener(csv_file_stream, program_options.draw_display, !program_options
        .disable_logging, program_options.draw_id, 500);
    StatusListener status_listener;

    // configure the Detector by assigning listeners
    detector.setBodyListener(&body_listener);
    detector.setProcessStatusListener(&status_listener);

    // start the detector
    detector.start();

    do {
        // the VideoReader will handle decoding frames from the input video file
        VideoReader video_reader(program_options.input_video_path);

        cv::Mat mat;
        Timestamp timestamp_ms;
        while (video_reader.GetFrame(mat, timestamp_ms)) {
            // create a Frame from the video input and process it with the Detector
            vision::Frame
                f(mat.size().width, mat.size().height, mat.data, vision::Frame::ColorFormat::BGR, timestamp_ms);

            body_listener.startTimer();
            detector.process(f);
            body_listener.processResults(f);

            //To save output video file
            if (program_options.write_video) {
                program_options.output_video << body_listener.getImageData();
            }
        }

        cout << "******************************************************************\n"
             << "Percent of samples w/bodies present: " << body_listener.getSamplesWithBodiesPercent()
             << "%\n"
             << "Body callback interval: " << body_listener.getCallbackInterval() << "ms\n"
             << "Average processed fps: "
             << body_listener.totalFramesCount() * 1.0f / body_listener.getTotalTimeToProcessFrames()
             << " (detector ran every " << body_listener.getCallbackInterval()<< "ms)\n"
             << "******************************************************************\n";

        detector.reset();
        body_listener.reset();
    } while (program_options.loop);
}

void processFaceVideo(vision::SyncFrameDetector& detector,
                      std::ofstream& csv_file_stream,
                      ProgramOptionsVideo& program_options) {
    // configure the Detector by enabling features
    detector.enable({vision::Feature::EMOTIONS, vision::Feature::EXPRESSIONS, vision::Feature::IDENTITY,
                     vision::Feature::APPEARANCES, vision::Feature::GAZE});

    if(program_options.show_drowsiness) {
        detector.enable( vision::Feature::DROWSINESS);
    }

    // prepare listeners
    PlottingImageListener image_listener(csv_file_stream, program_options);

    StatusListener status_listener;

    // configure the Detector by assigning listeners
    detector.setImageListener(&image_listener);
    detector.setProcessStatusListener(&status_listener);

    // start the detector
    detector.start();

    do {
        // the VideoReader will handle decoding frames from the input video file
        VideoReader video_reader(program_options.input_video_path);

        cv::Mat mat;
        Timestamp timestamp_ms;
        while (video_reader.GetFrame(mat, timestamp_ms)) {
            // create a Frame from the video input and process it with the Detector
            vision::Frame
                f(mat.size().width, mat.size().height, mat.data, vision::Frame::ColorFormat::BGR, timestamp_ms);

            image_listener.startTimer();
            detector.process(f);
            image_listener.processResults(f);

            //To save output video file
            if (program_options.write_video) {
                program_options.output_video << image_listener.getImageData();
            }
        }

        cout << "******************************************************************\n"
             << "Processed Frame count: " << image_listener.getProcessedFrames() << endl
             << "Frames w/faces: " << image_listener.getFramesWithFaces() << endl
             << "Percent of frames w/faces: " << image_listener.getFramesWithFacesPercent() << "%\n"
             << "Average processed fps: " <<
             image_listener.totalFramesCount() * 1.0f / image_listener.getTotalTimeToProcessFrames() << endl
             << "******************************************************************\n";

        detector.reset();
        image_listener.reset();
    } while (program_options.loop);
}

bool verifyTypeOfProcess(const po::variables_map& args,
                         std::string& detection_type_str,
                         ProgramOptionsVideo& program_options) {

    //Check for object or occupant or body argument present or not. If nothing is present then enable face by default.
    const bool is_occupant = args.count("occupant");
    const bool is_object = args.count("object");
    const bool is_body = args.count("body");
    const bool is_all = is_occupant && is_object && is_body;

    if ((is_occupant && is_object) || (is_object && is_body) || (is_body && is_occupant) || is_all) {
        return false;
    }
    else if (args.count("object")) {
        std::cout << "Setting up object detection\n";
        program_options.detection_type = program_options.OBJECT;
        detection_type_str = "_objects";
    }
    else if (args.count("occupant")) {
        std::cout << "Setting up occupant detection\n";
        program_options.detection_type = program_options.OCCUPANT;
        detection_type_str = "_occupants";
    }
    else if (args.count("body")) {
        std::cout << "Setting up body detection\n";
        program_options.detection_type = program_options.BODY;
        detection_type_str = "_bodies";
    }
    else {
        detection_type_str = "_faces";
        //check for drowsiness only when faces are enabled
        program_options.show_drowsiness = args.count("drowsiness");
        //append _drowsiness to csv file
        if(program_options.show_drowsiness)
            detection_type_str += "_drowsiness";
        std::cout << "Setting up face detection\n";
    }
    return true;
}

int main(int argsc, char** argsv) {

    //setting up output precision
    const int precision = 2;
    std::cerr << std::fixed << std::setprecision(precision);
    std::cout << std::fixed << std::setprecision(precision);

    //Gathering program options
    ProgramOptionsVideo program_options;
    po::options_description
        description("Project for demoing the Affectiva Detector class (processing video files).");
    assembleProgramOptions(description, program_options);

    //verifying command line arguments
    po::variables_map args;
    try {
        po::store(po::command_line_parser(argsc, argsv).options(description).run(), args);
        if (args["help"].as<bool>()) {
            std::cout << description << std::endl;
            return 0;
        }
        po::notify(args);
    }
    catch (po::error& e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << "For help, use the -h option." << std::endl << std::endl;
        return 1;
    }

    //To set CSV file's suffix
    std::string detection_type_str;

    if (!verifyTypeOfProcess(args, detection_type_str, program_options)) {
        std::cerr << "ERROR: Can't use multiple detection types at the same time\n\n";
        std::cerr << "For help, use the -h option.\n\n";
        return 1;
    }

    program_options.write_video = args.count("output");
    if (program_options.write_video) {
        // must use .avi extension!
        string output_ext = boost::filesystem::extension(program_options.output_video_path);
        if (output_ext != ".avi") {
            std::cerr << "Invalid output file extension, must use .avi\n";
            return 1;
        }
    }

    // set data_dir to env_var if not set on cmd line
    program_options.data_dir = validatePath(program_options.data_dir, DATA_DIR_ENV_VAR);

    if (program_options.draw_id && !program_options.draw_display) {
        std::cerr << "Can't draw face id while drawing to screen is disabled\n";
        std::cerr << description << std::endl;
        return 1;
    }

    std::shared_ptr<vision::SyncFrameDetector> detector;

    try {
        // create the Detector
        detector = std::make_shared<vision::SyncFrameDetector>(program_options.data_dir, program_options.num_faces);

        //initialize the output file
        boost::filesystem::path pathObj(program_options.input_video_path);

        std::string csv_path_new = pathObj.stem().string();

        csv_path_new += detection_type_str + ".csv";
        boost::filesystem::path csv_path(csv_path_new);
        std::ofstream csv_file_stream(csv_path.c_str());

        if (!csv_file_stream.is_open()) {
            std::cerr << "Unable to open csv file " << csv_path << std::endl;
            return 1;
        }

        //Get resolution and fps from input video
        cv::VideoCapture video(program_options.input_video_path);
        auto sniffed_fps = static_cast<float>(video.get(CV_CAP_PROP_FPS));
        int frameHeight = static_cast<int>(video.get(CV_CAP_PROP_FRAME_HEIGHT));
        int frameWidth = static_cast<int>(video.get(CV_CAP_PROP_FRAME_WIDTH));
        video.release();
        if (program_options.sampling_frame_rate <= 0) {
            // If user did not specify --sfps (i.e. // default of 0), used the sniffed_fps
            program_options.sampling_frame_rate = sniffed_fps;
            std::cout << "Using estimated video FPS for output video: " << sniffed_fps <<std::endl;
        }

        //Setup video writer
        if (program_options.write_video) {
            program_options.output_video.open(program_options.output_video_path,
                                              CV_FOURCC('D', 'X', '5', '0'),
                                              program_options.sampling_frame_rate,
                                              cv::Size(frameWidth, frameHeight),
                                              true);

            if (!program_options.output_video.isOpened()) {
                std::cerr << "Error opening output video: " << program_options.output_video_path << std::endl;
                return 1;
            }
        }

        switch (program_options.detection_type) {
            case program_options.OBJECT:
                processObjectVideo(*detector, csv_file_stream, program_options);
                break;
            case program_options.OCCUPANT:
                processOccupantVideo(*detector, csv_file_stream, program_options);
                break;
            case program_options.BODY:
                processBodyVideo(*detector, csv_file_stream, program_options);
                break;
            case program_options.FACE:
                processFaceVideo(*detector, csv_file_stream, program_options);
                break;
            default:
                std::cerr << "This should never happen " << program_options.detection_type << std::endl;
                return 1;
        }

        if (detector) {
            detector->stop();
        }
        csv_file_stream.close();

        std::cout << "Output written to file: " << csv_path << std::endl;
    }
    catch (std::exception& ex) {
        StatusListener::printException(ex);
        // if video_reader couldn't load the video/image, it will throw. Since the detector was started before initializing the video_reader, We need to call `detector->stop()` to avoid crashing
        if (detector) {
            detector->stop();
        }
        return 1;
    }
    return 0;
}
