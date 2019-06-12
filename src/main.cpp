#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <dlib/assert.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/dir_nav.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <map>

namespace chrono = std::chrono;
using fseconds = chrono::duration<float>;

// Neural network definition for face detection
template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;

using net_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

// Neural network definition for face recognition (ResNet50)
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

class FaceDetector
{
    public:
        FaceDetector()
        {
            this->light_detector = dlib::get_frontal_face_detector();
            this->detector = &FaceDetector::light_detect;
        }
        FaceDetector(const std::string net_path)
        {
            dlib::deserialize(net_path) >> this->dnn_detector;
            this->detector = &FaceDetector::dnn_detect;
        }

        std::vector<dlib::rectangle> detect(dlib::matrix<dlib::rgb_pixel> img)
        {
            return (this->*detector)(img);
        }
    private:
        typedef std::vector<dlib::rectangle> (FaceDetector::*DetectorPtr) (dlib::matrix<dlib::rgb_pixel>);
        DetectorPtr detector;
        net_type dnn_detector;
        dlib::frontal_face_detector light_detector ;
        std::vector<dlib::rectangle> light_detect(dlib::matrix<dlib::rgb_pixel> img)
        {
            return this->light_detector(img);
        }
        std::vector<dlib::rectangle> dnn_detect(dlib::matrix<dlib::rgb_pixel> img)
        {
            std::vector<dlib::rectangle> dets;
            auto mmod_rects = this->dnn_detector(img);
            for (const auto &r : mmod_rects)
            {
                dets.push_back(r);
            }
            return dets;
        }
};

class FaceAligner
{
    public:
        FaceAligner(const std::string model_path)
        {
            dlib::deserialize(model_path) >> this->pose_model;
        };
        dlib::full_object_detection align(dlib::matrix<dlib::rgb_pixel> img, dlib::rectangle det)
        {
            return this->pose_model(img, det);
        }
    private:
        dlib::shape_predictor pose_model;
};

int main(int argc, char** argv)
{
    dlib::command_line_parser parser;
    parser.add_option("input", "Path to video file to process (defaults to webcam)", 1);
    parser.add_option("mirror", "Mirror mode (left-right flip)");
    parser.add_option("light", "Use a lighter detection model");
    parser.add_option("threshold", "Face recognition threshold (default: 0.5)", 1);
    parser.add_option("enroll-dir", "Enrollment directory (default: enrollment)", 1);
    parser.add_option("pyramid-levels", "Pyramid levels for the face detector (default: 1)", 1);
    parser.add_option("scale-factor", "Scaling factor for the input image (default: 1.0)", 1);
    parser.add_option("fps", "Force the frames per second for the webcam", 1);
    parser.add_option("help","Display this help message.");
    parser.parse(argc, argv);

    if (parser.option("help"))
    {
        std::cout << "Usage: " << argv[0] << " [options] <list.txt>" << std::endl;
        parser.print_options();
        return EXIT_SUCCESS;
    }

    try
    {
        double threshold = get_option(parser, "threshold", 0.5);
        std::string enroll_dir = get_option(parser, "enroll-dir", "enrollment");
        int pyramid_levels = get_option(parser, "pyramid-levels", 1);
        double scale_factor = get_option(parser, "scale-factor", 1.0);

        std::string video_path;
        cv::VideoCapture vid_src;
        if(parser.option("input"))
        {
            video_path = parser.option("input").argument();
            cv::VideoCapture file(video_path);
            vid_src = file;
        }
        else
        {
            cv::VideoCapture cap(0);
            if (parser.option("fps")) {
                int fps = atoi(parser.option("fps").argument().c_str());
                cap.set(cv::CAP_PROP_FPS, fps);
            }
            vid_src = cap;
        }

        // open the webcam
        if (!vid_src.isOpened())
        {
            std::cerr << "Unable to connect to camera" << std::endl;
            return EXIT_FAILURE;
        }

        // Initialize the face detector
        FaceDetector face_detector;
        if (!parser.option("light"))
        {
            face_detector = FaceDetector("models/mmod_human_face_detector.dat");
        }

        // Initialize the face aligner
        FaceAligner face_aligner("models/shape_predictor_5_face_landmarks.dat");

        // Load the neural network for face recognition
        anet_type anet;
        dlib::deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> anet;

        // A mapping between face_descriptors and indentities
        std::map<dlib::matrix<float, 0, 1>, std::string> enr_map;
        // ----------------- ENROLLMENT -----------------
        {
            auto t0 = chrono::steady_clock::now();
            std::cout << "Scanning '" << enroll_dir << "' directory and generating face descriptors." << std::endl;
            dlib::directory root(enroll_dir);
            auto files = get_files_in_directory_tree(root, dlib::match_endings(".jpg .JPG .png .PNG"), 1);
            std::vector<std::string> names;
            std::vector<dlib::matrix<dlib::rgb_pixel>> enr_imgs;
            std::vector<dlib::full_object_detection> enr_shapes;
            for (const auto& f : files)
            {
                dlib::matrix<dlib::rgb_pixel> enr_img;
                load_image(enr_img, f.full_name());
                names.push_back(get_parent_directory(f).name());
                enr_imgs.push_back(enr_img);
            }
            // Detect faces on enrollment images
            for (const auto& enr_img : enr_imgs)
            {
                const auto& dets = face_detector.detect(enr_img);
                // Align and store detected faces
                for (const auto& det : dets)
                {
                    enr_shapes.push_back(face_aligner.align(enr_img, det));
                }
            }
            // Make sure we found as many faces as enrollment images
            DLIB_CASSERT(names.size() == enr_shapes.size());
            std::vector<dlib::matrix<dlib::rgb_pixel>> enr_faces;
            for (size_t i = 0; i < enr_shapes.size(); i++)
            {
                dlib::matrix<dlib::rgb_pixel> face_chip;
                extract_image_chip(enr_imgs[i], get_face_chip_details(enr_shapes[i], 150, 0.25), face_chip);
                enr_faces.push_back(std::move(face_chip));
            }
            std::vector<dlib::matrix<float, 0, 1>> face_descriptors = anet(enr_faces);
            DLIB_CASSERT(names.size() == face_descriptors.size());
            for (size_t i = 0; i < names.size(); i++)
            {
                enr_map[face_descriptors[i]] = names[i];
            }
            auto t1 = chrono::steady_clock::now();
            std::cout << std::fixed << std::setprecision(3) <<
                "Computed " << face_descriptors.size() << " face descriptors in " <<
                chrono::duration_cast<fseconds>(t1 - t0).count() << " seconds" << std::endl;
        }
        // ----------------------------------------------

        // create the display windows
        dlib::image_window win, det_win;
        win.set_title("Webcam");
        det_win.set_title("Face detections");

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            auto t0 = chrono::steady_clock::now();
            // Grab a frame
            cv::Mat cv_tmp;
            if (!vid_src.read(cv_tmp))
            {
                break;
            }
            // Convert the OpenCV BRG image into a Dlib RGB image
            cv::Mat cv_tmp_rgb;
            cv::cvtColor(cv_tmp, cv_tmp_rgb, cv::COLOR_BGR2RGB);
            dlib::cv_image<dlib::rgb_pixel> cv_img(cv_tmp_rgb);

            // Handle the mirroring
            dlib::matrix<dlib::rgb_pixel> mir_img, img;
            if (parser.option("mirror"))
            {
                dlib::flip_image_left_right(cv_img, mir_img);
                dlib::assign_image(img, mir_img);
            }
            else
            {
                dlib::assign_image(img, cv_img);
            }

            // Handle scaling
            if (scale_factor != 1.0)
            {
                dlib::resize_image(scale_factor, img);
            }

            // vector to store all face landmarks
            std::vector<dlib::full_object_detection> shapes;
            // vector to store all aligned faces
            std::vector<dlib::matrix<dlib::rgb_pixel>> faces;

            int cur_pyr_lvl = 1;
            while (cur_pyr_lvl < pyramid_levels)
            {
                pyramid_up(img);
                cur_pyr_lvl++;
            }

            // detect faces in current frame
            auto dets = face_detector.detect(img);
            // store alignment information for each face
            for (const auto& det : dets)
            {
                shapes.push_back(face_aligner.align(img, det));
            }

            // align detected faces
            for (const auto& shape : shapes)
            {
                dlib::matrix<dlib::rgb_pixel> face_chip;
                extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(std::move(face_chip));
            }

            std::vector<dlib::matrix<float, 0, 1>> face_descriptors = anet(faces);

            for (size_t i = 0; i < face_descriptors.size(); i++)
            {
                for (const auto& entry : enr_map)
                {
                    if (length(face_descriptors[i] - entry.first) < threshold)
                    {
                        cv::Mat cv_face = dlib::toMat(faces[i]);
                        cv::putText(cv_face, entry.second, cv::Point(10, 140), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(255, 102, 0), 1);
                        break;
                    }
                }
            }

            det_win.set_image(tile_images(faces));
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(render_face_detections(shapes));
            auto t1 = chrono::steady_clock::now();
            auto real_fps = 1.0 / chrono::duration_cast<fseconds>(t1 - t0).count();
            std::cout << std::fixed << std::setprecision(3) <<
                "Detected faces: " << shapes.size() << "\tFPS: " << real_fps << "    \r" << std::flush;
        }
        std::cout << std::endl;
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

