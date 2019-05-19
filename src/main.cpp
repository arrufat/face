#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <chrono>
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

using namespace dlib;
using namespace std;
typedef std::chrono::high_resolution_clock Clock;

// Neural network definition for face detection
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// Neural network definition for face recognition (ResNet50)
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

class FaceDetector
{
    public:
        FaceDetector()
        {
            this->light_detector = get_frontal_face_detector();
            this->detector = &FaceDetector::light_detect;
        }
        FaceDetector(const string net_path)
        {
            deserialize(net_path) >> this->dnn_detector;
            this->detector = &FaceDetector::dnn_detect;
        }
        ~FaceDetector() {}

        std::vector<rectangle> detect(matrix<rgb_pixel> img)
        {
            return (this->*detector)(img);
        }
    private:
        typedef std::vector<rectangle> (FaceDetector::*DetectorPtr) (matrix<rgb_pixel>);
        DetectorPtr detector;
        net_type dnn_detector;
        frontal_face_detector light_detector ;
        std::vector<rectangle> light_detect(matrix<rgb_pixel> img)
        {
            return this->light_detector(img);
        }
        std::vector<rectangle> dnn_detect(matrix<rgb_pixel> img)
        {
            std::vector<rectangle> dets;
            auto mmod_rects = this->dnn_detector(img);
            for (auto &r : mmod_rects)
            {
                dets.push_back(r);
            }
            return dets;
        }
};

class FaceAligner
{
    public:
        FaceAligner(const string model_path)
        {
            deserialize(model_path) >> this->pose_model;
        };
        full_object_detection align(matrix<rgb_pixel> img, rectangle det)
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
        cout << "Usage: " << argv[0] << " [options] <list.txt>\n";
        parser.print_options();
        return EXIT_SUCCESS;
    }

    try
    {
        double threshold = get_option(parser, "threshold", 0.5);
        string enroll_dir = get_option(parser, "enroll-dir", "enrollment");
        int pyramid_levels = get_option(parser, "pyramid-levels", 1);
        double scale_factor = get_option(parser, "scale-factor", 1.0);

        string video_path;
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
            cerr << "Unable to connect to camera" << endl;
            return 1;
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
        deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> anet;

        // A mapping between face_descriptors and indentities
        std::map<matrix<float, 0, 1>, string> enr_map;
        // ----------------- ENROLLMENT -----------------
        {
            auto t1 = Clock::now();
            cout << "Scanning '" << enroll_dir << "' directory and generating face descriptors." << endl;
            directory root(enroll_dir);
            auto files = get_files_in_directory_tree(root, match_endings(".jpg .JPG .png .PNG"), 1);
            std::vector<string> names;
            std::vector<matrix<rgb_pixel>> enr_imgs;
            std::vector<full_object_detection> enr_shapes;
            for (auto &f : files)
            {
                matrix<rgb_pixel> enr_img;
                load_image(enr_img, f.full_name());
                names.push_back(get_parent_directory(f).name());
                enr_imgs.push_back(enr_img);
            }
            // Detect faces on enrollment images
            for (auto &enr_img : enr_imgs)
            {
                auto dets = face_detector.detect(enr_img);
                // Align and store detected faces
                for (auto &det : dets)
                {
                    enr_shapes.push_back(face_aligner.align(enr_img, det));
                }
            }
            // Make sure we found as many faces as enrollment images
            DLIB_CASSERT(names.size() == enr_shapes.size());
            std::vector<matrix<rgb_pixel>> enr_faces;
            for (size_t i = 0; i < enr_shapes.size(); i++)
            {
                matrix<rgb_pixel> face_chip;
                extract_image_chip(enr_imgs[i], get_face_chip_details(enr_shapes[i], 150, 0.25), face_chip);
                enr_faces.push_back(move(face_chip));
            }
            std::vector<matrix<float, 0, 1>> face_descriptors = anet(enr_faces);
            DLIB_CASSERT(names.size() == face_descriptors.size());
            for (size_t i = 0; i < names.size(); i++)
            {
                enr_map[face_descriptors[i]] = names[i];
            }
            auto t2 = Clock::now();
            cout << "Computed " << face_descriptors.size() << " face descriptors in ";
            cout << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() * 1e-9 << " seconds" << endl;
        }
        // ----------------------------------------------

        // create the display windows
        image_window win, det_win;
        win.set_title("Webcam");
        det_win.set_title("Face detections");

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            auto t1 = Clock::now();
            // Grab a frame
            cv::Mat cv_tmp;
            if (!vid_src.read(cv_tmp))
            {
                break;
            }
            // Convert the OpenCV BRG image into a Dlib RGB image
            cv::Mat cv_tmp_rgb;
            cv::cvtColor(cv_tmp, cv_tmp_rgb, cv::COLOR_BGR2RGB);
            cv_image<rgb_pixel> cv_img(cv_tmp_rgb);

            // Handle the mirroring
            dlib::matrix<rgb_pixel> mir_img, img;
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
            std::vector<full_object_detection> shapes;
            // vector to store all aligned faces
            std::vector<matrix<rgb_pixel>> faces;

            int cur_pyr_lvl = 1;
            while (cur_pyr_lvl < pyramid_levels)
            {
                pyramid_up(img);
                cur_pyr_lvl++;
            }

            // detect faces in current frame
            auto dets = face_detector.detect(img);
            // store alignment information for each face
            for (auto &det : dets)
            {
                shapes.push_back(face_aligner.align(img, det));
            }

            // align detected faces
            for (auto &shape : shapes)
            {
                matrix<rgb_pixel> face_chip;
                extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
                faces.push_back(move(face_chip));
            }

            std::vector<matrix<float, 0, 1>> face_descriptors = anet(faces);

            for (size_t i = 0; i < face_descriptors.size(); i++)
            {
                for (auto const &entry : enr_map)
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
            auto t2 = Clock::now();
            cout << "Detected faces: " << shapes.size() << "\tTime to process frame: ";
            cout << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() * 1e-9 << " seconds\r" << flush;
        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

