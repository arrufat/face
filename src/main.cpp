#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
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

int main(int argc, char** argv)
{
    dlib::command_line_parser parser;
    parser.add_option("input", "Path to video file to process", 1);
    parser.add_option("mirror", "Mirror mode (left-right flip)");
    parser.add_option("light", "Use a lighter detection model");
    parser.add_option("threshold", "Face recognition threshold (default: 0.6)", 1);
    parser.add_option("enroll-dir", "Path to the enrollment directory (default: enrollment)", 1);
    parser.add_option("h","Display this help message.");
    parser.parse(argc, argv);

    if (parser.option("h"))
    {
        cout << "Usage: " << argv[0] << " [options] <list.txt>\n";
        parser.print_options();
        return EXIT_SUCCESS;
    }

    try
    {
        double threshold = get_option(parser, "threshold", 0.6);
        string enroll_dir = get_option(parser, "enroll-dir", "enrollment");

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
            vid_src = cap;
        }

        // open the webcam
        if (!vid_src.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        // create the display windows
        image_window win, det_win;
        win.set_title("Webcam");
        det_win.set_title("Face detections");

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("models/shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Load the neural network for face detection
        net_type net;
        deserialize("models/mmod_human_face_detector.dat") >> net;

        // Load the neural network for face recognition
        anet_type anet;
        deserialize("models/dlib_face_recognition_resnet_model_v1.dat") >> anet;

        std::map<matrix<float, 0, 1>, string> enr_map;
        // ----------------- ENROLLMENT -----------------
        {
            cout << "Scanning enrollment directory: " << enroll_dir << endl;
            directory root(enroll_dir);
            auto files = get_files_in_directory_tree(root, match_endings(".jpg .JPG .png .PNG"), 1);
            std::vector<string> names;
            std::vector<matrix<rgb_pixel>> enr_imgs;
            std::vector<full_object_detection> enr_shapes;
            for (auto f : files)
            {
                matrix<rgb_pixel> enr_img;
                load_image(enr_img, f.full_name());
                names.push_back(get_parent_directory(f).name());
                enr_imgs.push_back(enr_img);
            }

            if (parser.option("light"))
            {
                for (auto enr_img : enr_imgs)
                {
                    auto dets = detector(enr_img);
                    for (auto d : dets)
                    {
                        enr_shapes.push_back(pose_model(enr_img, d));
                    }
                }
            }
            else
            {
                for (auto enr_img : enr_imgs)
                {
                    auto dets = net(enr_img);
                    for (auto d : dets)
                    {
                        enr_shapes.push_back(pose_model(enr_img, d));
                    }
                }
            }
            // Make sure we found as many faces as enrollment images
            assert(names.size() == enr_shapes.size());
            std::vector<matrix<rgb_pixel>> enr_faces;
            for (size_t i = 0; i < enr_shapes.size(); i++)
            {
                matrix<rgb_pixel> face_chip;
                extract_image_chip(enr_imgs[i], get_face_chip_details(enr_shapes[i], 150, 0.25), face_chip);
                enr_faces.push_back(move(face_chip));
            }
            std::vector<matrix<float, 0, 1>> face_descriptors = anet(enr_faces);
            cout << "Computed " << face_descriptors.size() << " face_descriptors" << endl;
            assert(names.size() == face_descriptors.size());
            for (size_t i = 0; i < names.size(); i++)
            {
                enr_map[face_descriptors[i]] = names[i];
            }
        }
        // ----------------------------------------------

        // vector to store all face landmarks
        std::vector<full_object_detection> shapes;
        // vector to store all detected faces
        std::vector<matrix<rgb_pixel>> faces;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
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

            // Select the light or the neural network approach
            if (parser.option("light"))
            {
                // Detect faces
                std::vector<rectangle> faces = detector(img);
                // Find the pose of each face.
                for (unsigned long i = 0; i < faces.size(); ++i)
                {
                    shapes.push_back(pose_model(img, faces[i]));
                }
            }
            else
            {
                // Detect faces using the neural network
                while (img.size() < 1800 * 1800) {
                    pyramid_up(img);
                }
                auto dets = net(img);

                for (unsigned long i = 0; i < dets.size(); i++)
                {
                    full_object_detection shape = pose_model(img, dets[i]);
                    shapes.push_back(shape);
                }
            }

            for (auto shape : shapes)
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
                        cv::putText(cv_face, entry.second, cv::Point(50, 125), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(255, 102, 0), 1);
                        break;
                    }
                }
            }

            det_win.set_image(tile_images(faces));
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(render_face_detections(shapes));
            shapes.clear();
            faces.clear();
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

