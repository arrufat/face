#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dlib/cmd_line_parser.h>
#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

int main(int argc, char** argv)
{
    dlib::command_line_parser parser;
    parser.add_option("mirror", "Mirror mode (left-right flip)");
    parser.add_option("fast", "Use a faster, less accurate model");
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
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("models/shape_predictor_68_face_landmarks.dat") >> pose_model;

        // Load the neural network for face detection
        net_type net;
        deserialize("models/mmod_human_face_detector.dat") >> net;
        // vector to store all face landmarks
        std::vector<full_object_detection> shapes;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv::Mat tmp_rgb;
            cv::cvtColor(temp, tmp_rgb, cv::COLOR_BGR2RGB);
            cv_image<rgb_pixel> cv_img(tmp_rgb);
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

            if (parser.option("fast"))
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
                /* while (img.size() < 1800 * 1800) { */
                /*     pyramid_up(img); */
                /* } */
                auto dets = net(img);
                for (unsigned long i = 0; i < dets.size(); i++)
                {
                    full_object_detection shape = pose_model(img, dets[i]);
                    shapes.push_back(shape);
                }
            }
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(render_face_detections(shapes));
            shapes.clear();
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

