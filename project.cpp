#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
// #include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"


using namespace std;
using namespace cv;

int main(int argc, char** argv) {

	if(argc < 3) {
		cerr << "Usage: " << argv[0] << " <video> <patch>" << endl;
		return -1;
	}

	string video_fname = argv[1];
	string patch_fname = argv[2];

	VideoCapture cap(video_fname);
    if (!cap.isOpened()) {
    	cerr << "Unable to capture video " << video_fname << endl;
    	return -1;
    }

    Mat patch = imread(patch_fname);

    SiftFeatureDetector detector;
	
	vector<KeyPoint> keypoints_1;
	
	detector.detect(patch, keypoints_1);

    Mat edges;
    namedWindow("Image Matching", WINDOW_AUTOSIZE);
    while(true) {
        Mat frame;
        cap >> frame; // get a new frame from camera
        cvtColor(frame, edges, CV_BGR2GRAY);
        GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
        Canny(edges, edges, 0, 30, 3);
        imshow("edges", edges);
        if(waitKey(30) >= 0) break;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}