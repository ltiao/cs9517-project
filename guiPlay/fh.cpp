#include <stdio.h>
#include <iostream>
#include <sstream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/flann/flann.hpp"

using namespace std;
using namespace cv;

#define SURF_D 0
#define SIFT_D 1
#define ORB_D 2
#define FAST_D 4

#define ORB_E 5
#define SURF_E 6
#define SIFT_E 7
#define BRIEF_E 8
#define FREAK_E 9

#define BFHAM_M 10
#define FLANN_M 11

int detector_code, extractor_code, matcher_code;
int offset;

// cli: -d <detector-code> -e <extractor-code>
// defaults to SURF SURF flann
// matcher BF(ham) for orb, flann otherwise
/* flags:
    1. Preprocess the entire thing. Might be a good idea to detect the number
       of frames ahead of time. Can detect up to 3 images
    2. realtime, which gives us access to the special gui. Can select regions and hold them
    3. frame skip flag. If set, to i, will only read in ever ith frame.

    a) When we are dealing with multiple images, only draw the lines for 
        the most prominent image. All 3 images will be displayed in another frame which
        holds the cropped images.
    */

// Mat img_object;
// vector<KeyPoint> keypoints_object;
// Mat descriptors_object;

void useDetector(Mat img, vector<KeyPoint> &keypoints) {
    if (detector_code == SURF_D) {
        SurfFeatureDetector detector;
        // FeatureDetector detector;
        detector.detect(img, keypoints);
    } else if (detector_code == SIFT_D) {
        SiftFeatureDetector detector;
        detector.detect(img, keypoints);
    } else if (detector_code == ORB_D) {
        OrbFeatureDetector detector;
        detector.detect(img, keypoints);
    } else if (detector_code == FAST_D) {
        FastFeatureDetector detector;
        detector.detect(img, keypoints);
    }
}

void useExtractor(Mat img, vector<KeyPoint> &keypoints, Mat &descriptors) {
    if (extractor_code == ORB_E) {
        OrbDescriptorExtractor extractor;
        extractor.compute(img, keypoints, descriptors);
    } else if (extractor_code == SURF_E) {
        SurfDescriptorExtractor extractor;
        extractor.compute(img, keypoints, descriptors);
    } else if (extractor_code == SIFT_E) {
        SiftDescriptorExtractor extractor;
        extractor.compute(img, keypoints, descriptors);
    } else if (extractor_code == BRIEF_E) {
        BriefDescriptorExtractor extractor;
        extractor.compute(img, keypoints, descriptors);
    }
}

void useMatcher(Mat query, Mat train, vector<vector<DMatch> > &matches) {
    if (matcher_code == BFHAM_M) {
        BFMatcher matcher(NORM_HAMMING);
        matcher.knnMatch(query, train, matches, 2);
    } else if (matcher_code == FLANN_M) {
        FlannBasedMatcher matcher;
        matcher.knnMatch(query, train, matches, 2);
    }
}

vector<DMatch> goodMatches(vector<vector<DMatch> > matches, int rows, double thresh = 0.9) {
    vector< DMatch > good_matches;
    for (int i = 0; i < rows; i++) {
        if (matches[i][0].distance < thresh * matches[i][1].distance) {
            good_matches.push_back(matches[i][0]);
        }
    }
    return good_matches;
}

Mat drawHomography(vector<KeyPoint> keypoints_obj, 
                   vector<KeyPoint> keypoints_scene,
                   vector<DMatch> good_matches,
                   Mat img_matches, Mat img_obj, double &score, Mat &mask) {
     //-- Localize the object
    vector<Point2f> obj, scene;
    for( int i = 0; i < good_matches.size(); i++ ) {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_obj[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

    // cout << "image is about to show\n";

    if (good_matches.size() < 4) return img_matches;

    // Mat mask;
    Mat H = findHomography( obj, scene, CV_RANSAC, 1, mask);

    score = (sum(mask))[0]*1.0 / good_matches.size()*1.0;

    // cout << " score: " << score;

    //-- Get the corners from the image_1 ( the object to be "detected" )
    vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); 
    obj_corners[1] = cvPoint( img_obj.cols, 0 );
    obj_corners[2] = cvPoint( img_obj.cols, img_obj.rows ); 
    obj_corners[3] = cvPoint( 0, img_obj.rows );
    vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )

    line( img_matches, scene_corners[0], scene_corners[1], Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 4 );
    
    return img_matches;
}

Mat process_images(vector<Mat> img_objects, vector<vector<KeyPoint> > keypoints_objects, 
                   vector<Mat> descriptors_objects, Mat img_scene, 
                   vector<vector<DMatch> > &good_matches) {

    vector<KeyPoint> keypoints_scene;
    Mat descriptors_scene;
    vector<vector<vector<DMatch> > > matches(img_objects.size());

    useDetector(img_scene, keypoints_scene);
    useExtractor(img_scene, keypoints_scene, descriptors_scene);

    for (int i = 0; i < img_objects.size(); i++) {
        useMatcher(descriptors_objects[i], descriptors_scene, matches[i]);
        good_matches[i] = goodMatches(matches[i], descriptors_objects[i].rows);
    }

    Mat img_matches = img_scene.clone();
    vector<double> scores(img_objects.size());
    vector<Mat> masks(img_objects.size());

    for (int i = 0; i < img_objects.size(); i++) {
        img_matches = drawHomography(keypoints_objects[i], keypoints_scene, 
                                     good_matches[i], img_matches, img_objects[i], 
                                     scores[i], masks[i]);
        // cout << "| for: " << i << " score: " << scores[i];
    }
    // cout << endl;

    int b = *(max_element(scores.begin(), scores.end()));
    Mat img_drawn;

    drawMatches( img_objects[b], keypoints_objects[b], img_matches, keypoints_scene,
                 good_matches[b], img_drawn, Scalar::all(-1), Scalar::all(-1),
                 masks[b], DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    offset = img_objects[b].cols;
    // cout << "offset: " << offset << endl;;

    return img_drawn;
}

void printUsage() {
    cout << "Usage [-p] [-v input_video] [-i input_image] [-d detector_code] " 
         << "[-e extractor_code]" << endl;
}

string ctos(char a[]) {
    ostringstream ss;
    ss << a;
    return ss.str();
}

int getCode(char a[]) {
    string s = ctos(a);
    int r = 0;
    if (s == "SURF_D") r = SURF_D;
    if (s == "SIFT_D") r = SIFT_D;
    if (s == "ORB_D")  r = ORB_D;
    if (s == "FAST_D") r = FAST_D;
    if (s == "ORB_E")  r = ORB_E;
    if (s == "SURF_E") r = SURF_E;
    if (s == "SIFT_E") r = SIFT_E;
    return r;
}



// globals since a mouse action can update these
vector<Mat> img_objects;
vector<vector<KeyPoint> > keypoints_objects;
vector<Mat> descriptors_objects;
vector<KeyPoint> keypoints_object;
Mat descriptors_object;

void makeSomeStuff() {

    for (int i = 0; i < img_objects.size(); i++) {
        useDetector(img_objects[i], keypoints_object);
        useExtractor(img_objects[i], keypoints_object, descriptors_object);
        keypoints_objects.push_back(keypoints_object);
        descriptors_objects.push_back(descriptors_object);
        cout << "num: " << i << " num keypoints:" << keypoints_object.size() << endl;;
    }
}

Mat img, orig_img;
bool drawing = false;
int sx, sy, ex, ey;

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN ) {
        drawing = true;
        sx = x; sy = y;  
        ex = x; ey = y;
    } else if (event == EVENT_LBUTTONUP) {
        drawing = false;
        if (sx != x && sy != y) {
            if (sx > x) swap(sx, x);
            if (sy > y) swap(sy, y);
            Mat disp = orig_img(Rect(sx-offset, sy, x-sx, y-sy)).clone();
            // imshow("Crop", disp);

            img_objects.clear();
            img_objects.push_back(disp);
            makeSomeStuff();

            // add to makesomestuff here
        }
    } else if (event == EVENT_MOUSEMOVE ) {
        if (drawing) { ex = x; ey = y; } 
    }    
}

void drawImage() {
    if (drawing) {
        Mat img2 = img.clone();
        rectangle(img2, Point(sx,sy), Point(ex,ey), Scalar(255,0,0), 2, 8);   
        imshow("Movie", img2);
    } else {
        imshow("Movie", img);
    }
}



int main( int argc, char** argv ) {

    // options
    bool preprocess = false;
    string input = "";
    vector<string> images;

    // default detector and extractor codes
    detector_code = SURF_D;
    extractor_code = SURF_E;


    for (int i = 1; i < argc; i++) {
        if (ctos(argv[i]) == "-p") preprocess = true;
        if (ctos(argv[i]) == "-v") input = ctos(argv[i+1]);
        if (ctos(argv[i]) == "-i") images.push_back(ctos(argv[i+1]));
        if (ctos(argv[i]) == "-d") 
            if (getCode(argv[i+1]) > 0)
                detector_code = getCode(argv[i+1]);
        if (ctos(argv[i]) == "-e") 
            if (getCode(argv[i+1]) > 0)
                extractor_code = getCode(argv[i+1]);
    }

    // get the right matcher
    if (extractor_code == ORB_E || extractor_code == BRIEF_E) matcher_code = BFHAM_M;
    else matcher_code = FLANN_M;

    // read in the images
    for (int i = 0; i < images.size(); i++) img_objects.push_back(imread(images[i]));

    VideoCapture cap;
    if (input.length() == 0) {
        VideoCapture t(0);
        cap = t;
    } else {
        VideoCapture t(input);
        cap = t; 
    }

    if(!cap.isOpened())  // check if we succeeded
        return -1;

    makeSomeStuff();

    namedWindow("Movie", 1);
    setMouseCallback("Movie", CallBackFunc, NULL);

    clock_t startTime = clock();

    Mat img_scene;
    // vector<Mat> processed;
    int f = 0;
    for(;;) {
        f++;
        if (f % 10 == 0) {
            double g = double(clock() - startTime)/ (double) CLOCKS_PER_SEC;
            cout << "frame: " << f << " | seconds: " 
                      << g << " | fps: " << f*1.0/g << endl;
        }
        vector<vector<DMatch> > good_matches(img_objects.size());

        if (!drawing) {
            cap >> img_scene;
            if (img_scene.empty()) break;
            orig_img = img_scene;
            Mat m = process_images(img_objects, keypoints_objects, descriptors_objects, 
                                   img_scene, good_matches);
            // offset is found here
            img = m;
        }
        
        drawImage();
        

        // processed.push_back(m);

        // print out some stats
        // for (int i = 0; i < keypoints_objects.size(); i++) {
        //     cout << good_matches[i].size()*1.0 / keypoints_objects[i].size()*1.0 << " ";
        // }
        // cout << endl;

        if(waitKey(1) >= 0) break;  
    }
    return 0;
}