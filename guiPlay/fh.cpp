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
#define FAST_D 3
#define STAR_D 4

#define ORB_E 5
#define SURF_E 6
#define SIFT_E 7
#define BRIEF_E 8

#define BFHAM_M 9
#define FLANN_M 10

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
    if (s == "STAR_D") r = STAR_D;

    if (s == "ORB_E")  r = ORB_E;
    if (s == "SURF_E") r = SURF_E;
    if (s == "SIFT_E") r = SIFT_E;
    if (s == "BRIEF_E") r = BRIEF_E;
    
    return r;
}

int detector_code, extractor_code, matcher_code, n;

void useDetector(Mat img, vector<KeyPoint> &keypoints) {
    if (detector_code == SURF_D) {
        SurfFeatureDetector detector;
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
    } else if (detector_code == STAR_D) {
        SiftFeatureDetector detector;
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
    if (good_matches.size() < 4) return img_matches;
    
    // Mat mask;
    Mat H = findHomography( obj, scene, CV_RANSAC, 1, mask);
    score = (sum(mask))[0]*1.0 / good_matches.size()*1.0;

    //-- Get the corners from the image_1 ( the object to be "detected" )
    vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); 
    obj_corners[1] = cvPoint( img_obj.cols, 0 );
    obj_corners[2] = cvPoint( img_obj.cols, img_obj.rows ); 
    obj_corners[3] = cvPoint( 0, img_obj.rows );
    vector<Point2f> scene_corners(4);

    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches, scene_corners[0], scene_corners[1], Scalar( 0, 255, 0), 2 );
    line( img_matches, scene_corners[1], scene_corners[2], Scalar( 0, 255, 0), 2 );
    line( img_matches, scene_corners[2], scene_corners[3], Scalar( 0, 255, 0), 2 );
    line( img_matches, scene_corners[3], scene_corners[0], Scalar( 0, 255, 0), 2 );
    
    return img_matches;
}

bool chosen = false;
int focus = 0;

Mat process_images(vector<Mat> img_objects, vector<vector<KeyPoint> > keypoints_objects,
                   vector<Mat> descriptors_objects, Mat img_scene, 
                   vector<vector<DMatch> > &good_matches) {

    vector<KeyPoint> keypoints_scene;
    Mat descriptors_scene;
    vector<vector<vector<DMatch> > > matches(img_objects.size());

    useDetector(img_scene, keypoints_scene);
    useExtractor(img_scene, keypoints_scene, descriptors_scene);

    for (int i = 0; i < n; i++) {
        useMatcher(descriptors_objects[i], descriptors_scene, matches[i]);
        good_matches[i] = goodMatches(matches[i], descriptors_objects[i].rows);
    }

    Mat img_matches = img_scene.clone();
    vector<double> scores(n);
    vector<Mat> masks(n);

    for (int i = 0; i < n; i++) {
        img_matches = drawHomography(keypoints_objects[i], keypoints_scene, 
                                     good_matches[i], img_matches, img_objects[i], 
                                     scores[i], masks[i]);
    }

    int b = 0;
    if (n > 1 && !chosen) {
        double max = scores[0];
        for (int j = 1; j < n; j++) {
            if (scores[j] > max) {
                max = scores[j];
                b = j;
            }
        }
    } else if (n > 1 && chosen && focus < n) {
        b = focus;
    }
    
    Mat img_drawn;
    vector<DMatch> good_matches2 = good_matches[b];

    for (int i = 0; i < good_matches2.size(); i++) {
        swap(good_matches2[i].queryIdx, good_matches2[i].trainIdx);
    }

    drawMatches( img_matches, keypoints_scene, img_objects[b], keypoints_objects[b],
                 good_matches2, img_drawn, Scalar::all(-1), Scalar::all(-1),
                 masks[b], DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    return img_drawn;
}

Mat img, orig_img, cropped;
bool drawing = false, isCropped = false;
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
            cropped = orig_img(Rect(sx, sy, x-sx, y-sy)).clone();
            isCropped = true;
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

vector<Mat> img_objects;
vector<vector<KeyPoint> > keypoints_objects;
vector<Mat> descriptors_objects;
vector<KeyPoint> keypoints_object;

void makeSomeStuff() {

    keypoints_objects.clear();
    descriptors_objects.clear();
    
    Mat descriptors_object;
    for (int i = 0; i < img_objects.size(); i++) {
        useDetector(img_objects[i], keypoints_object);
        useExtractor(img_objects[i], keypoints_object, descriptors_object);
        keypoints_objects.push_back(keypoints_object);
        descriptors_objects.push_back(descriptors_object);
        cout << "Image: " << (i+1) << " #keypoints: " << keypoints_object.size() << endl;;
    }
}

void showInputImages() {
    namedWindow("Input 1", 1);
    imshow("Input 1", img_objects[0]);
    if (n >= 2) {
        namedWindow("Input 2", 1);
        imshow("Input 2", img_objects[1]);
    }
    if (n >= 3) {
        namedWindow("Input 3", 1);
        imshow("Input 3", img_objects[2]);
    }
}

int rot = 0;
void processCrops() {
    if (isCropped) {
        rot = (rot + 1) % 3;
        if (n == 3) img_objects[rot] = cropped;
        else img_objects.push_back(cropped);
        n = min(n+1, 3);
        makeSomeStuff();
        isCropped = false;
        showInputImages();
    }
}

void printUsage() {
    cout << "Usage [-p] [-v input_video] [-i input_image] [-d detector_code] " 
         << "[-e extractor_code]" << endl;
}

int main( int argc, char** argv ) {

    // options
    bool preprocess = false;
    string input = "";
    vector<string> images;
    string crop_image;

    // default detector and extractor codes
    detector_code = SURF_D;
    extractor_code = SURF_E;

    for (int i = 1; i < argc; i++) {
        if (ctos(argv[i]) == "-p") preprocess = true;
        if (ctos(argv[i]) == "-v") input = ctos(argv[i+1]);
        if (ctos(argv[i]) == "-c") crop_image = ctos(argv[i+1]);
        if (ctos(argv[i]) == "-i") images.push_back(ctos(argv[i+1]));
        if (ctos(argv[i]) == "-d") 
            if (getCode(argv[i+1]) > 0)
                detector_code = getCode(argv[i+1]);
        if (ctos(argv[i]) == "-e") 
            if (getCode(argv[i+1]) > 0)
                extractor_code = getCode(argv[i+1]);
    }
    n = images.size();

    // get the right matcher
    if (extractor_code == ORB_E || extractor_code == BRIEF_E) matcher_code = BFHAM_M;
    else matcher_code = FLANN_M;

    namedWindow("Movie", 1);
    setMouseCallback("Movie", CallBackFunc, NULL);

    if (crop_image.length() > 0) 
        cout << "Click and drag over desired objects. Press any key to continue\n";

    while(crop_image.length() > 0) {
        Mat crop_i = imread(crop_image);
        img = orig_img = crop_i;
        drawImage();
        if (waitKey(30) >= 0) break;
        processCrops();
    }

    if (n == 0 && crop_image.length() == 0) { printUsage(); return 0; }

    for (int i = 0; i < images.size(); i++) img_objects.push_back(imread(images[i]));
    makeSomeStuff();
    showInputImages();

    VideoCapture cap;
    if (input.length() == 0) {
        cap.open(0);
    } else {
        cap.open(input);
    }

    if(!cap.isOpened()) return -1;


    clock_t startTime = clock();
    Mat img_scene;
    vector<Mat> savedMats;
    int f = 0;

    // implement an option to specify the ability to crop your own 
    // object images based on the first image.
    for(;;) {
        f++;
        processCrops();

        if (f % 10 == 0) {
            double g = double(clock() - startTime)/ (double) CLOCKS_PER_SEC;
            printf("frame: %4d | seconds: %lf | fpd %lf\n", f, g, f*1.0/g);
        }
        vector<vector<DMatch> > good_matches(img_objects.size());

        if (!drawing) {
            cap >> img_scene;
            if (img_scene.empty()) {
                cout << "finished reading\n";
                break;
            }
            // shrink if webcam
            if (input.length() == 0) resize(img_scene, img_scene, Size(0,0), 0.5, 0.5);   
    
            
            orig_img = img_scene;
            Mat m = process_images(img_objects, keypoints_objects, descriptors_objects,
                                   img_scene, good_matches);
            // offset is found here
            savedMats.push_back(m);
            img = m;
        }
        
        if (!preprocess) drawImage();

        int key = waitKey(1);
        if (key >= '1' && key <= '3') {
            chosen = true; 
            focus = key - '1';
            cout << "Focus changed to: " << (key - '1') << endl;
        } else if (key == '0') {
            chosen = false;
            cout << "Focus changed to best match\n";
        } else if (key >= 0) break;
    }

    if (preprocess) {
        cout << "Preprocessing done. Press any key to continue\n";
        waitKey(0);
        namedWindow("Processed Movie", 1);
        for (int i = 0; i < savedMats.size(); i++) {
            imshow("Processed Movie", savedMats[i]);
            if (waitKey(30) >= 0) break;
        }
    }


    double g = double(clock() - startTime)/ (double) CLOCKS_PER_SEC;
    cout << "frame: " << f << " | seconds: " 
         << g << " | fps: " << f*1.0/g << endl;

    return 0;
}