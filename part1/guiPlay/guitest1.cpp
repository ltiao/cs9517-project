#include "opencv2/highgui/highgui.hpp"
#include <iostream>
 
using namespace std;
using namespace cv;

Mat img;
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
            Mat disp = img(Rect(sx, sy, x-sx, y-sy));
            imshow("Crop", disp);
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
 
int main() {
    VideoCapture cap("./clip_test.m4v");
    if(!cap.isOpened())  // check if we succeeded
        return -1;
 
    namedWindow("Movie",1);
    namedWindow("Crop",1);
    setMouseCallback("Movie", CallBackFunc, NULL);
    for(;;) {
        cap >> img; // get a new frame from camera
        drawImage();   
        if(waitKey(30) >= 0) break;
    }
 
    return 0;
}