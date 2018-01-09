// Eli Lilly code-single image No MIP

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
//#include<conio.h>
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>	



using namespace cv;
using namespace std;

ofstream myfile;
RNG rng;
vector <Mat> enhance;

const int NeuralTHRESH = 1400; //threshold is the amount of green pixels in the bounding rectangle around blucontours
const float tune = 0.1;
const int houghthresh = 275;
string imname;

//Function for writing Image
void writeImage(string name,Mat img)
{
string filename=format("%s_%s.tif",imname.c_str(),name.c_str());
imwrite(filename,img);
}

// changes contrast and brightness of each z slice for each channel

Mat changeimg(Mat image, float alpha, float beta)
{
	alpha = alpha / 10;
	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	Mat new_image = Mat::zeros(image.size(), CV_16UC3);


	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				//cout << new_image.at<Vec3s>(y,x)[c] << endl;

				new_image.at<Vec3s>(y, x)[c] = (alpha*(image.at<Vec3s>(y, x)[c]) + beta);
			}
		}
	}
  
	normalize(new_image, new_image, 0, 65535, NORM_MINMAX);
	return new_image;
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
float vari(int a, int b, int c, int d)
{
	float mean = (a + b + c + d) / 4;
	float var = (((a - mean)*(a - mean)) + ((b - mean)*(b - mean)) + ((c - mean)*(c - mean)) + ((d - mean)*(d - mean))) / 4;
	return var;
}

//----------------------Finds count of L, M, H intensity synapses around // LMH syn or dendrites --------------------------------
void neighboursyn(Mat redlow, Mat redmed, Mat redhigh, vector<int> CoordinatesX, vector<int> CoordinatesY, ofstream & myfile, int w, int h)
{
	unsigned int totl = 0, totm = 0, toth = 0;
	double totlvar = 0, totmvar = 0, tothvar = 0;
	int lcount1; int mcount1; int hcount1; int lcount2; int mcount2; int hcount2; int lcount3; int mcount3; int hcount3; int lcount4; int mcount4; int hcount4;

	for (int i = 0; i < CoordinatesX.size(); i++)
	{

		Point a = Point(CoordinatesX[i], CoordinatesY[i]);
		lcount1 = 0; mcount1 = 0; hcount1 = 0; lcount2 = 0; mcount2 = 0; hcount2 = 0; lcount3 = 0; mcount3 = 0; hcount3 = 0; lcount4 = 0; mcount4 = 0; hcount4 = 0;


		if (((a.x) + w  < redlow.rows) && ((a.x) - w > 0) && ((a.y) + h < redlow.cols) && ((a.y) - h> 0))

		{
			CvRect myrect = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrect.x >= 0 && myrect.y >= 0 && myrect.width + myrect.x < redlow.cols && myrect.height + myrect.y < redlow.rows)
			{
				int px = (a.x + (0.5*w)); int py = (a.y + (h / 2));
				//if (px >= 0 && py >= 0 && myrect.width + px < redlow.cols && myrect.height + py < redlow.rows)
				Mat lroi = redlow(myrect);// creating a new image from roi of redlow
				cv::Rect top_left(cv::Point(0, 0), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Rect top_right(cv::Point(0, lroi.size().width / 2), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Rect bottom_left(cv::Point(lroi.size().height / 2, 0), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Rect bottom_right(cv::Point(lroi.size().height / 2, lroi.size().width / 2), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Mat Image1, Image2, Image3, Image4;
				Image1 = lroi(top_right);
				Image2 = lroi(top_left);
				Image3 = lroi(bottom_right);
				Image4 = lroi(bottom_left);

				lcount1 = countNonZero(Image1);
				lcount2 = countNonZero(Image2);
				lcount3 = countNonZero(Image3);
				lcount4 = countNonZero(Image4);
				float lvar = vari(lcount1, lcount2, lcount3, lcount4);
				totlvar = totlvar + lvar;
			}

		}
		if (((a.x) + w + 20 < redmed.rows) && ((a.x) - w - 20 > 0) && ((a.y) + h + 20 < redmed.cols) && ((a.y) - h - 20> 0))

		{
			CvRect myrectM = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectM.x >= 0 && myrectM.y >= 0 && myrectM.width + myrectM.x < redmed.cols && myrectM.height + myrectM.y < redmed.rows)
			{
				Mat mroi = redmed(myrectM);// creating a new image from roi of redmed

				cv::Rect top_left(cv::Point(0, 0), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Rect top_right(cv::Point(0, mroi.size().width / 2), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Rect bottom_left(cv::Point(mroi.size().height / 2, 0), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Rect bottom_right(cv::Point(mroi.size().height / 2, mroi.size().width / 2), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Mat Image1, Image2, Image3, Image4;
				Image1 = mroi(top_right);
				Image2 = mroi(top_left);
				Image3 = mroi(bottom_right);
				Image4 = mroi(bottom_left);
				mcount1 = countNonZero(Image1);
				mcount2 = countNonZero(Image2);
				mcount3 = countNonZero(Image3);
				mcount4 = countNonZero(Image4);
				float mvar = vari(mcount1, mcount2, mcount3, mcount4);
				totmvar = totmvar + mvar;
			}
		}


		if (((a.x) + w + 20 < redhigh.rows) && ((a.x) - w - 20 > 0) && ((a.y) + h + 20 < redhigh.cols) && ((a.y) - h - 20> 0))

		{
			CvRect myrectH = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectH.x >= 0 && myrectH.y >= 0 && myrectH.width + myrectH.x < redhigh.cols && myrectH.height + myrectH.y < redhigh.rows)
			{
				Mat hroi = redhigh(myrectH);// creating a new image from roi of redmed


				cv::Rect top_left(cv::Point(0, 0), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Rect top_right(cv::Point(0, hroi.size().width / 2), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Rect bottom_left(cv::Point(hroi.size().height / 2, 0), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Rect bottom_right(cv::Point(hroi.size().height / 2, hroi.size().width / 2), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Mat Image1, Image2, Image3, Image4;
				Image1 = hroi(top_right);
				Image2 = hroi(top_left);
				Image3 = hroi(bottom_right);
				Image4 = hroi(bottom_left);
				hcount1 = countNonZero(Image1);
				hcount2 = countNonZero(Image2);
				hcount3 = countNonZero(Image3);
				hcount4 = countNonZero(Image4);
				float hvar = vari(hcount1, hcount2, hcount3, hcount4);
				tothvar = tothvar + hvar;

			}
		}


	}
	if (CoordinatesX.size()>0)
		myfile << totlvar / CoordinatesX.size() << "," << totmvar / CoordinatesX.size() << "," << tothvar / CoordinatesX.size() << ",";
	else
		myfile << 0 << "," << 0 << "," << 0 << ",";
}
//calculates avg count of Low, Medium and High int synapses around L/M/H intensity synapse points
void aroundsyncalc(Mat redlow, Mat redmed, Mat redhigh, vector<int> CoordinatesX, vector<int> CoordinatesY, ofstream & myfile)
{
	unsigned int totl = 0, totm = 0, toth = 0; int lcount; Mat dst; int mcount; int hcount; int w = 40, h = 40;

	for (int i = 0; i < CoordinatesX.size(); i++)
	{

		Point a = Point(CoordinatesX[i], CoordinatesY[i]);
		lcount = 0; mcount = 0; hcount = 0;

		if (((a.x) + w + 4 < redlow.rows) && ((a.x) - w > 0) && ((a.y) + h + 4 < redlow.cols) && ((a.y) - h > 0))

		{
			CvRect myrect = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrect.x >= 0 && myrect.y >= 0 && myrect.width + myrect.x < redlow.cols && myrect.height + myrect.y < redlow.rows)
			{
				Mat lroi = redlow(myrect);// creating a new image from roi of redmed
				lcount = countNonZero(lroi);
				totl = totl + lcount;
				//cout <<"totl:  "<< totl<< endl;

			}
		}


		if (((a.x) + w + 4 < redmed.rows) && ((a.x) - w > 0) && ((a.y) + h + 4 < redmed.cols) && ((a.y) - h > 0))

		{
			CvRect myrectM = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectM.x >= 0 && myrectM.y >= 0 && myrectM.width + myrectM.x < redmed.cols && myrectM.height + myrectM.y < redmed.rows)
			{
				Mat mroi = redmed(myrectM);// creating a new image from roi of redmed
				mcount = countNonZero(mroi);
				totm = totm + mcount;
			}
		}


		if (((a.x) + w + 4 < redhigh.rows) && ((a.x) - w > 0) && ((a.y) + h + 4 < redhigh.cols) && ((a.y) - h > 0))

		{
			CvRect myrectH = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectH.x >= 0 && myrectH.y >= 0 && myrectH.width + myrectH.x < redhigh.cols && myrectH.height + myrectH.y < redhigh.rows)
			{
				Mat hroi = redhigh(myrectH);// creating a new image from roi of redmed
				hcount = countNonZero(hroi);
				toth = toth + hcount;
			}
		}


	}

	// avg count of Low, Medium and High int synapses around L/M/H intensity synapse points


	if (CoordinatesX.size() != 0)
		myfile <<totl / CoordinatesX.size() << "," << totm / CoordinatesX.size() << "," << toth / CoordinatesX.size() << ",";
	else
		myfile << 0 << "," << 0 << "," << 0 << ",";
}

//-- - [7]----
void redinf(string imname, Mat im, ofstream &myfile)
{
	Mat enhancedl; vector< int> CoordinateLX; vector< int> CoordinateLY; vector< int> CoordinateMX; vector< int> CoordinateMY;
	vector< int> CoordinateHX; vector< int> CoordinateHY;

	// Enhance the red low channel
	cv::inRange(im, cv::Scalar(0, 0, 5000), cv::Scalar(10, 10, 8000), enhancedl);
	erode(enhancedl, enhancedl, Mat());
	Mat redlow = enhancedl.clone();
	Mat nonZeroCoordinatesL;
	findNonZero(enhancedl, nonZeroCoordinatesL);

	Mat enhancedm;
	// Enhance the red medium channel
	cv::inRange(im, cv::Scalar(0, 0, 8000), cv::Scalar(10, 10, 30000), enhancedm);
	erode(enhancedm, enhancedm, Mat());
	Mat redmed = enhancedm.clone();
	Mat nonZeroCoordinatesM;
	findNonZero(enhancedm, nonZeroCoordinatesM);

	Mat enhancedh;
	// Enhance the red high channel
	cv::inRange(im, cv::Scalar(0, 0, 30000), cv::Scalar(10, 10, 65535), enhancedh);
	erode(enhancedm, enhancedm, Mat());
	Mat redhigh = enhancedh.clone();
	Mat nonZeroCoordinatesH;
	findNonZero(enhancedh, nonZeroCoordinatesH);


	for (int i = 0; i < nonZeroCoordinatesL.total(); i++)
	{
		CoordinateLX.push_back(((nonZeroCoordinatesL.at<Point>(i).x)));
		CoordinateLY.push_back(((nonZeroCoordinatesL.at<Point>(i).y)));
	}
	for (int i = 0; i < nonZeroCoordinatesM.total(); i++)
	{
		CoordinateMX.push_back(((nonZeroCoordinatesM.at<Point>(i).x)));
		CoordinateMY.push_back(((nonZeroCoordinatesM.at<Point>(i).y)));
	}
	for (int i = 0; i < nonZeroCoordinatesH.total(); i++)
	{
		CoordinateHX.push_back(((nonZeroCoordinatesH.at<Point>(i).x)));
		CoordinateHY.push_back(((nonZeroCoordinatesH.at<Point>(i).y)));
	}

	// find avg count of redlow, redmed, redhigh synapse around Low, M, H int synpse
	myfile<<",";
	aroundsyncalc(redlow, redmed, redhigh, CoordinateLX, CoordinateLY, myfile);// find avg count of redlow, redmed, redhigh synapse around Low int synpse
	aroundsyncalc(redlow, redmed, redhigh, CoordinateMX, CoordinateMY, myfile);
	aroundsyncalc(redlow, redmed, redhigh, CoordinateHX, CoordinateHY, myfile);
	neighboursyn(redlow, redmed, redhigh, CoordinateLX, CoordinateLY, myfile, 40, 40);
	neighboursyn(redlow, redmed, redhigh, CoordinateMX, CoordinateMY, myfile, 40, 40);
	neighboursyn(redlow, redmed, redhigh, CoordinateHX, CoordinateHY, myfile, 40, 40);
	neighboursyn(redlow, redmed, redhigh, CoordinateLX, CoordinateLY, myfile, 80, 80);
	neighboursyn(redlow, redmed, redhigh, CoordinateMX, CoordinateMY, myfile, 80, 80);
	neighboursyn(redlow, redmed, redhigh, CoordinateHX, CoordinateHY, myfile, 80, 80);
}

//---[5]-------------
void reddetect(Mat im, Mat & redlow, Mat & redmed, Mat & redhigh,string imname, int i)
{

	Mat enhancedl;
	// Enhance the red low channel
	cv::inRange(im, cv::Scalar(0, 0, 5000), cv::Scalar(10, 10, 8000), enhancedl);
	erode(enhancedl, enhancedl, Mat());
	redlow = enhancedl.clone();
	string imredl=format("z%d_redlow",i);
	writeImage(imredl, redlow);

	Mat enhancedm;
	// Enhance the red medium channel
	cv::inRange(im, cv::Scalar(0, 0, 8000), cv::Scalar(10, 10, 30000), enhancedm);
	erode(enhancedm, enhancedm, Mat());
	erode(enhancedm, enhancedm, Mat());
	dilate(enhancedm, enhancedm, Mat());
	redmed = enhancedm.clone();
	string imredm=format("z%d_redmed",i);
	writeImage(imredm, redmed);

	Mat enhancedh;
	// Enhance the red high channel
	cv::inRange(im, cv::Scalar(0, 0, 30000), cv::Scalar(10, 10, 65535), enhancedh);
	erode(enhancedh, enhancedh, Mat());
	redhigh = enhancedh.clone();
	string imredh=format("z%d_redhigh",i);
	writeImage(imredh, redhigh);


}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void redcount(vector<vector<Point> > blucontoursr, int j)
{
	int countlrsum = 0, countrmsum = 0, countrhsum = 0;
	int grim = j + 1;
	//  draw bounding rectangle around blue contours
	// Check for presence of reds around the blue contours
	vector<Rect> brect(blucontoursr.size());
	Mat rl, rm, rh;
	for (int io = 0; io < blucontoursr.size(); io++)
	{
		brect[io] = boundingRect(Mat(blucontoursr[io]));

		Mat image = enhance[grim];
		Mat image_roi = image(brect[io]);// creating a new image from roi
		cv::inRange(image_roi, cv::Scalar(0, 0, 5000), cv::Scalar(10, 10, 8000), rl);// channel red low
		erode(rl, rl, Mat());
		int countlr = countNonZero(rl);
		countlrsum = countlr + countlrsum;

		cv::inRange(image_roi, cv::Scalar(0, 0, 8000), cv::Scalar(10, 10, 30000), rm);// channel red medium
		erode(rm, rm, Mat());
		int countrm = countNonZero(rm);
		countrmsum = countrm + countrmsum;

		cv::inRange(image_roi, cv::Scalar(0, 0, 30000), cv::Scalar(10, 10, 65535), rh);// channel red high
		erode(rh, rh, Mat());
		int countrh = countNonZero(rh);
		countrhsum = countrh + countrhsum;



	}


	countlrsum = countlrsum / blucontoursr.size();
	countrmsum = countrmsum / blucontoursr.size();
	countrhsum = countrhsum / blucontoursr.size();
}



void dendritecalc(string imname, Mat  redlow, Mat redmed, Mat redhigh, Mat highint, Mat all, int highintno, int lowintno, ofstream & myfile)
{
	Mat highredlo(highint.size(), CV_8UC1, Scalar(0, 0, 0));
	Mat highredmed(highint.size(), CV_8UC1, Scalar(0, 0, 0)); Mat highredhi(highint.size(), CV_8UC1, Scalar(0, 0, 0));
	Mat lowredlo(highint.size(), CV_8UC1, Scalar(0, 0, 0));; Mat lowredmed(highint.size(), CV_8UC1, Scalar(0, 0, 0)); Mat lowredhi(highint.size(), CV_8UC1, Scalar(0, 0, 0));
	float avgLH, avgMH, avgHH, avgLL, avgML, avgHL;
	Mat lowint = all - highint;
	bitwise_and(highint, redlow, highredlo);
	bitwise_and(highint, redmed, highredmed);
	bitwise_and(highint, redhigh, highredhi);
	bitwise_and(lowint, redlow, lowredlo);
	bitwise_and(lowint, redmed, lowredmed);
	bitwise_and(lowint, redhigh, lowredhi);



	if (highintno == 0)
	{
		avgLH = 0; avgMH = 0; avgHH = 0;
	}

	else
	{
		avgLH = countNonZero(highredlo) / highintno; // low intensity synapses on high intensity dendrites
		avgMH = countNonZero(highredmed) / highintno;// med intensity synapses on high intensity dendrites
		avgHH = countNonZero(highredhi) / highintno;// high intensity synapses on high intensity dendrites
	}
	if (lowintno == 0)
	{
		avgLL = 0; avgML = 0; avgHL = 0;
	}
	else
	{
		avgLL = countNonZero(lowredlo) / lowintno; // low intensity synapses on low intensity dendrites
		avgML = countNonZero(lowredmed) / lowintno; // med intensity synapses on low intensity dendrites
		avgHL = countNonZero(lowredhi) / lowintno; // high intensity synapses on low intensity dendrites
	}


	//myfile << "dendritecalc" << "," << "Image name" << "," << "Average low int synpases arnd high width dendrites" << "," << "Average med int synpases arnd high width dendrites" << "," << "Average high int synpases arnd high width dendrites" << ", " << "Average low int synpases arnd small width dendrites" << "," << "Average med int synpases arnd small width dendrites" << "," << "Average high int synpases arnd small width dendrites" << ",";
	myfile  << avgLH << "," << avgMH << "," << avgHH << ", " << avgLL << "," << avgML << "," << avgHL << ",";

}




//----[6]----------detects dedrite,classifies as dendrite/axon, calc metrics-
Mat createmaskimage(Mat image, Mat dXX, Mat dYY, Mat dXY)
{
	Mat maskimg(image.rows, image.cols, CV_8U);
	maskimg = cv::Scalar(0, 0, 0);
	cv::Mat hessian_matrix(2, 2, CV_32F);
	Mat eigenvec = Mat::ones(2, 2, CV_32F);
	std::vector<float> eigenvalues(2);

	//----------Inside image

	for (int i = 1; i < image.rows; i++) {
		for (int j = 1; j < image.cols; j++) {
			hessian_matrix.at<float>(Point(0, 0)) = dXX.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 1)) = dYY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(0, 1)) = dXY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 0)) = dXY.at<float>(Point(j, i));

			// find eigen values of hessian matrix /* Larger  eigenvalue  show the  direction  of  intensity change, while  smallest  eigenvalue  show  the  direction  of  vein.*/
			eigen(hessian_matrix, eigenvalues, eigenvec);

			/*Main Condition*/if (abs(eigenvalues[0])<5 && abs(eigenvalues[1])>6)

			{
				/*Condition 1*/	if ((abs(eigenvec.at<float>(Point(0, 0))) > 0) && (abs(eigenvec.at<float>(Point(0, 0)) < 0.4)))
				{
					circle(maskimg, cv::Point(j, i), 1, cv::Scalar(255, 255, 255), 2, 8, 0);//orange
				}
				/*Condition 2*/	if ((abs(eigenvec.at<float>(Point(0, 0))) > 0.5) && (abs(eigenvec.at<float>(Point(0, 0)) < 1.2)))
				{
					circle(maskimg, cv::Point(j, i), 1, cv::Scalar(255, 255, 255), 2, 8, 0);//blue
				}
			}
		}
	}
	//imwrite("maskimg.png", maskimg);
	return maskimg;
}

void filterHessian(string imname, int z,Mat image, Mat redlow, Mat redmed, Mat redhigh, ofstream &myfile)
{
	int co = 0;
	imwrite("remove.png", image); image = imread("remove.png");// to covert the 16 bit image to 8 bit
	Mat org = image.clone();
	Mat orgclone = org.clone();
	Mat outputDend(org.size(),CV_8UC4);
	outputDend=cv::Scalar(255,255,255,0);
	cvtColor(image, image, CV_BGR2GRAY);
	Mat checkimg(image.rows, image.cols, CV_8U);
	Mat overlapimage(image.rows, image.cols, CV_16U);
	Mat dendritetips(image.rows, image.cols, CV_8U);
	Mat overlapbinimage(image.rows, image.cols, CV_16U);
	cv::Mat dXX, dYY, dXY;
	std::vector<float> eigenvalues(2);
	cv::Mat hessian_matrix(2, 2, CV_32F);
	Mat eigenvec = Mat::ones(2, 2, CV_32F);
	//std::vector<float> eigenvec(2,2); //Mat eigenvec, eigenvalues;

	//calculte derivatives
	cv::Sobel(image, dXX, CV_32F, 2, 0);
	cv::Sobel(image, dYY, CV_32F, 0, 2);
	cv::Sobel(image, dXY, CV_32F, 1, 1);
int gauKsize=3;
	//apply gaussian filtering to the image
	cv::Mat gau = cv::getGaussianKernel(gauKsize, -1, CV_32F);
	cv::sepFilter2D(dXX, dXX, CV_32F, gau.t(), gau);
	cv::sepFilter2D(dYY, dYY, CV_32F, gau.t(), gau);
	cv::sepFilter2D(dXY, dXY, CV_32F, gau.t(), gau);

	Mat maskimage = createmaskimage(image, dXX, dYY, dXY);// creates thresholded image of all the possible dendrites
	//create high intensity thresholded image to bin dendrites into developed and less developed dendrites
	Mat highIntgreenthreshimg(image.rows, image.cols, CV_16U);
	cv::inRange(org, cv::Scalar(0, 150, 0), cv::Scalar(100, 255, 100), highIntgreenthreshimg);//
	dilate(highIntgreenthreshimg, highIntgreenthreshimg, Mat());
	erode(highIntgreenthreshimg, highIntgreenthreshimg, Mat());

	//----------Inside image
	int countofdendrites = 0;
	int developed = 0;
	int lessdeveloped = 0;
	for (int i = 1; i < image.rows; i++) {
		for (int j = 1; j < image.cols; j++) {
			hessian_matrix.at<float>(Point(0, 0)) = dXX.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 1)) = dYY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(0, 1)) = dXY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 0)) = dXY.at<float>(Point(j, i));

			// find eigen values of hessian matrix /* Larger  eigenvalue  show the  direction  of  intensity change, while  smallest  eigenvalue  show  the  direction  of  vein.*/
			eigen(hessian_matrix, eigenvalues, eigenvec);
			//find all sets of dendrites (horizontal an vertical)
			/*Main Condition*/if (abs(eigenvalues[0])<5 && abs(eigenvalues[1])>6)
			{
				checkimg = cv::Scalar(0, 0, 0);
				overlapimage = cv::Scalar(0, 0, 0);
				dendritetips = cv::Scalar(0, 0, 0);
				overlapbinimage = cv::Scalar(0, 0, 0);

				/*Condition 1*/	if ((abs(eigenvec.at<float>(Point(0, 0))) > 0) && (abs(eigenvec.at<float>(Point(0, 0)) < 0.4)))	// for vertical dendrites
				{
					circle(checkimg, cv::Point(j, i), 1, cv::Scalar(255, 255, 255), 3, 8, 0);//orange
					circle(outputDend, cv::Point(j, i), 1, cv::Scalar(0, 128, 255,255), 2, 8, 0);//orange
				}
				/*Condition 2*/	else if ((abs(eigenvec.at<float>(Point(0, 0))) > 0.6) && (abs(eigenvec.at<float>(Point(0, 0)) < 1.2))) // for horizontal dendrites
				{
					circle(checkimg, cv::Point(j, i), 1, cv::Scalar(255, 255, 255), 3, 8, 0);//blue
					circle(outputDend, cv::Point(j, i), 1, cv::Scalar(0, 128, 255,255), 2, 8, 0);//orange
				}
				else{}

				bitwise_and(checkimg, maskimage, overlapimage);// to detct region of overlap inorder to find dendrite tips/start points 

				// classifies dendrites as developd and under dveloped based on overlap of dendrite tips with high intensity green images
				if (countNonZero(overlapimage)>25)
				{
					countofdendrites++;
					circle(org, cv::Point(j, i), 1, cv::Scalar(255, 125, 0), 3, 8, 0);//blue;
					circle(dendritetips, cv::Point(j, i), 1, cv::Scalar(255, 255, 255), 3, 8, 0);//blue;
					bitwise_and(dendritetips, highIntgreenthreshimg, overlapbinimage);
					if (countNonZero(overlapbinimage) > 5)
					{
						developed++;
						circle(org, cv::Point(j, i), 1, cv::Scalar(255, 255, 0), 3, 8, 0);//blue;
					}
					else
						lessdeveloped++;
				}
			}
		}
	}
	string name = format("z%d_dend",z);
	writeImage(name, outputDend);
	myfile<< countofdendrites << " ," << developed << ", " << lessdeveloped << ",";
	dendritecalc(imname, redlow, redmed, redhigh, highIntgreenthreshimg, maskimage, developed, lessdeveloped, myfile);
	

}


void calcCellMetrics(string imname, int i, vector < vector<Point> > blucontours, ofstream &myfile)
{
	// average area of blue contours
	float bltotarea = 0, blavgarea; int lowcount = 0, medcount = 0, hihcount = 0;
	for (int m = 0; m < blucontours.size(); m++)
	{
		float chaid = contourArea(blucontours[m], false);
		if (chaid <= 1000)
			lowcount = lowcount + 1;
		if (chaid > 1000 && chaid <= 3500)
			medcount = medcount + 1;
		if (chaid > 3500)
			hihcount = hihcount + 1;

		bltotarea = bltotarea + chaid;
	}
	if (blucontours.size() != 0)
		blavgarea = bltotarea / blucontours.size();
	else
		blavgarea = 0;

	// average aspect ratio of blue contours
	float bltotasp = 0, blavgasp;
	for (int m = 0; m < blucontours.size(); m++)
	{
		vector<Rect> brect(blucontours.size());
		float asprat;
		brect[m] = boundingRect(Mat(blucontours[m]));// bpunding rect
		asprat = brect[m].height / brect[m].width;// aspect ratio
		bltotasp = bltotasp + asprat;
	}
	if (blucontours.size() != 0)
		blavgasp = bltotasp / blucontours.size();
	else
		blavgasp = 0;

	// average diameter of blue contours
	float bltotdia = 0, blavgdia;
	vector<Point2f>center(blucontours.size());
	vector<float>radius(blucontours.size());
	for (int m = 0; m < blucontours.size(); m++)
	{
		minEnclosingCircle((Mat)blucontours[m], center[m], radius[m]);
		bltotdia = bltotdia + (2 * radius[m]);
	}
	if (blucontours.size() != 0)
		blavgdia = bltotdia / blucontours.size();
	else
		blavgdia = 0;
		
	myfile << blucontours.size() << "," << hihcount << "," << medcount << "," << lowcount << "," << blavgarea << "," << blavgasp << "," << blavgdia;


}


// -----[3]------cell metrics for every plane in an image (5 z-planes)
void cellmetrics(string imname, int i, vector < vector<Point> > blucontours, vector < vector<Point> > nucontours, vector <vector<Point> > astcontours, ofstream &myfile)
{
	calcCellMetrics(imname, i, blucontours, myfile);
	myfile << ",";
	calcCellMetrics(imname, i, nucontours, myfile);
	myfile <<",";
	calcCellMetrics(imname, i, astcontours, myfile);


}
//----[4]----- finds the number of low/med/high synapses near an astrocyte and neural cell
void synapcalc(string imname, int i, vector <Mat> stackim, vector<vector<Point> > nucontours)
{
	Mat redres;
	Mat img = stackim[i];
	string imm=format("%s_z%d", imname.c_str(), i);
	//cv::Mat normalized;
	vector <Mat> redlow, redmed, redhigh;
	//cv::normalize(img, normalized, 0, 255, cv::NORM_MINMAX, CV_16UC1);


	// counting the number of red  intensity pixels around a nuclei cell
	int totrl = 0, avglo;
	if (i == 1 || (i % 3 == 1))// channel red low
	{
		// Enhance the red low channel
		cv::inRange(img, cv::Scalar(0, 0, 5000), cv::Scalar(10, 10, 8000), redres);
		erode(redres, redres, Mat());
		vector<Rect> brect(nucontours.size());
		for (int io = 0; io < nucontours.size(); io++)
		{
			brect[io] = boundingRect(Mat(nucontours[io]));

			Mat image = redres;
			Mat image_roi = redres(brect[io]);// creating a new image from roi
			int count = countNonZero(image_roi);
			totrl = totrl + count;
		}
		if (nucontours.size() != 0)
			avglo = totrl / nucontours.size();
		else
			avglo = 0;
	}

	// counting the average number of red medium intensity pixels around a nuclei cell
	int totrm = 0, avgme;
	if (i == 1 || (i % 3 == 1))// channel red medium
	{
		// Enhance the red medium channel
		cv::inRange(img, cv::Scalar(0, 0, 8000), cv::Scalar(10, 10, 30000), redres);
		erode(redres, redres, Mat());

		vector<Rect> bmrect(nucontours.size());
		for (int io = 0; io < nucontours.size(); io++)
		{
			bmrect[io] = boundingRect(Mat(nucontours[io]));

			Mat image = redres;
			Mat image_roi = redres(bmrect[io]);// creating a new image from roi
			int countm = countNonZero(image_roi);
			totrm = totrm + countm;
		}
		if (nucontours.size() != 0)
			avgme = totrm / nucontours.size();
		else
			avgme = 0;

	}

	// counting the average number of red high intensity pixels around a nuclei cell
	int totrh = 0, avglh;
	if (i == 1 || (i % 3 == 1))// channel red high
	{
		// Enhance the red high channel
		cv::inRange(img, cv::Scalar(0, 0, 30000), cv::Scalar(10, 10, 65535), redres);
		erode(redres, redres, Mat());
		vector<Rect> bhrect(nucontours.size());
		for (int io = 0; io < nucontours.size(); io++)
		{
			bhrect[io] = boundingRect(Mat(nucontours[io]));

			Mat image = redres;
			Mat image_roi = redres(bhrect[io]);// creating a new image from roi
			int counth = countNonZero(image_roi);
			totrh = totrh + counth;
		}
		if (nucontours.size() != 0)
			avglh = totrh / nucontours.size();
		else
			avglh = 0;

	}
	// total no of synapses around, avg low intensity around , avg medium intensity synapses around, avg high intensity synapses around
	myfile << avglo + avgme + avglh << "," << avglo << "," << avgme << "," << avglh;

}
// thresholds all three channels for all z planes of one image and writes it

//----------[1]
void enhanceImage(string imname, vector <Mat> stackim, vector <Mat> &enhance, int i)
{

		Mat bimg = stackim[i]; 
		Mat gimg = stackim[i+2]; 
		Mat rimg = stackim[i+1]; 
		Mat gry;
		vector <Mat> redlow, redmed, redhigh;

		//-----------------------------------------------------------------------------------------------------------------------------------
		// Enhance the image using Gaussian blur and thresholding
		cv::Mat benhanced;
		
			// Enhance the blue channel
			cv::inRange(bimg, cv::Scalar(13000, 0,0), cv::Scalar(50000, 50, 20), benhanced);// blue threshold
			bitwise_not(benhanced, benhanced);
			enhance.push_back(benhanced);
			//string bname = format("enh_%d.tif", i);//temporary file: no need to write
			//imwrite(bname, benhanced);
		
		cv::Mat renhanced;
			// Enhance the red channel
			cv::inRange(rimg, cv::Scalar(0, 0, 7000), cv::Scalar(10, 10, 65535), renhanced);
			enhance.push_back(renhanced);
			//string rname = format("enh_%d.tif", (i+1));//temporary file: no need to write
			//imwrite(rname, renhanced);
		

	cv::Mat genhanced;
			// Enhance the green channel
			cv::inRange(gimg, cv::Scalar(0, 100, 0), cv::Scalar(10, 9500, 10), genhanced);
			bitwise_not(genhanced, genhanced);
			enhance.push_back(genhanced);
			//string gname = format("enh_%d.tif", (i+2));//temporary file: no need to write
			//imwrite(gname, genhanced);

}
// displays combined image with blue thresholds
void drawthreshold(vector <Mat> stackim, int i, string imname)
{
	Mat added;
	addWeighted(stackim[i], 0.5, stackim[i + 1], 0.5, 0, added);
	addWeighted(added, 0.5, stackim[i + 2], 0.5, 0, added);
	added = 5 * added;
	string name = format("z%d_mod",  i);
	writeImage(name, added);
}
//----------[2]
// counts number of astrocytes, cells and nueral cells

void cellcount(string imname, vector <Mat> stackim, vector <Mat> enhance, int i)
{
	//myfile << endl;
	vector<vector<Point> > contours; // Vector for storing contour
	vector<Vec4i> hierarchy;
	vector<vector<Point> > blucontours;
	vector<vector<Point> > nucontours;
	vector<vector<Point> > astcontours;
	vector<vector<Point> > othcontours;
	int tau;
	Mat drawimg;
	// draws and saves blue contours
		//string imm = format("%s_z%d", imname, i);
	string imm = format("%s_z%d", imname.c_str(), i);
	blucontours.clear();
	astcontours.clear();
	nucontours.clear();
		if (i == 0 || (i % 3 == 0))// channel blue
		{
			findContours(enhance[i], contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // find all blue contours
			blucontours.clear();
			// FINDING CELLS: and saving the contours that are above a certain area
			for (int d = 0; d < contours.size(); d++)
			{
				float chaid = contourArea(contours[d], false);
				if (chaid > 200)
					blucontours.push_back(contours[d]);
			}
			tau = i + 2; // decides image index for green channel

			// FIND NUERAL CELLS: draw bounding rectangle around blue contours
			// Check for presence of dendrites(green) around the blue contours
			vector<Rect> brect(blucontours.size());
			vector<Rect> arect(blucontours.size());
			vector< float> asprat(blucontours.size());

			vector<vector<Point> > hull(blucontours.size());
			vector<vector<int> > hullsI(blucontours.size()); // Indices to contour points
			vector<vector<Vec4i> > defects(blucontours.size());
			for (int io = 0; io < blucontours.size(); io++)
			{
				brect[io] = boundingRect(Mat(blucontours[io]));
				float chaid = contourArea(blucontours[io], false);
				Mat image = enhance[tau];
				Mat imageR = enhance[tau - 1];
				Mat image_roi = image(brect[io]);// creating a new image from roi
				Mat image_red = imageR(brect[io]);// creating a new image from roi
				int count = countNonZero(image_roi);
				int countR = countNonZero(image_red);
			
				convexHull(blucontours[io], hull[io], false);
				convexHull(blucontours[io], hullsI[io], false);

					if (hullsI[io].size() > 3) // You need more than 3 indices          
						convexityDefects(blucontours[io], hullsI[io], defects[io]);
					if ((chaid > 1000) &&  (hullsI[io].size() > 3) && (defects[io].size() <22)&& (chaid < 10000))
				    {
					
						if (count > 800 && countR>200) //neural contours (threshold is the amount of green pixels in the bounding rectangle around blucontours) 
							nucontours.push_back(blucontours[io]);
				
						else
						{
						// FIND ASTROCYTES: large, oval & textured
						// Checking for aspect ratio (Oval), area and presense of large number of black pixels(textured)

						
							asprat[io] = brect[io].height / brect[io].width;// aspect ratio
							Mat imagee = enhance[i];
							Mat image_roi = imagee(arect[io]);// creating a new image from roi
							int countbl = (image_roi.rows*image_roi.cols) - countNonZero(image_roi);// counting number of black pixels
							if ((countbl < 500) && (asprat[io] >= 0.9 && asprat[io] <= 1.1))// if not very stippled and aspect ratio closer to 1 (more circular)-not astrocyte
								othcontours.push_back(blucontours[io]);// other contours
							else
								astcontours.push_back(blucontours[io]);//astrocytes
					}

				}
			}
			Mat outputcont(stackim[i].size(),CV_8UC4);
			outputcont=cv::Scalar(255,255,255,0);
			for (int nu = 0; nu < nucontours.size(); nu++)
				drawContours(outputcont, nucontours, nu, Scalar(0, 66335, 0,65535), 2, 8, vector<Vec4i>(), 0, Point());//green
			for (int du = 0; du < astcontours.size(); du++)  // finding and saving the contours that are above a certain area
				drawContours(outputcont, astcontours, du, Scalar(0, 65535, 65535,65535), 2, 8, vector<Vec4i>(), 0, Point());//yellow
			for (int ou = 0; ou < othcontours.size(); ou++)  // finding and saving the contours that are above a certain area
				drawContours(outputcont, othcontours, ou, Scalar(35000, 0, 65535,65535), 2, 8, vector<Vec4i>(), 0, Point());//pink


			/*~~~~~~*///drawthreshold(stackim, i, imname);
      string name = format("z%d_mod", i);
	    writeImage(name, outputcont);
			cellmetrics(imm, i, blucontours, nucontours, astcontours, myfile); // for the particular z-layer

			int beta = i + 1;// red channel
			myfile << "," ;
			synapcalc(imm, beta, stackim, nucontours); // calculating presence of red aroung neural cells
			myfile << ","  ;
			synapcalc(imm, beta, stackim, astcontours);// calculating count of red around astrocytes
		}

}


int main(int argc, char** argv)
{

  std::vector<cv::Mat> channel; 
  int valred = 255, valblue = 255, valgreen = 255; 
  Mat im_color;
	vector<Mat> stackim; 
  string raw_path;
  imname=argv[1];
	myfile.open("elilyfiles.csv");//std::ofstream::out | std::ofstream::app
	
  // reading 16 bit image

				channel.clear();
				stackim.clear();
				raw_path = format("%s.tif",argv[1]);  // 002002-1
				string imname = argv[1];

				cout << format("Processing %s", imname.c_str()) << endl;
        myfile<< "Image name"<<","<<"Number of blue contours"<<","<<"Number of blue contours with large area" <<","<<"Number of blue contours with medium area"<<","<<"Number of blue contours with small area"<<","<<"Average area of blue contours"<<","<<"Average aspect ratio of blue contours"<<","<<"Average diameter of blue contours"<<",";
				myfile<<"Number of Neural contours"<<","<<"Number of Neural contours with large area" <<","<<"Number of Neural Contours with medium area"<<","<<"Number of Neural Contours with small area"<<","<<"Average area of Neural Contours"<<","<<"Average aspect ratio of Neural Contours"<<","<<"Average diameter of Neural Contours"<<",";
		    myfile<<"Number of astrocytes contours"<<","<<"Number of astrocytes with large area"<<"," <<"Number of astrocytes with medium area"<<","<<"Number of astrocytes with small area"<<","<<"Average area of astrocytes"<<","<<"Average aspect ratio of astrocytes"<<","<<"Average diameter of astrocytes";
        myfile << ","<<"Total number of synapses(avg) around neural contours"<<","<<"Number of low intensity synapse around neural contours"<<","<<"Number of medium intensity synapse around neural contours"<<","<<"Number of high intensity synapse around neural contours";
				myfile <<","<<"Total number of synapses(avg) around astrocytes"<<","<<"Number of low intensity synapse around astrocytes"<<","<<"Number of medium intensity synapse around astrocytes"<<","<<"Number of high intensity synapse around astrocytes";
				myfile<<","<<"Average low int synapse around low int syn"<<","<<"Average med int synapse around low int syn"<<","<<"Average high int synapse around low int syn"<<",";
	      myfile<<"Average low int synapse around med int syn"<<","<<"Average med int synapse around med int syn"<<","<<"Average high int synapse around med int syn"<<",";
	      myfile<<"Average low int synapse around high int syn"<<","<<"Average med int synapse around high int syn"<<","<<"Average high int synapse around high int syn"<<",";
				myfile <<"Count of Low Int Syn arnd. Low Int. Syn(Dist=40)"<< ","<<"Count of Med Int Syn arnd. Low Int. Syn(Dist=40)"<< ","<<"Count of High Int Syn arnd. Low Int. Syn(Dist=40)"<<",";
	      myfile <<"Count of Low Int Syn arnd. med Int. Syn(Dist=40)"<< ","<<"Count of Med Int Syn arnd. med Int. Syn(Dist=40)"<< ","<<"Count of High Int Syn arnd. med Int. Syn(Dist=40)"<<",";
	      myfile <<"Count of Low Int Syn arnd. high Int. Syn(Dist=40)"<< ","<<"Count of Med Int Syn arnd. high Int. Syn(Dist=40)"<< ","<<"Count of High Int Syn arnd. high Int. Syn(Dist=40)"<<",";
		    myfile <<"Count of Low Int Syn arnd. Low Int. Syn(Dist=80)"<< ","<<"Count of Med Int Syn arnd. Low Int. Syn(Dist=80)"<< ","<<"Count of High Int Syn arnd. Low Int. Syn(Dist=80)"<<",";
	      myfile <<"Count of Low Int Syn arnd. med Int. Syn(Dist=80)"<< ","<<"Count of Med Int Syn arnd. med Int. Syn(Dist=80)"<< ","<<"Count of High Int Syn arnd. med Int. Syn(Dist=80)"<<",";
	      myfile <<"Count of Low Int Syn arnd. high Int. Syn(Dist=80)"<< ","<<"Count of Med Int Syn arnd. high Int. Syn(Dist=80)"<< ","<<"Count of High Int Syn arnd. high Int. Syn(Dist=80)"<<",";
				myfile<<"Total number of dendrites"<<","<<"Number of developed dendrites"<<","<<"Number of less developed dendrites"<<",";
				myfile << "Average low int synpases arnd high width dendrites" << "," << "Average med int synpases arnd high width dendrites" << "," << "Average high int synpases arnd high width dendrites" << ", " << "Average low int synpases arnd small width dendrites" << "," << "Average med int synpases arnd small width dendrites" << "," << "Average high int synpases arnd small width dendrites" << endl;
				cv::imreadmulti(raw_path, channel, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

				if (channel.empty())
				{	cout << "On no!!Image not found!" << endl; return(0);}

				for (unsigned int i = 0; i < channel.size(); i++)//channel.size() =number of z planes
				{

					Mat src = Mat::zeros(channel[i].rows, channel[i].cols, CV_16UC1);;
					Mat empty_image = Mat::zeros(channel[i].rows, channel[i].cols, CV_16UC1);
					Mat result_blue(channel[i].rows, channel[i].cols, CV_16UC3); // notice the 3 channels here!
					Mat result_green(channel[i].rows, channel[i].cols, CV_16UC3); // notice the 3 channels here!
					Mat result_red(channel[i].rows, channel[i].cols, CV_16UC3); // notice the 3 channels here!

					src = channel[i];///stores values of channel [i] temporarily

					if (i == 0 || (i % 3 == 0))// channel blue
					{
						/*if I have 8bit gray, and create a new empty 24bit RGB, I can copy the entire 8bit gray into one of the BGR channels (say, R),
						leaving the others black, and that effectively colorizes the pixels in a range of red.
						Similar, if the user wants to make it, say, RGB(80,100,120) then I can set each of the RGB channels to the source grayscale
						intensity multiplied by (R/255) or (G/255) or (B/255) respectively. This seems to work visually.
						It does need to be a per-pixel operation though cause the color applies only to a user-defined range of grayscale intensities.*/
						Mat in1[] = { src, empty_image, empty_image };
						int from_to1[] = { 0, 0, 1, 1, 2, 2 };
						mixChannels(in1, 3, &result_blue, 1, from_to1, 3);
						result_blue = changeimg(result_blue, 50, 0);
						string iname = format("z%d",i);
						writeImage(iname, result_blue);
						stackim.push_back(result_blue);
					}
					if (i == 2 || (i % 3 == 2))// channel red
					{
						Mat in2[] = { empty_image, src, empty_image };
						int from_to2[] = { 0, 0, 1, 1, 2, 2 };
						mixChannels(in2, 3, &result_red, 1, from_to2, 3);
						result_red = changeimg(result_red, 50, 0);
						string iname = format("z%d",i);
						writeImage(iname, result_red);
						stackim.push_back(result_red);
					}
					if (i == 1 || (i % 3 == 1))// channel green
					{
						Mat in3[] = { empty_image, empty_image, src };
						int from_to3[] = { 0, 0, 1, 1, 2, 2 };
						mixChannels(in3, 3, &result_green, 1, from_to3, 3);
						result_green = changeimg(result_green, 50, 0);
						string iname = format("z%d",i);
						writeImage(iname, result_green);
						stackim.push_back(result_green);

					}


				}
				//---------dendrite stuff-----------------
				Mat redlow, redmed, redhigh;

        int countofloop=0;
				for (int i = 0; i < stackim.size(); i++)
				{
				  
          countofloop++;
          if ((i==0)|| (i%3==0))//channel blue
          {
          string writename = format("%s_%d", imname.c_str(), i);
          myfile<<writename<<",";
         
          enhanceImage(imname, stackim, enhance,i);// thresholds all three channels for all z planes of one image and writes it
        
          cellcount(imname, stackim, enhance,i);// finds blue contours, astrocytes, nueral cells and other cells and calculates the metrics for these
          }
					if (i == 2 || (i % 3 == 2))// channel green
					{
            
						reddetect(stackim[i - 1], redlow, redmed, redhigh,argv[1],i-1);
						
						filterHessian(argv[1],i, stackim[i], redlow, redmed, redhigh, myfile);//detects dendrites
						//dendritedetect(imname, stackim[i], redlow, redmed, redhigh, myfile);
					}
					if (i == 1 || (i % 3 == 1))
						redinf(imname, stackim[i], myfile);
        if(countofloop==3)
        {
          myfile<<endl;countofloop=0;
          
        }
        
				}
	enhance.clear();

	myfile.close();
	return(0);
}





