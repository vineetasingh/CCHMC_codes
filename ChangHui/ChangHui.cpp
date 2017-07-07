//Program to analyze Chang Hui's images using MIP(Works on single images)

#include <iostream>     // std::cout
#include <algorithm>    // std::min_element, std::max_element
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include<vector>
#include "opencv2/opencv.hpp"
#include <tiffio.h>

using namespace cv;
using namespace std;
int gauKsize = 11;
ofstream myfile;

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

				new_image.at<Vec3w>(y, x)[c] = (alpha*(image.at<Vec3w>(y, x)[c]) + beta);
			}
		}
	}

	normalize(new_image, new_image, 0, 65535, NORM_MINMAX);
	return new_image;
}

Mat mip(vector<Mat> input, int channel)
{
	Mat layer1 = input[0];
	Mat result = Mat::zeros(layer1.rows, layer1.cols, layer1.type());
	Mat res = Mat::ones(layer1.rows, layer1.cols, layer1.type());

	//cout << layer1.size() << " " << result.size() << endl;
	float max = 0; int i, j, k;
	for (i = 0; i< layer1.rows; i++)
	{
		for (j = 0; j < layer1.cols; j++)
		{
			max = 0;// layer1.at<ushort>(Point(j, i));
			for (k = 0; k < input.size(); k++)
			{
				if ((input[k].at<Vec3s>(Point(j, i)))[channel] > max)
				{
					//cout << i << " " << j << " " << k << " " << input[k].at<ushort>(Point(j, i)) << " " << max << endl;
					max = input[k].at<Vec3s>(Point(j, i))[channel];
					//cout << max<<endl;
					result.at<Vec3s>(Point(j, i))[channel] = input[k].at<Vec3s>(Point(j, i))[channel];


				}

			}
		}
	}

	return result;
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
	//myfile << "neignborsyn" << "," << "Average no of low int synapse arnd  int syn(40)" << "," << "Average no of med int synapse arnd  int syn(40)" << "," << "Average no of high int synapse arnd low int syn(40)" << ",";
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
	//myfile << "," << "aroundsyn" << "," << "Average no of low int synapses around" << "," << "Average no of med int synapses around"  << "," << "Average no of high int synapses around"  << ",";
	if (CoordinatesX.size() != 0)
		myfile  << totl / CoordinatesX.size() << "," << totm / CoordinatesX.size() << "," << toth / CoordinatesX.size() << ",";
	else
		myfile << 0 << "," << 0 << "," << 0 << ",";
}

//-- - [7]----
void redinf(string imname, Mat im, ofstream &myfile)
{
	Mat enhancedl; vector< int> CoordinateLX; vector< int> CoordinateLY; vector< int> CoordinateMX; vector< int> CoordinateMY;
	vector< int> CoordinateHX; vector< int> CoordinateHY;

	// Enhance the red low channel
	cv::inRange(im, cv::Scalar(0, 0, 10000), cv::Scalar(500, 500, 20000), enhancedl);
	erode(enhancedl, enhancedl, Mat());
	Mat redlow = enhancedl.clone();
	Mat nonZeroCoordinatesL;
	findNonZero(enhancedl, nonZeroCoordinatesL);

	Mat enhancedm;
	// Enhance the red medium channel
	cv::inRange(im, cv::Scalar(0, 0, 20000), cv::Scalar(500, 500, 40000), enhancedm);
	erode(enhancedm, enhancedm, Mat());
	Mat redmed = enhancedm.clone();
	Mat nonZeroCoordinatesM;
	findNonZero(enhancedm, nonZeroCoordinatesM);

	Mat enhancedh;
	// Enhance the red high channel
	cv::inRange(im, cv::Scalar(0, 0, 40000), cv::Scalar(500, 500, 65535), enhancedh);
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
void reddetect(Mat im, Mat & redlow, Mat & redmed, Mat & redhigh)
{
	//(0, 0, 40000), cv::Scalar(500, 500, 65535)- high intensity; (0, 0, 20000), cv::Scalar(500, 500, 40000)- med intensity; (0, 0, 10000), cv::Scalar(500, 500, 20000)- low intensity
	Mat enhancedl;
	// Enhance the red low channel
	cv::inRange(im, cv::Scalar(0, 0, 10000), cv::Scalar(500, 500, 20000), enhancedl);
	redlow = enhancedl.clone();
	imwrite("redlow.tif", enhancedl);

	Mat enhancedm;
	// Enhance the red medium channel
	cv::inRange(im, cv::Scalar(0, 0, 20000), cv::Scalar(500, 500, 40000), enhancedm);
	redmed = enhancedm.clone();
	imwrite("redmed.tif", redmed);

	Mat enhancedh;
	// Enhance the red high channel
	cv::inRange(im, cv::Scalar(0, 0, 40000), cv::Scalar(500, 500, 65535), enhancedh);
	redhigh = enhancedh.clone();
	imwrite("redhigh.tif", redhigh);


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

void filterHessian(string imname, Mat image, Mat redlow, Mat redmed, Mat redhigh, ofstream &myfile)
{
	int co = 0;
	imwrite("remove.png", image); image = imread("remove.png");// to covert the 16 bit mage to 8 bit
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
	string name = format("%s_dend.png", imname.c_str());
	imwrite(name, outputDend);
	myfile << countofdendrites << " ," << developed << ", " << lessdeveloped << ",";
	dendritecalc(imname, redlow, redmed, redhigh, highIntgreenthreshimg, maskimage, developed, lessdeveloped, myfile);
	

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



int main(int argc, char** argv)
{
	std::vector<cv::Mat> channel;
	string imname = format("%s.tif", argv[1]);
	vector<Mat> redstackim;
	vector<Mat> grstackim;
	vector<Mat> stackim;
  Mat redlow, redmed, redhigh;
  myfile.open("ChangHuit.csv");
	cout << format("Processing %s", imname.c_str()) << endl;
  myfile << "Image name" << ",";
  myfile << "Total no of dendrites" << "," << " No of Developed dendrites  " << ", " << " No of Less Developed dendrites  " << ",";
	myfile << "Average low int synpases arnd high width dendrites" << "," << "Average med int synpases arnd high width dendrites" << "," << "Average high int synpases arnd high width dendrites" << ", " << "Average low int synpases arnd small width dendrites" << "," << "Average med int synpases arnd small width dendrites" << "," << "Average high int synpases arnd small width dendrites" << ",";
	//3 times
	myfile << "Average no of low int synapses around low int synapses " << "," << "Average no of med int synapses around low int synapses" << "," << "Average no of high int synapses around low int synapses" << ",";
	myfile << "Average no of low int synapses around med int synapses" << "," << "Average no of med int synapses around med int synapses" << "," << "Average no of high int synapses around med int synapses" << ",";
	myfile << "Average no of low int synapses around high int synapses" << "," << "Average no of med int synapses around high int synapses" << "," << "Average no of high int synapses around  high high int synapses" << ",";
	//6 times
	myfile << "Average no of low int synapse arnd low int syn(40)" << "," << "Average no of med int synapse arnd low int syn(40)" << "," << "Average no of high int synapse arnd low int syn(40)" << ",";
	myfile << "Average no of low int synapse arnd med int syn(40)" << "," << "Average no of med int synapse arnd med int syn(40)" << "," << "Average no of high int synapse arnd med int syn(40)" << ",";
	myfile << "Average no of low int synapse arnd high int syn(40)" << "," << "Average no of med int synapse arnd high int syn(40)" << "," << "Average no of high int synapse arnd high int syn(40)" << ",";
	myfile << "Average no of low int synapse arnd low int syn(80)" << "," << "Average no of med int synapse arnd  low int syn(80)" << "," << "Average no of high int synapse arnd low int syn(80)" << ",";
	myfile << "Average no of low int synapse arnd med int syn(80)" << "," << "Average no of med int synapse arnd med int syn(80)" << "," << "Average no of high int synapse arnd med int syn(80)" << ",";
	myfile << "Average no of low int synapse arnd high int syn(80)" << "," << "Average no of med int synapse arnd high int syn(80)" << "," << "Average no of high int synapse arnd high int syn(80)" << "," << endl;
  cv::imreadmulti(imname, channel, CV_LOAD_IMAGE_ANYDEPTH| CV_LOAD_IMAGE_ANYCOLOR );// to read multiple channels

	if (channel.empty())
	{
	  cout << "Image not found!" << endl;
		return 0;
	}
  myfile <<  argv[1]<<",";
	for (unsigned int i = 0; i < channel.size(); i++)
	{

		Mat src = Mat::zeros(channel[i].rows, channel[i].cols, CV_16UC1);
		Mat empty_image = Mat::zeros(channel[i].rows, channel[i].cols, CV_16UC1); // notice the 2 channels here!
		Mat result_blue = Mat::zeros(channel[i].rows, channel[i].cols, CV_16UC3);
		Mat result_green = Mat::zeros(channel[i].rows, channel[i].cols, CV_16UC3); // notice the 2 channels here!
		Mat result_red = Mat::zeros(channel[i].rows, channel[i].cols, CV_16UC3); // notice the 2 channels here!
		src = channel[i];///stores values of channel [i] temporarily

		if ((i == 0) || (i % 2 == 0))// channel red
		{
			/*if I have 8bit gray, and create a new empty 24bit RGB, I can copy the entire 8bit gray into one of the BGR channels (say, R),
			leaving the others black, and that effectively colorizes the pixels in a range of red.
			Similar, if the user wants to make it, say, RGB(80,100,120) then I can set each of the RGB channels to the source grayscale
			intensity multiplied by (R/255) or (G/255) or (B/255) respectively. This seems to work visually.
			It does need to be a per-pixel operation though cause the color applies only to a user-defined range of grayscale intensities.*/
			Mat in1[] = { empty_image, empty_image, empty_image };
			int from_to1[] = { 0, 0, 1, 1, 2, 2 };
			mixChannels(in1, 3, &result_blue, 1, from_to1, 3);
			//result_blue = changeimg(result_blue, 10, 0);
			//string iname = format("Vin_%d.tif", i);
			//imwrite(iname, result_blue);
			//bluestackim.push_back(result_blue);

			Mat in3[] = { empty_image, empty_image, src };
			int from_to3[] = { 0, 0, 1, 1, 2, 2 };
			mixChannels(in3, 3, &result_red, 1, from_to3, 3);
			result_red = changeimg(result_red, 200, 0);
			//imwrite("result_red.tif", result_red);
			string iname = format("Layer_%d.tif", i);
			imwrite(iname, result_red);
			redstackim.push_back(result_red);
			
		}

		if (i == 1 || (i % 2 == 1))// channel green
		{
			Mat in2[] = { empty_image, src, empty_image };
			int from_to2[] = { 0, 0, 1, 1, 2, 2 };
			mixChannels(in2, 3, &result_green, 1, from_to2, 3);
			result_green = changeimg(result_green, 200, 100);
			//imwrite("result_gre.tif", result_green);
			string iname = format("Layer_%d.tif", i);
			imwrite(iname, result_green);
			grstackim.push_back(result_green);

		}

	}
		Mat resultgreen = mip(grstackim, 1);
		Mat greenmip = changeimg(resultgreen, 10, 10);
		imwrite("Greenmip.tif", greenmip);
		Mat resultred = mip(redstackim, 2);
		Mat redmip = changeimg(resultred, 10, 10);
		imwrite("Redmip.tif", redmip);
		stackim.push_back(greenmip);
	  stackim.push_back(redmip);
		for (int i = 0; i < stackim.size(); i++)
	  {

		if (i == 1 || (i % 2 == 1))// channel red
		  {

			  reddetect(stackim[i], redlow, redmed, redhigh);//detects synapses
			  filterHessian(argv[1], stackim[i - 1], redlow, redmed, redhigh, myfile);//detects dendrites
			  redinf(argv[1], stackim[i], myfile);
		  }


	  }

	
	return 0;
}
