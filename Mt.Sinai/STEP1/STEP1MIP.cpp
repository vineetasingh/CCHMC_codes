
#include <iostream>     // std::cout
#include <algorithm>    // std::min_element, std::max_element
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
#include<vector>
//#include "opencv2/opencv.hpp "


#define MIN(a,b) ((a) < (b) ? (a) : (b))
using namespace cv;
using namespace std;
const int somathreshold = 2000;
const int bluecontourthreshold = 1000;
RNG rng;


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
				if ((input[k].at<Vec3b>(Point(j, i)))[channel] > max)
				{
					//cout << i << " " << j << " " << k << " " << input[k].at<ushort>(Point(j, i)) << " " << max << endl;
					max = input[k].at<Vec3b>(Point(j, i))[channel];
					//cout << max<<endl;
					result.at<Vec3b>(Point(j, i))[channel] = input[k].at<Vec3b>(Point(j, i))[channel];


				}

			}
		}
	}

	return result;
}

Mat changeimg(Mat image, float alpha, float beta)
{
	alpha = alpha / 10;
	/// Do the operation new_image(i,j) = alpha*image(i,j) + beta
	Mat new_image = Mat::zeros(image.size(), image.type());
	Mat blurr;

	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			for (int c = 0; c < 3; c++)
			{
				//cout << new_image.at<Vec3s>(y,x)[c] << endl;

				new_image.at<Vec3b>(Point(j, i))[c] = (alpha*(image.at<Vec3b>(Point(j, i))[c]) + beta);
				//new_image.at<Vec3w>(y, x)[c] = 250 * pow(new_image.at<Vec3w>(y, x)[c], 0.5);

			}
		}
	}

	cv::GaussianBlur(new_image, blurr, cv::Size(0, 0), 3);
	cv::addWeighted(new_image, 2.0, blurr, -1.0, 0, new_image);
	normalize(new_image, new_image, 0, 255, NORM_MINMAX);
	return new_image;
}



void arounddendrite(vector<RotatedRect> rotRect, ofstream &myfile, int CountLW, int CountMW, int CountHW)
{// average count of low, med and high int synapses aroung LMH width dendrites
	Mat thr_high_Synapse = imread("z_highsynapse.tif");
	Mat Medium_Synapse_thr = imread("z_medsynapse.tif");
	Mat Low_Synapse_thr = imread("z_lowsynapse.tif");
	int LWlcount = 0, LWmcount = 0, LWhcount = 0, MWlcount = 0, MWmcount = 0, MWhcount = 0, HWlcount = 0, HWmcount = 0, HWhcount = 0;
	int totLWlcount = 0, totLWmcount = 0, totLWhcount = 0, totMWlcount = 0, totMWmcount = 0, totMWhcount = 0, totHWlcount = 0, totHWmcount = 0, totHWhcount = 0;
	int avgLWlcount = 0, avgLWmcount = 0, avgLWhcount = 0, avgMWlcount = 0, avgMWmcount = 0, avgMWhcount = 0, avgHWlcount = 0, avgHWmcount = 0, avgHWhcount = 0;

	int totlowIntSyn = countNonZero(Low_Synapse_thr.reshape(1));
	int totmedIntSyn = countNonZero(Medium_Synapse_thr.reshape(1));
	int tothigIntSyn = countNonZero(thr_high_Synapse.reshape(1));

	for (int i = 0; i < rotRect.size(); i++)
	{
		LWlcount = 0, LWmcount = 0, LWhcount = 0, MWlcount = 0, MWmcount = 0, MWhcount = 0, HWlcount = 0, HWmcount = 0, HWhcount = 0;
		Mat Hrotated, Mrotated, Lrotated;
		cv::Mat Hcropped, Mcropped, Lcropped;
		cv::Mat rot_mat = cv::getRotationMatrix2D(rotRect[i].center, rotRect[i].angle, 1);
		if (rotRect[i].angle < -45.)
			std::swap(rotRect[i].size.width, rotRect[i].size.height);
		warpAffine(thr_high_Synapse, Hrotated, rot_mat, thr_high_Synapse.size(), INTER_CUBIC);
		// crop the resulting image
		cv::getRectSubPix(Hrotated, rotRect[i].size, rotRect[i].center, Hcropped);
		warpAffine(Medium_Synapse_thr, Mrotated, rot_mat, Medium_Synapse_thr.size(), INTER_CUBIC);
		// crop the resulting image
		cv::getRectSubPix(Mrotated, rotRect[i].size, rotRect[i].center, Mcropped);
		warpAffine(Medium_Synapse_thr, Lrotated, rot_mat, Low_Synapse_thr.size(), INTER_CUBIC);
		// crop the resulting image
		cv::getRectSubPix(Lrotated, rotRect[i].size, rotRect[i].center, Lcropped);// low int synapse rotated rect (ROI)  image
		float wdth = MIN(rotRect[i].size.width, rotRect[i].size.height);
		if (wdth >= 1)
		{
			if (wdth <= 50) //low width dendrite
			{
				LWlcount = countNonZero(Lcropped.reshape(1)); // no of low int syn around a Low width dendrite
				LWmcount = countNonZero(Mcropped.reshape(1));// no of med int syn around a Low width dendrite
				LWhcount = countNonZero(Hcropped.reshape(1));// no of high int syn around a Low width dendrite
				//cout << countNonZero(Lcropped) << ", " << countNonZero(Mcropped) << ",  " << countNonZero(Hcropped) << endl;
				totLWlcount += LWlcount; totLWmcount += LWmcount;  totLWhcount += LWhcount;
			}
			else if (wdth > 50 && wdth <= 100)
			{
				MWlcount = countNonZero(Lcropped.reshape(1));
				MWmcount = countNonZero(Mcropped.reshape(1));
				MWhcount = countNonZero(Hcropped.reshape(1));
				totMWlcount += MWlcount; totMWmcount += MWmcount;  totMWhcount += MWhcount;
			}
			else
			{
				HWlcount = countNonZero(Lcropped.reshape(1));
				HWmcount = countNonZero(Mcropped.reshape(1));
				HWhcount = countNonZero(Hcropped.reshape(1));
				totHWlcount += HWlcount; totHWmcount += HWmcount;  totHWhcount += HWhcount;
			}
		}
	}
	if (CountLW == 0) { avgLWlcount = 0; avgLWmcount = 0; avgLWhcount = 0; }
	else { avgLWlcount = totLWlcount / CountLW;  avgLWmcount = totLWmcount / CountLW; avgLWhcount = totLWhcount / CountLW; }
	if (CountMW == 0) { avgMWlcount = 0; avgMWmcount = 0; avgMWhcount = 0; }
	else { avgMWlcount = totMWlcount / CountMW; avgMWmcount = totMWmcount / CountMW; avgMWhcount = totMWhcount / CountMW; }
	if (CountHW == 0) { avgHWlcount = 0; avgHWmcount = 0; avgHWhcount = 0; }
	else { avgHWlcount = totHWlcount / CountHW; avgHWmcount = totHWmcount / CountHW; avgHWhcount = totHWhcount / CountHW; }
	//myfile << "," << "arounddendrite" << "," << "Total number of low intensity synapse" <<"," << "Total number of med intensity synapses" << "," << "Total number of high intensity synapses" <<","<< "Avg Low Int Syn count arnd. Low width" << "," << "Avg Med Int Syn count arnd. Low width" << "," << "Avg Low Int Syn count arnd. High width" << "," << "Avg Low Int Syn count arnd. Med width" << "," << " Avg Med Int Syn count arnd. Med width" << "," << "Avg High Int Syn count arnd. Med width" << "," << "Avg Low Int Syn count arnd. Large width " << "," << "Avg Med Int Syn count arnd. Large width" << "," << "Avg High Int Syn count arnd. Large width";
	myfile << totlowIntSyn << "," << totmedIntSyn << "," << tothigIntSyn << "," << avgLWlcount << "," << avgLWmcount << "," << avgLWhcount << "," << avgMWlcount << "," << avgMWmcount << "," << avgMWhcount << "," << avgHWlcount << "," << avgHWmcount << "," << avgHWhcount;
}

void dendritedetect(vector<vector<Point> > hcontours, ofstream & myfile)
{
	int Swid = 0, Mwid = 0, Lwid = 0, totSwid = 0, totMwid = 0, totLwid = 0, avgSwid = 0, avgMwid = 0, avgLwid = 0, avgSlen = 0, avgMlen = 0, avgLlen = 0;
	int totSlen = 0, totMlen = 0, totLlen = 0; int avgoverallwdth = 0, avgoverallen = 0;
	vector<RotatedRect> minRect(hcontours.size());
	int wdth = 0, swidth = 0, mwidth = 0, lwidth = 0;
	int slen = 0, mlen = 0, llen = 0;
	for (int i = 0; i < hcontours.size(); i++)
	{
		minRect[i] = minAreaRect(Mat(hcontours[i]));
	}
	for (int i = 0; i< hcontours.size(); i++)
	{

		Size2f s = minRect[i].size;
		wdth = MIN(s.width, s.height);
		if (wdth >= 1) // binning based on rectangle widths
		{
			if (wdth <= 50)
			{
				totSwid += wdth;
				swidth++;
			}
			else if (wdth > 50 && wdth <= 100)
			{
				totMwid += wdth;
				mwidth++;
			}
			else
			{
				totLwid += wdth;
				lwidth++;
			}
		}
		if ((arcLength(hcontours[i], true)) <= 350)//binning based on length of contours
		{
			totSlen += (arcLength(hcontours[i], true));
			slen++;
		}
		else if ((arcLength(hcontours[i], true) >= 350) && (arcLength(hcontours[i], true) <= 600))
		{
			totMlen += (arcLength(hcontours[i], true));
			mlen++;
		}
		else
		{
			totLlen += (arcLength(hcontours[i], true));
			llen++;
		}

	}
	if (slen == 0) avgSlen = 0;
	else avgSlen = totSlen / slen;
	if (mlen == 0) avgMlen = 0;
	else avgMlen = totMlen / mlen;
	if (llen == 0) avgLlen = 0;
	else  avgLlen = totLlen / llen;
	if (swidth == 0) avgSwid = 0;
	else avgSwid = totSwid / swidth;
	if (mwidth == 0) avgMwid = 0;
	else avgMwid = totMwid / mwidth;
	if (lwidth == 0) avgLwid = 0;
	else avgLwid = totLwid / lwidth;
	if ((swidth + mwidth + lwidth) == 0) avgoverallwdth = 0;
	else avgoverallwdth = (totSwid + totMwid + totLwid) / (swidth + mwidth + lwidth);
	if ((slen + mlen + llen) == 0)avgoverallen = 0;
	else avgoverallen = (totSlen + totMlen + totLlen) / (slen + mlen + llen);
	myfile <<"," << swidth + mwidth + lwidth << "," << swidth << "," << mwidth << "," << lwidth << "," << slen << "," << mlen << "," << llen << "," << avgoverallen << "," << avgSlen << "," << avgMlen << "," << avgLlen << "," << avgoverallwdth << "," << avgSwid << "," << avgMwid << "," << avgLwid << ",";
	/*~~~~~~~~~~~~~~*/arounddendrite(minRect, myfile, swidth, mwidth, lwidth);
	//myfile << "," << "DENDRITE" << "," << "Total No of Dendrites" << "," << "No.of Small width Dendrites" << "," << " No.of med width Dendrites" << "," << "No.of large width Dendrites" << "," << "No.of Small length Dendrites" << "," << "No.of Med length Dendrites" << "," << "No.of Large length Dendrites" << "," << "Avg len of all Dendrites" << "," << "Avg len of small length Dendrites" << "," << "Avg len of med length Dendrites" << "," << "Avg len of large length Dendrites" << "," << "Avg wdth of all Dendrites" << "," << "Avg wdth of small width Dendrites" << "," << "Avg wdth of med width Dendrites" << "," << "Avg wdth of large width Dendrites" << ",";

}
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

void filterHessian(string imname, Mat image, ofstream &myfile)
{
	string stringden = format("%s_dend_processed.png", imname.c_str());
	int co = 0;
	int gauKsize = 11;
	Mat org = image.clone();
	Mat orgclone = org.clone();

	Mat outputDend(org.size(),CV_8UC4);
	outputDend=cv::Scalar(255,255,255,0);

	cvtColor(image, image, CV_BGR2GRAY);
	Mat checkimg(image.rows, image.cols, CV_8U);
	Mat overlapimage(image.rows, image.cols, CV_8U);
	Mat dendritetips(image.rows, image.cols, CV_8U);
	Mat overlapbinimage(image.rows, image.cols, CV_8U);
	cv::Mat dXX, dYY, dXY;
	std::vector<float> eigenvalues(2);
	cv::Mat hessian_matrix(2, 2, CV_32F);
	Mat eigenvec = Mat::ones(2, 2, CV_32F);
	vector<vector<Point> > contours; vector<vector<Point> > finalcontours;
	vector<Vec4i> hierarchy;
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
	dilate(maskimage, maskimage, Mat());
	findContours(maskimage, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	for (int d = 0; d < contours.size(); d++)
	{
		float chaid = contourArea(contours[d], false);
		if (chaid > 2500)
		{
			finalcontours.push_back(contours[d]);
			drawContours(outputDend, contours, d, Scalar(0, 255, 125,255), 2, 8, vector<Vec4i>(), 0, Point());
		}
	}
	imwrite(stringden, outputDend);
	//cout << finalcontours.size()<< endl;
	dendritedetect(finalcontours, myfile);
	//myfile << "," << "Dendrite begins:  " << ", " << imname << " ," << countofdendrites << " ," << developed << ", " << lessdeveloped << ",";
	//dendritecalc(imname, redlow, redmed, redhigh, highIntgreenthreshimg, maskimage, developed, lessdeveloped, myfile);

}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void aroundbluecontours(Mat rhigh, Mat rmed, Mat rlow, vector<vector<Point> > contours, ofstream &myfile)
{
	int single_redh_count = 0, single_redm_count = 0, single_redl_count = 0, totrh = 0, totrm = 0, totrl = 0, avgrh = 0, avgrm = 0, avgrl = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		single_redh_count = 0, single_redm_count = 0, single_redl_count = 0;
		Rect minRect;
		minRect = boundingRect(Mat(contours[i]));
		Mat image_high = rhigh(minRect);
		Mat image_med = rmed(minRect);
		Mat image_low = rlow(minRect);
		single_redh_count = countNonZero(image_high);
		single_redm_count = countNonZero(image_med);
		single_redl_count = countNonZero(image_low);
		totrh += single_redh_count;
		totrm += single_redm_count;
		totrl += single_redl_count;
	}
	avgrh = totrh / contours.size();
	avgrm = totrm / contours.size();
	avgrl = totrl / contours.size();
	myfile << avgrh << "," << avgrm << "," << avgrl;

}
void findsoma(Mat im, Mat imr, vector<vector<Point> > blue_contours, vector<vector<Point> > intersection_contours, ofstream &myfile, string name)
{
	vector<vector<Point> > astrocytes;
	vector<Vec4i> astrocyteshierarchy;
	vector<vector<Point> > neurons;
	vector<Vec4i> neuronshierarchy;
	vector<vector<Point> > final_soma_contours;
	vector<Vec4i> final_soma_hierarchy;
	vector<vector<Point> > refine_intersect_contours;
	vector<Vec4i> afhierarchy;
	vector<vector<Point> > refine_blue_contours;
	vector<Vec4i> bfhierarchy;
	vector<vector<Point> > scontours;
	vector<Vec4i> shierarchy;
	int soma_indices[100];
	vector<vector<Point> > remcontours;
	vector<Moments> moment_blue(blue_contours.size());
	vector<Point2f> masscenter_blue(blue_contours.size());
	int ib = -1;
	int count = 0;
	vector<Point> som;
	vector<Moments> moment_intersect(intersection_contours.size());
	vector<Point2f> masscenter_intersect(intersection_contours.size());
	int i1 = -1;
	for (int i = 0; i < intersection_contours.size(); i++) // filter areas off intersection contour
	{
		if (fabs(contourArea(intersection_contours[i])) > somathreshold)
			refine_intersect_contours.push_back(intersection_contours[i]);
	}
	for (int i = 0; i < blue_contours.size(); i++) // filter areas of blue contours
	{
		if (fabs(contourArea(blue_contours[i])) > bluecontourthreshold)
			refine_blue_contours.push_back(blue_contours[i]);
	}
	/// Get the moments of blue contours
	for (int i = 0; i < refine_blue_contours.size(); i++)
		moment_blue[i] = moments(refine_blue_contours[i], false);
	///  Get the mass centers of blue contours:
	for (int i = 0; i < refine_blue_contours.size(); i++)
		masscenter_blue[i] = Point2f(moment_blue[i].m10 / moment_blue[i].m00, moment_blue[i].m01 / moment_blue[i].m00);
	/// Get the moments of intersection contours
	for (int i = 0; i < refine_intersect_contours.size(); i++)
		moment_intersect[i] = moments(refine_intersect_contours[i], false);
	///  Get the mass centers of intersection contours:
	for (int i = 0; i < refine_intersect_contours.size(); i++)
		masscenter_intersect[i] = Point2f(moment_intersect[i].m10 / moment_intersect[i].m00, moment_intersect[i].m01 / moment_intersect[i].m00);
	// check if centroid of any of the intersection contours is near the blue contours: If yes then- we have found a soma
	float d = 0; int j1 = 0; int ifin = -1; int j5 = -1; int jfin = 0;
	if (refine_blue_contours.size() == 0)
	{
		myfile << "," << "TOTAL" << "," << 0;
		myfile << "," << "SOMA" << " ," << 0 << " ," << 0 << " ," << 0 << ", " << 0 << " ," << 0;
		myfile << "," << "Astrocyte" << " ," << 0 << " ," << 0 << " ," << 0;
		myfile << "," << "Others" << " ," << 0 << " ," << 0 << " ," << 0;
	}
	else
	{
		for (int i = 0; i < refine_intersect_contours.size(); i++)
		{
			som.clear();
			float dmin = 1000000; int index = 0;
			//cout << "SIZE" << refine_blue_contours.size() << endl;
			for (int j = 0; j < refine_blue_contours.size(); j++)
			{
				d = sqrt(((masscenter_intersect[i].x - masscenter_blue[j].x)*(masscenter_intersect[i].x - masscenter_blue[j].x)) + ((masscenter_intersect[i].y - masscenter_blue[j].y)*(masscenter_intersect[i].y - masscenter_blue[j].y)));
				if (d < dmin)
				{
					dmin = d;
					index = j;
				}
			}
			final_soma_contours.push_back(refine_blue_contours[index]);
			soma_indices[i] = index;
		}
		// loop to find remaining contours
		for (int g = 0; g < refine_blue_contours.size(); g++)
		{
			int cc = 0;
			for (int h = 0; h < 100; h++)
			{
				if (g == soma_indices[h])// depicts a soma
					cc++;
			}
			if (cc == 0)
				remcontours.push_back(refine_blue_contours[g]); // not a soma
		}
		int alowcirc = 0, ahighcirc = 0;// circularity of astrocytes
		int nlowcirc = 0, nhighcirc = 0;// circularity of neural cells
		for (int k = 0; k < remcontours.size(); k++)
		{
			cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(remcontours[k]));
			float aspect_ratio = float(min_area_rect.size.width) / min_area_rect.size.height;
			if (aspect_ratio > 1.0) {
				aspect_ratio = 1.0 / aspect_ratio;
			}
			if (aspect_ratio >= 0.45 && aspect_ratio <= 0.6)
			{
				astrocytes.push_back(remcontours[k]);
				if (aspect_ratio <= 0.55)
					alowcirc++;
				else
					ahighcirc++;
			}
			else
			{
				neurons.push_back(remcontours[k]);
				if (aspect_ratio <= 0.55)
					nlowcirc++;
				else
					nhighcirc++;
			}
		}

		Mat thr_high_Synapse;
		cv::inRange(imr, cv::Scalar(0, 0, 200), cv::Scalar(30, 30, 255), thr_high_Synapse);//RED-HIGH INTENSITY
		// medium intensity
		Mat thr_medium_Synapse;
		cv::inRange(imr, cv::Scalar(0, 0, 100), cv::Scalar(10, 10, 200), thr_medium_Synapse);//RED -LOW+HIGH INTENSITY

		Mat thr_low_Synapse;
		cv::inRange(imr, cv::Scalar(0, 0, 50), cv::Scalar(10, 10, 100), thr_low_Synapse);




		Mat outputImage(im.size(),CV_8UC4);
		outputImage=cv::Scalar(255,255,255,0);
		int siz;
		vector<Point2f>center(final_soma_contours.size());
		vector<float>radius(final_soma_contours.size());
		vector<float> aspect_ratio(final_soma_contours.size());
		vector<float> diameter;
		float sumaspect = 0;
		float meanaspect = 0;
		float stddevaspect = 0;
		float sumdiameter = 0;
		float meandiameter = 0;
		float stddevdiameter = 0;
		siz = final_soma_contours.size();
		for (int k = 0; k < final_soma_contours.size(); k++)
		{
			minEnclosingCircle((Mat)final_soma_contours[k], center[k], radius[k]);
			circle(outputImage, center[k], radius[k], Scalar(20, 230, 125,255), 6, 8, 0);
			cv::RotatedRect min_area_rect = minAreaRect(cv::Mat(final_soma_contours[k]));
			aspect_ratio[k] = float(min_area_rect.size.width) / min_area_rect.size.height;
			if (aspect_ratio[k] > 1.0)
				aspect_ratio[k] = 1.0 / aspect_ratio[k];
		}
		for (int k = 0; k < final_soma_contours.size(); k++)
		{
			sumaspect += aspect_ratio[k];
			sumdiameter += (2 * radius[k]);
		}
		meanaspect = sumaspect / siz;
		meandiameter = sumdiameter / siz;
		for (int k = 0; k < final_soma_contours.size(); k++)
		{
			stddevaspect = ((meanaspect - aspect_ratio[k])*(meanaspect - aspect_ratio[k])) / siz;
			stddevdiameter = ((meandiameter - (2 * radius[k]))*(meandiameter - (2 * radius[k]))) / siz;
		}
		myfile << "," << refine_blue_contours.size();
		myfile << "," << final_soma_contours.size() << " ," << meandiameter << " ," << stddevdiameter << ", " << meanaspect << " ," << stddevaspect;
		myfile << "," << astrocytes.size() << " ," << alowcirc << " ," << ahighcirc;
		myfile << "," << neurons.size() << " ," << nlowcirc << " ," << nhighcirc;
		for (int j = 0; j < astrocytes.size(); j++)
			drawContours(outputImage, astrocytes, j, Scalar(255, 255, 255,255), 2, cv::LINE_8, vector<Vec4i>(), 0, Point());
		for (int j = 0; j < neurons.size(); j++)
			drawContours(outputImage, neurons, j, Scalar(255, 125, 255,255), 2, cv::LINE_8, vector<Vec4i>(), 0, Point());

		string out_string = format("%s_blue_processed.png", name.c_str());
		imwrite(out_string, outputImage);
		myfile << "," ;
		if (final_soma_contours.size()>0)
			aroundbluecontours(thr_high_Synapse, thr_medium_Synapse, thr_low_Synapse, final_soma_contours, myfile);
		else
			myfile << "0" << "," << "0" << "," << "0";

		myfile << ",";
		if (astrocytes.size()>0)
			aroundbluecontours(thr_high_Synapse, thr_medium_Synapse, thr_low_Synapse, astrocytes, myfile);
		else
			myfile << "0" << "," << "0" << "," << "0";

		myfile << "," ;
		if (neurons.size()>0)
			aroundbluecontours(thr_high_Synapse, thr_medium_Synapse, thr_low_Synapse, neurons, myfile);
		else
			myfile << "0" << "," << "0" << "," << "0";



	}
}
void areabinning(vector<vector<Point> > contours, string str, ofstream &myfile)
{
	int c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = fabs(contourArea((contours[i])));
		if (area >= 5 && area < 100)
			c1++;
		if (area >= 100 && area < 200)
			c2++;
		if (area >= 200 && area < 300)
			c3++;
		if (area >= 300 && area < 400)
			c4++;
		if (area >= 400 && area < 500)
			c5++;
		if (area >= 500)
			c6++;
	}
	myfile << " ," << "SYNAPSE" << "," << str << "," << c1 << "," << c2 << "," << c3 << "," << c4 << "," << c5 << "," << c6 << ",";
}
void synapse(Mat &imm, ofstream &myfile)
{

	Mat thr_high_Synapse;
	cv::inRange(imm, cv::Scalar(0, 0, 200), cv::Scalar(30, 30, 255), thr_high_Synapse);//RED-HIGH INTENSITY
	int highcount = countNonZero(thr_high_Synapse);
	// medium intensity
	Mat thr_medium_Synapse;

	cv::inRange(imm, cv::Scalar(0, 0, 100), cv::Scalar(10, 10, 200), thr_medium_Synapse);//RED -LOW+HIGH INTENSITY
	int medcount = countNonZero(thr_medium_Synapse);

	Mat thr_low_Synapse;
	cv::inRange(imm, cv::Scalar(0, 0, 50), cv::Scalar(10, 10, 100), thr_low_Synapse);
	int lowcount = countNonZero(thr_low_Synapse);

	imwrite("z_highsynapse.tif", thr_high_Synapse);
	imwrite("z_medsynapse.tif", thr_medium_Synapse);
	imwrite("z_lowsynapse.tif", thr_low_Synapse);
	/*~~~~~~~~~~~~*///areabinning(low_intensity_contours, "low intensity", myfile);
}
vector<vector<Point> > watershedcontours(Mat src, Mat grayB, Mat bw)
{
	int noofruns = 100;
	double min, max;
	Point maxLoc;
	vector<vector<Point> > contours_check, large_contours;
	Mat bin;
	vector<Mat> storewatershed;
	//cv::inRange(src, cv::Scalar(20, 0, 0), cv::Scalar(255, 30, 60), bw);//BLUE THRESH
	//imshow("Binary Image", bw);
	//thresholding nuclei (blue) in the image

	Mat kernel_op = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	Mat morph1;
	morphologyEx(bw, morph1, CV_MOP_OPEN, kernel_op);
	//imshow("openmorphology", morph1);
	Mat morph;
	Mat kernel2 = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(morph1, morph, CV_MOP_CLOSE, kernel2);
	erode(morph, morph, Mat());
	//imshow("after morphology", morph);
	bw = morph;
	// Perform the distance transform algorithm
	Mat dist;
	distanceTransform(bw, dist, CV_DIST_L2, 3);
	normalize(dist, dist, 0, 1., NORM_MINMAX);
	//imshow("Distance Transform Image", dist);
	threshold(dist, dist, .55, 1., CV_THRESH_BINARY);
	// Dilate a bit the dist image
	Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
	dilate(dist, dist, kernel1);
	//imshow("Peaks", dist);
	//cout << countNonZero(dist) << endl;
	if (countNonZero(dist) > 90000)
		noofruns = 80;
	// Create the CV_8U version of the distance image
	// It is needed for findContours()
	Mat dist_8u, distback_8u;
	dist.convertTo(dist_8u, CV_8U);
	// Find total markers
	vector<vector<Point> > contours, backcontours;
	findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	Mat dist_back;
	threshold(dist, dist_back, 0, 0.5, CV_THRESH_BINARY_INV);
	Mat kern_erod = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	for (int y = 0; y < noofruns; y++)
	{
		erode(dist_back, dist_back, kern_erod);
	}
	//imshow("PeaksBackground", dist_back);
	dist_back.convertTo(distback_8u, CV_8U);
	findContours(distback_8u, backcontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	Mat markers = Mat::zeros(dist.size(), CV_32SC1);
	// Draw the foreground markers
	for (size_t i = 0; i < contours.size(); i++)
	{
		drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
	}
	for (size_t i = 0; i < backcontours.size(); i++)
	{
		drawContours(markers, backcontours, (i), Scalar(255, 255, 255), -1);
	}
	int ncomp = contours.size();
	//imshow("Markers", markers * 10000);
	// Perform the watershed algorithm
	watershed(src, markers);
	vector<Vec3b> colors;
	for (size_t i = 0; i < contours.size(); i++)
	{
		int b = theRNG().uniform(0, 255);
		int g = theRNG().uniform(0, 255);
		int r = theRNG().uniform(0, 255);
		colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
	}
	vector<Mat> images2; vector<vector<Point> > final_contours, all_finalcontours;
	for (int obj = 1; obj < contours.size() + 1; obj++)
	{
		Mat dst2;
		src.copyTo(dst2, (markers == obj));
		cv::inRange(dst2, cv::Scalar(40, 0, 0), cv::Scalar(255, 10, 10), dst2);//BLUE THRESH
		morphologyEx(dst2, dst2, CV_MOP_CLOSE, kernel2);
		morphologyEx(dst2, dst2, CV_MOP_CLOSE, kernel2);
		findContours(dst2, final_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		for (int u = 0; u < final_contours.size(); u++)
			all_finalcontours.push_back(final_contours[u]);
		images2.push_back(dst2.clone());
	}
	for (size_t p = 0; p < all_finalcontours.size(); p++)
	{

		drawContours(src, all_finalcontours, p, Scalar(0,255,0), 1, cv::LINE_8, vector<Vec4i>(), 0, Point());
	}
	//imwrite("nuclei_processed.png", src);
	return all_finalcontours;



}
int main(int argc, char** argv)// folder name
{
	std::string file_contents;
	ofstream myfile;
	myfile.open("STEP1i.csv");
	 string name;
	int max_z_planes=40;
	vector<vector<Point> > blue_contours;
	vector<Vec4i> bhierarchy;
	vector<vector<Point> > intersect_contours;
	vector<vector<Point> > mod_intersect_contours;
	vector<Vec4i> ahierarchy;
	Mat bthr, src_grayG, src_grayB, gth; Mat added; Mat intersection;
	using namespace cv;
	vector<vector<Point> > tcontours;
	vector<Vec4i> thierarchy;
	vector<Mat> stackimr; vector<Mat> stackimg; vector<Mat> stackimb;
	myfile << "Image name"<< "," << "Total No of Dendrites" << "," << "No.of Small width Dendrites" << "," << " No.of med width Dendrites" << "," << "No.of large width Dendrites" << "," << "No.of Small length Dendrites" << "," << "No.of Med length Dendrites" << "," << "No.of Large length Dendrites" << "," << "Avg len of all Dendrites" << "," << "Avg len of small length Dendrites" << "," << "Avg len of med length Dendrites" << "," << "Avg len of large length Dendrites" << "," << "Avg wdth of all Dendrites" << "," << "Avg wdth of small width Dendrites" << "," << "Avg wdth of med width Dendrites" << "," << "Avg wdth of large width Dendrites"<< "," << "Total number of low intensity synapse" <<"," << "Total number of med intensity synapses" << "," << "Total number of high intensity synapses" <<","<< "Avg Low Int Syn count arnd. Low width" << "," << "Avg Med Int Syn count arnd. Low width" << "," << "Avg Low Int Syn count arnd. High width" << "," << "Avg Low Int Syn count arnd. Med width" << "," << " Avg Med Int Syn count arnd. Med width" << "," << "Avg High Int Syn count arnd. Med width" << "," << "Avg Low Int Syn count arnd. Large width " << "," << "Avg Med Int Syn count arnd. Large width" << "," << "Avg High Int Syn count arnd. Large width"<< "," << "Blue contour count" <<"," << "Soma count" << "," << "Avg Soma diameter" << "," << "Soma diameter std.dev" << "," << "Avg soma circularity" << "," << "Std dev of soma circularity" << "," << "Astrocyte count" << "," << "Count of low circ.astrocytes" << "," << "Count of high circ.astrocytes" << "," << "Other cells count" << "," << "Count of low circ.Other cells" << "," << "Count of high circ.Other cells" << "," << "Avg Count of high int synapse around soma" << "," << "Avg Count of med int synapse around soma" << "," << "Avg Count of low int synapse around   soma" << "," << "Avg Count of high int synapse around astrocytes" << "," << "Avg Count of med int synapse around astrocyte" << "," << "Avg Count of low int synapse around astrocyte" << "," << "Avg Count of high int synapse around  other cells" << "," << "Avg Count of med int synapse around  other cells" << "," << "Avg Count of low int synapse around  other cells" <<endl;
	// Read image

		stackimr.clear();
		stackimg.clear();
		stackimb.clear();
		Mat resultb, resultg, resultr;
		// Read image (single image, different z-stack layers)
	

		for (int n = 1; n <= max_z_planes; n++)//differnt z scales (variable)
		{
			
				name = format("%s//z%d_3layers_original.tif", argv[1], n);
				cout << name<<endl;
		

			Mat imm = imread(name);
			if (imm.empty()) //if image is empty- read next line of txt file- move on to the next folder-we did not find a particular z layer (process with stack created till now)
			{
				cout << "No more images found" << endl;  goto POP;
			}
			//string result = format("C:\\Users\\VIneeta\\Documents\\Visual Studio 2013\\Projects\\OpenCV-Test\\OpenCV-Test\\Old STEP1i\\%s\\z%d_3layers_processed.tif", fstr.c_str(), n);
			string result = format("%s_processed.png", name.c_str());

			//MIP part
			Mat bgr[3];   //destination array
			split(imm, bgr);//split source 

			Mat result_blue(bgr[0].rows, bgr[0].cols, CV_8UC3); // notice the 3 channels here!
			Mat result_green(bgr[1].rows, bgr[1].cols, CV_8UC3); // notice the 3 channels here!
			Mat result_red(bgr[2].rows, bgr[2].cols, CV_8UC3); // notice the 3 channels here!
			Mat empty_image = Mat::zeros(bgr[0].rows, bgr[0].cols, CV_8UC1);

			Mat in1[] = { bgr[0], empty_image, empty_image };
			int from_to1[] = { 0, 0, 1, 1, 2, 2 };
			mixChannels(in1, 3, &result_blue, 1, from_to1, 3);
			stackimb.push_back(result_blue);


			Mat in2[] = { empty_image, bgr[1], empty_image };
			int from_to2[] = { 0, 0, 1, 1, 2, 2 };
			mixChannels(in2, 3, &result_green, 1, from_to2, 3);
			stackimg.push_back(result_green);


			Mat in3[] = { empty_image, empty_image, bgr[2] };
			int from_to3[] = { 0, 0, 1, 1, 2, 2 };
			mixChannels(in3, 3, &result_red, 1, from_to3, 3);
			stackimr.push_back(result_red);

		}
POP:
		resultb = mip(stackimb, 0);
		resultg = mip(stackimg, 1);
		resultr = mip(stackimr, 2);
		resultr = changeimg(resultr, 25, 5);//final MIP, enhace images (resultr=red, resultb=blue, resultg=green)
		resultg = changeimg(resultg, 10, 5);
		//	imshow(" mip",result);
		string stringb = format("%s_blue_MIP.png", argv[1]);
		string stringg = format("%s_green_MIP.png", argv[1]);
		string stringr = format("%s_red_MIP.png", argv[1]);
		imwrite(stringb, resultb);
		imwrite(stringg, resultg);
		imwrite(stringr, resultr);

		string imagnam = format("%s", name.c_str());
		myfile << argv[1];
		cvtColor(resultg, src_grayG, CV_BGR2GRAY);
		/*~~~~~~~*/ synapse(resultr, myfile);// detects low, medium and high intensity contours-change function to just count no of pixels (countNonzero) rather than 
		//detecting dendrites
		/*~~~~~~~*/filterHessian(argv[1], resultg, myfile);

		// thresholding bright green and yellow (soma tails/dendrites)
		cv::threshold(src_grayG, src_grayG, 50, 255, cv::THRESH_TOZERO);
		bitwise_not(src_grayG, src_grayG);
		cv::GaussianBlur(src_grayG, gth, cv::Size(3, 3), 0, 0);
		cv::threshold(gth, gth, 200, 255, cv::THRESH_BINARY);
		bitwise_not(gth, gth);
		dilate(gth, gth, Mat());
		//imwrite("z_gth.png", gth);



		//thresholding nuclei (blue) in the image
		cv::inRange(resultb, cv::Scalar(40, 0, 0), cv::Scalar(255, 10, 10), bthr);//BLUE THRESH
		dilate(bthr, bthr, Mat());
		dilate(bthr, bthr, Mat());
		//imwrite("z_bth.png", bthr);
		bitwise_and(gth, bthr, intersection);//intersection of nuclei with dendrite
		findContours(bthr, blue_contours, bhierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the blue contours in the image
		//dilate(intersection, intersection, Mat());
		//imwrite("z_int.png", intersection);


		findContours(intersection, intersect_contours, ahierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); // Find the contours in the intersection image
		/*~~~~~~~*/blue_contours = watershedcontours(resultb, src_grayB, bthr);
		/*~~~~~~~*/ findsoma(resultb, resultr, blue_contours, intersect_contours, myfile, argv[1]);
		//-------------------------------------------------------------------------------------------------------------------------------------------   
		//imwrite(result, imm);
		myfile << endl;
	

	myfile.close();
	waitKey(0);
}
