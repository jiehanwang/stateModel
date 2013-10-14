/*************************************************
Copyright:   HCI, VIPL, ICT
Author:      Mingquan Ye & Hanjie Wang
Date:        2013-04-26
Description: Read original key frame pictures from P51,P52,P53,P54. 
             Output separated or fused galleries using HOG feature.
			 Save the separated or fused key frames. In "D:\iData\Kinect sign data\Test\20130529\keyFrame(or keyFrame_combine)"
**************************************************/
#include "stdafx.h"
#include<iostream>
#include<fstream>
#include<vector>
#include<algorithm>
#include<string>
#include <opencv2\opencv.hpp>
#include<atlstr.h>
#include<cmath>
#include <direct.h>
#include <iomanip>
using namespace std;
using namespace cv;

#define ReadFromDat  //Read data from five saved gallery files. There must be the files of course.

#ifndef ReadFromDat
	#define OutPutImages
#endif

#define GalleryNum  5
#define SIZE 64
#define Word_num 370         //Posture number
#define LRB 3                   //Left, right, both
#define FusedLRB 1
#define HOG_dimension 324//1764//720
#define MaxCombine 100
#define MarryThre 0.2
#define MaxKeyNo 25
#define maxClassNum 500
#define ep 0.00001

struct Tra
{
	bool exist;
	int signID;
	int frameNum;
	int hx;   //For head
	int hy;
	int hz;
	int segNum;
	vector<int> segFrameID;
	vector<int> lx; //For left hand
	vector<int> ly;
	vector<int> lz;
	vector<int> rx; //For right hand
	vector<int> ry;
	vector<int> rz;
};

struct Pair
{
	int man;
	int woman;
	double love;
	int married;  //1: married. 0: unmarried. 2: may be
};

struct ForSaveImage
{
	int galleryIndex;
	int index;
};

struct State
{
	int r;             //Indicator
	int l;
	int b;
	int R;             //Hand posture label
	int L;
	int B;
	CvPoint3D32f PL;   //Hand position
	CvPoint3D32f PR;  
	int TL;            //Hand trajectory label
	int TR;
	double frequency;  //Frequency of this state in the combined gallery.
// 	int previous;      //-1 is the start.
// 	int next;          //-2 is the end. 
};

CString Add="\\*.*";
ifstream infile;
IplImage* choose_pic[10000];    //To choose the best picture

	//Separate samples. 5 for each word now.
//Posture
vector<IplImage*> Route[GalleryNum][Word_num][LRB];   //Picture in file folder p50
vector<double> HOG[GalleryNum][Word_num][LRB][MaxKeyNo];    //HOG feature for each key frame. 3 channels are merged into 1.
vector<double> HOG_LRB[GalleryNum][Word_num][LRB][MaxKeyNo];    //HOG label for each key frame, 3 channels.
int            indicator[GalleryNum][Word_num][MaxKeyNo][LRB];//Indicate whether left, right, both images are existing. 
int            label[GalleryNum][Word_num][MaxKeyNo][LRB];    //L, R, B's class label in each state.
//State
int            keyFrameNo[GalleryNum][Word_num];     //All the three channels have the same key frame number.
State          myState[GalleryNum][Word_num][MaxKeyNo];         //This is the state before gallery generating.  
                                                                  									  //A black image will take place of no existing image.

	//Posture cluster result. Input.
int            classNum[LRB];  //The class number of left, right, both postures. 
float          postureC[LRB][maxClassNum][HOG_dimension];
float          postureMatrix[LRB][maxClassNum][maxClassNum];



	//Combine
//Posture
vector<double> HOG_final[Word_num][LRB][MaxCombine];
int           indicator_final[Word_num][MaxCombine][LRB];//Indicate whether left, right, both images are existing. 
int           label_final[Word_num][MaxCombine][LRB];    //L, R, B's class label in each state.
//State
int           keyFrameNo_final[Word_num];
State         myState_final[Word_num][MaxCombine];         //This is the state before gallery generating.
double        tranfer_final[Word_num][MaxCombine+2][MaxCombine+2];  //A very sparse matrix for recording the transfer of state in each word.
															         //+2 means the transfer includes start and end.
//
ForSaveImage  myForSaveImage[Word_num][LRB][MaxCombine];
int           isCombined[Word_num][LRB][MaxCombine]; //1: yes; 0: no.

	//Output files.
ofstream outfileData;
ofstream outfileLabel;
ofstream outfileData_csv;
ofstream outfile_LRBLabel;

//////////////////////////////////////////////////////////////////////////
//Function definition.
double states_similar(State myState1, State myState2, 
	float postureMatrix[][maxClassNum][maxClassNum]);
//////////////////////////////////////////////////////////////////////////

void readstr(FILE *f,char *string)
{
	do
	{
		fgets(string, HOG_dimension*10, f);
	} while ((string[0] == '/') || (string[0] == '\n'));
	return;
}
void readstr2(FILE *f,char *string)
{
	do
	{
		fgets(string, maxClassNum*10, f);
	} while ((string[0] == '/') || (string[0] == '\n'));
	return;
}
void readInPostureC(CString route, int lrb)
{
	FILE *filein;
	char oneline[HOG_dimension*10];
	int itemNumber;
	filein = fopen(route, "rt");

	readstr(filein,oneline);
	sscanf(oneline, "NUMBER %d\n", &itemNumber);
	classNum[lrb] = itemNumber;
	for (int loop = 0; loop < itemNumber; loop++)
	{
		readstr(filein,oneline);
		char* sp = oneline; 
		float num; 
		int read; 
		int dimensionIndex = 0;
		while( sscanf(sp, "%f %n", &num, &read)!=EOF )
		{ 
			postureC[lrb][loop][dimensionIndex++] = num;
			sp += read-1; 
		} 
	}
	fclose(filein);
}

void readInPostureMatrix(CString route, int lrb)
{
	FILE *filein;
	char oneline[maxClassNum*10];
	int itemNumber;
	filein = fopen(route, "rt");

	itemNumber = classNum[lrb];
	for (int loop = 0; loop < itemNumber; loop++)
	{
		readstr2(filein,oneline);
		char* sp = oneline; 
		float num; 
		int read; 
		int classIndex = 0;
		while( sscanf(sp, "%f %n", &num, &read)!=EOF )
		{ 
			postureMatrix[lrb][loop][classIndex++] = num;
			sp += read-1; 
		} 
	}
	fclose(filein);
	//delete[] oneline;
}

double img_distance(IplImage *dst1,IplImage *dst2)//Return Euclid distance of two images.
{
	int i,j;
	uchar *ptr1;
	uchar *ptr2;

	double result=0.0;////////////
	for(i=0;i<dst1->height;i++)
	{
		ptr1=(uchar *)(dst1->imageData+i*dst1->widthStep);
		ptr2=(uchar *)(dst2->imageData+i*dst2->widthStep);

		for(j=0;j<dst1->width;j++)
			result+=(ptr1[j*dst1->nChannels]-ptr2[j*dst2->nChannels])*(ptr1[j*dst1->nChannels]-ptr2[j*dst2->nChannels]);
	}
	result=sqrt(result);
	return result;
}

IplImage *Resize(IplImage *_img)//Resize in OpenCV
{
	IplImage *_dst=cvCreateImage(cvSize(SIZE,SIZE),_img->depth,_img->nChannels);
	cvResize(_img,_dst);
	return _dst;
}

void TraverseAllRoute(CString BaseRoute,vector<IplImage *> Route[Word_num][LRB])//存储要处理的文件（图片）的路径
{
	WIN32_FIND_DATA FileData;
	HANDLE handle=FindFirstFile(BaseRoute+Add,&FileData);

	if(handle==INVALID_HANDLE_VALUE)
	{
		//cout<<"访问文件失败!"<<endl;
		//exit(0);
		return ;
	}
	CString temp;
	int i,j,k;//
	int m,n;//i,j,k,m,n,mn均为循环变量
	int Sec;//
	int Lindex,Rindex;//分别表示手势序号及其对应的左手、右手和双手
	int Count_keyframe;//关键帧数
	int a,b;//每个关键帧中所包括的图片序列起始和结束的编号
	char s[10];//临时字符串数组，用于使用函数itoa
	IplImage *T_img;//
	IplImage *T_avg=cvCreateImage(cvSize(SIZE,SIZE),8,1);//存放均值图像
	uchar *pp;//
	uchar *qq;//
	int Sum[105][105];//存储图片灰度求和的结果

	int Image_size;/////////////////////////////////
	double Image_distance;/////////////////////////
	double Image_temp;///////////////
	int Image_index;////////////////////////////

	while( FindNextFile(handle,&FileData) )
	{
		temp=FileData.cFileName;
		if( strcmp( temp.Right(3),"txt" )==0 )//查找txt文件
		{
			Sec=0;
			for(i=0;i<BaseRoute.GetLength();i++)
			{
				if( BaseRoute[i]=='_' )//
				{
					Sec++;
					if(Sec==1)//获取是第几个手势(手势编号)
					{
						Lindex=(BaseRoute[i+1]-48)*1000 + (BaseRoute[i+2]-48)*100 + (BaseRoute[i+3]-48)*10 + (BaseRoute[i+4]-48);
						break;
					}
				}
			}
			if( temp[0]=='l' )//左手LeftHand
				Rindex=0;
			else if( temp[0]=='r' )//右手RightHand
				Rindex=1;
			else 
				Rindex=2;//双手Both

			infile.open(BaseRoute+"\\"+temp,ios::in);//打开搜到的txt文档
			infile>>Count_keyframe;//输入关键帧数
			CString Temp;//

			for(i=0;i<Count_keyframe;i++)
			{
				Image_size=0;
				memset(Sum,0,sizeof(Sum));
				Image_distance=1.0*0xffffff;

				infile>>a>>b;
				for(j=a;j<=b;j++)
				{
						//Changed by Hanjie Wang.
// 					if( j>0 && j<10 )
// 						Temp=BaseRoute+"\\000"+itoa( j , s , 10 )+".jpg";
// 					else if( j>=10 && j<100 )
// 						Temp=BaseRoute+"\\00"+itoa( j , s , 10 )+".jpg";
// 					else if( j>=100 )
// 						Temp=BaseRoute+"\\0"+itoa( j , s , 10 )+".jpg";

					Temp=BaseRoute+"\\"+itoa( j , s , 10 )+".jpg";
					//IplImage *T_img;
					T_img=cvLoadImage(Temp,0);
					if (T_img == NULL)
					{
						T_img=cvLoadImage("black.jpg",0);
					}
					if(T_img!=NULL)//存在这幅图像
					{
						T_img=Resize(T_img);//在这里进行size的归一化
						cvSmooth(T_img,T_img,CV_GAUSSIAN,5,3);//平滑处理，消除噪声
						for(m=0 ; m<T_img->height ; m++)
						{
							pp=(uchar *)(T_img->imageData+m*T_img->widthStep);
							for(n=0;n<T_img->width;n++)
							{
								Sum[m][n]+=pp[n*T_img->nChannels];
							}
						}
						choose_pic[Image_size++]=T_img;
					}
					//cvReleaseImage(&T_img);
				}

				if(Image_size==0)
				{
					//outfile_LRBLabel<<0<<'\t';
				}
				else
				{
					for(m=0;m<SIZE;m++)
					{
						qq=(uchar *)(T_avg->imageData+m*T_avg->widthStep);
						for(n=0;n<SIZE;n++)
						{
							Sum[m][n]=Sum[m][n]/Image_size;
							qq[n*T_avg->nChannels]=Sum[m][n];
						}
					}
					Image_index = 0;
					for(k=0;k<Image_size;k++)
					{
						bool black = true;
						for(m=0 ; m<choose_pic[k]->height ; m++)
						{ 
							bool inBreak =  false;
							pp=(uchar *)(choose_pic[k]->imageData+m*choose_pic[k]->widthStep);
							for(n=0;n<choose_pic[k]->width;n++)
							{
								if (pp[n]>0)
								{
									black = false;
									inBreak = true;
									break;
								}
							}
							if (inBreak)
							{
								break;
							}
						}

						if (!black)
						{
							Image_temp=img_distance(choose_pic[k],T_avg);
							if(Image_temp<Image_distance)
							{
								Image_distance=Image_temp;
								Image_index=k;
							}
						}

						
					}
					Route[Lindex][Rindex].push_back(choose_pic[Image_index]);
					//outfile_LRBLabel<<1<<'\t';
				}
			}
			infile.close();
		}
		if( strcmp(temp,"..") )
		{
			TraverseAllRoute(BaseRoute+"\\"+temp,Route);
		}
	}
	//outfile_LRBLabel<<endl;
}

bool NotAllBlack(int posture,int lrb, int keyframe_count, vector<IplImage*>Route_0[][LRB])
{
	bool notall0 = false;
	for (int m=0;m<SIZE;m++)
	{ 
		bool temp = false;
		//uchar* src_ptr = (uchar*)(LRImage->imageData + m*LRImage->widthStep);
		uchar* src = (uchar*)(Route_0[posture][lrb][keyframe_count]->imageData + m*Route_0[posture][lrb][keyframe_count]->widthStep);
		//uchar* src_right = (uchar*)(Route_0[posture][1][keyframe_count]->imageData + m*Route_0[posture][1][keyframe_count]->widthStep);
		for (int n=0;n<SIZE;n++)
		{
			if (src[n] >0)
			{
				notall0 = true;
				temp = true;
				break;
			}
		}
		if (temp)
		{
			break;
		}
	}

	return notall0;
}

bool GetHOGHistogram_Patch(IplImage *img,vector<double> &hog_hist)//取得图像img的HOG特征向量
{
	//HOGDescriptor *hog=new HOGDescriptor(cvSize(SIZE,SIZE),cvSize(8,8),cvSize(4,4),cvSize(4,4),9);
	//HOGDescriptor *hog=new HOGDescriptor(cvSize(SIZE,SIZE),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
	HOGDescriptor *hog=new HOGDescriptor(cvSize(SIZE,SIZE),cvSize(32,32),cvSize(16,16),cvSize(16,16),9);
	/////////////////////window大小为64*64，block大小为8*8，block步长为4*4，cell大小为4*4
	Mat handMat(img);

	vector<float> *descriptors = new std::vector<float>();

	hog->compute(handMat, *descriptors,Size(0,0), Size(0,0));

	////////////////////window步长为0*0
	double total=0;
	int i;
	for(i=0;i<descriptors->size();i++)
		total+=abs((*descriptors)[i]);
	//	total=sqrt(total);
	for(i=0;i<descriptors->size();i++)
		hog_hist.push_back((*descriptors)[i]/total);
	return true; 
}

	//Measuring two HOG features.
double Histogram(vector<double>vec1,vector<double>vec2)
{
	double mat_score=0.0;
	int i;
	int _Size=vec1.size();
	for(i=0;i<_Size;i++)
	{
		mat_score+=vec1[i]<vec2[i] ? vec1[i] : vec2[i];
	}
	return  mat_score;
}

double Histogram_minD(vector<double>vec1,vector<double>vec2)
{
	double mat_score=0.0;
	int i;
	int _Size=vec1.size();
	for(i=0;i<_Size;i++)
	{
		mat_score+=(vec1[i]-vec2[i])*(vec1[i]-vec2[i]);
	}
	return  sqrt(mat_score);
}

/*************************************************
Function:       saveKeyFrame_combine
Description:    Save key frames. 
Calls:          None
Input:          1. posture: the index of posture. [0-369]
				2. gallery: the index of gallery. [0-5]
				3. lrb: left, right or both hand. [0-3]
				4. indexNow: the current index for output as in fused gallery.
				5. indexOri: the original index in separated gallery.
				6. myImage: the image used for saving.
Output:         1. Images saved in file folder "s_filefolder"
Return:         void
Others:         If "ReadFromDat" is not defined, this function will work.
*************************************************/
void saveKeyFrame_combine(int posture, int gallery, int lrb, int indexNow, int indexOri, IplImage* myImage)
{
#ifdef OutPutImages
	int i,j,k;
	CString s_filefolder;
	CString s_ImgFileName;
	s_filefolder.Format("..\\output\\keyFrame_combine_LRToOne");
	_mkdir(s_filefolder);
	s_filefolder.Format("..\\output\\keyFrame_combine_LRToOne\\%d",posture);
	_mkdir(s_filefolder);

	// 	gallery = 0;
	// 	indexOri = 0;

	if (lrb==0)
	{
		s_ImgFileName.Format("..\\output\\keyFrame_combine_LRToOne\\%d\\LRB_%d_G%d_%d.jpg",posture,indexNow,gallery,indexOri);
	}
	else if (lrb==1)
	{
		s_ImgFileName.Format("..\\output\\keyFrame_combine_LRToOne\\%d\\right_%d_G%d_%d.jpg",posture,indexNow,gallery,indexOri);
	}
	else if (lrb==2)
	{
		s_ImgFileName.Format("..\\output\\keyFrame_combine_LRToOne\\%d\\both_%d_G%d_%d.jpg",posture,indexNow,gallery,indexOri);
	}

// 	IplImage* wideImage = cvCreateImage(cvSize(SIZE*2,SIZE),8,1);
// 	cvResize(myImage, wideImage);

	cvSaveImage(s_ImgFileName, myImage);
#endif
}

void saveKeyFrame_separate( int filefolder,int wordIndex,int keyFrameIndex,  IplImage* Route)
{
	CString s_filefolder;
	CString s_ImgFileName;
	s_filefolder.Format("..\\output\\keyFrame_LRToOne");
	_mkdir(s_filefolder);//Create the folder.
	s_filefolder.Format("..\\output\\keyFrame_LRToOne\\P%d",filefolder);
	_mkdir(s_filefolder);//Create the folder.
	s_filefolder.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d",filefolder,wordIndex);
	_mkdir(s_filefolder);//Create the folder.

	s_ImgFileName.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d\\LRB_%d.jpg",filefolder,wordIndex,keyFrameIndex);
// 	IplImage* wideImage = cvCreateImage(cvSize(SIZE*2,SIZE),8,1);
// 	cvResize(Route, wideImage);
	cvSaveImage(s_ImgFileName, Route);

// 	for (i=0; i<posture_num; i++)
// 	{
// 		for(j=0;j<FusedLRB;j++)
// 		{
// 			for(k=0;k<Route[i][j].size();k++)
// 			{
// 				if (Route[i][j][k]!=NULL)
// 				{
// 					s_filefolder.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d",dataIndex,i);
// 					_mkdir(s_filefolder);//Create the folder.
// 					if (j==0)
// 					{
// 						s_ImgFileName.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d\\left_%d.jpg",dataIndex,i,k);
// 					}
// 					else if (j==1)
// 					{
// 						s_ImgFileName.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d\\right_%d.jpg",dataIndex,i,k);
// 					}
// 					else if (j==2)
// 					{
// 						s_ImgFileName.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d\\both_%d.jpg",dataIndex,i,k);
// 					}
// 
// 					cvSaveImage(s_ImgFileName, Route[i][j][k]);
// 				}
// 
// 			}
// 		}
// 	}

}

void saveKeyFrame_LRB(int filefolder,int wordIndex,int keyFrameIndex,  IplImage* Route, int lrb)
{
	CString s_filefolder;
	CString s_ImgFileName;
	s_filefolder.Format("..\\output\\keyFrame_LRB");
	_mkdir(s_filefolder);//Create the folder.
	s_filefolder.Format("..\\output\\keyFrame_LRB\\P%d",filefolder);
	_mkdir(s_filefolder);//Create the folder.
	s_filefolder.Format("..\\output\\keyFrame_LRB\\P%d\\%d",filefolder,wordIndex);
	_mkdir(s_filefolder);//Create the folder.

	if (lrb==0)
	{
		s_ImgFileName.Format("..\\output\\keyFrame_LRB\\P%d\\%d\\%d_left.jpg",filefolder,wordIndex,keyFrameIndex);
	}
	else if (lrb==1)
	{
		s_ImgFileName.Format("..\\output\\keyFrame_LRB\\P%d\\%d\\%d_right.jpg",filefolder,wordIndex,keyFrameIndex);
	}
	else if (lrb==2)
	{
		s_ImgFileName.Format("..\\output\\keyFrame_LRB\\P%d\\%d\\%d_both.jpg",filefolder,wordIndex,keyFrameIndex);
	}

	cvSaveImage(s_ImgFileName, Route);

// 	for (i=0; i<posture_num; i++)
// 	{
// 		for(j=0;j<FusedLRB;j++)
// 		{
// 			for(k=0;k<Route[i][j].size();k++)
// 			{
// 				if (Route[i][j][k]!=NULL)
// 				{
// 					s_filefolder.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d",dataIndex,i);
// 					_mkdir(s_filefolder);//Create the folder.
// 					if (j==0)
// 					{
// 						s_ImgFileName.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d\\left_%d.jpg",dataIndex,i,k);
// 					}
// 					else if (j==1)
// 					{
// 						s_ImgFileName.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d\\right_%d.jpg",dataIndex,i,k);
// 					}
// 					else if (j==2)
// 					{
// 						s_ImgFileName.Format("..\\output\\keyFrame_LRToOne\\P%d\\%d\\both_%d.jpg",dataIndex,i,k);
// 					}
// 
// 					cvSaveImage(s_ImgFileName, Route[i][j][k]);
// 				}
// 
// 			}
// 		}
// 	}
}

/*************************************************
Function:       dataTofeature
Description:    Read data and compute the HoG feature separately.
Calls:          1. TraverseAllRoute
				2. GetHOGHistogram_Patch
Input:          1. Testdata: folder name. 
				2. posture_num: the number of postures
				3. ikeyFrameNo[Reference]: array of int. The frame number of each gallery.
				4. Route_0[Reference]: array of Iplimage. The key frame images of each gallery.
				5. HOG_0[Reference]: array of vector<double>. The HOG feature each gallery. 
				6. folderIndex: the index of folder, P5X. {50,51,52,53,54}
Output:         1. ikeyFrameNo
				2. Route_0
				3. HOG_0
Return:         void
Others:         ikeyFrameNo, Route_0 and HOG_0 are defined global.
*************************************************/
void dataTofeature(	CString Testdata, 
					int posture_num,
					int ikeyFrameNo[],
					vector<IplImage*>Route_0[][LRB], 
					vector<double>HOG_0[][LRB][MaxKeyNo], 
					vector<double>HOG_LRB[][LRB][MaxKeyNo],
					int folderIndex)
{
	int i,j,k,m,n;
	CString folderName;
	//folderName.Format(folderIndex);
	folderName.Format("%d", folderIndex);
	int keyframe_count;
	//CString::Format(folderName,folderIndex);
	TraverseAllRoute("D:\\iData\\Kinect sign data\\Test\\"+Testdata+"\\P"+folderName,Route_0);
	cout<<"("<<folderIndex<<")Route searching of file p"<<folderIndex<< " is finished."<<endl;
	for(i=0;i<posture_num;i++)
	{
		keyframe_count = Route_0[i][0].size(); //Since all the three channel has the same size. 
		ikeyFrameNo[i] = keyframe_count;
		//outfileLabel.write( (char *)&keyframe_count,sizeof(keyframe_count));
		//cout<<"P"<<folderIndex<<" sentence: "<<i<<" key frame number: "<<keyframe_count<<endl;
		for(k=0;k<keyframe_count;k++)
		{
			IplImage* LRImage = cvCreateImage(cvSize(SIZE*2,SIZE),8,1);
			for (m=0;m<SIZE;m++)
			{ 
				uchar* src_left = (uchar*)(Route_0[i][0][k]->imageData + m*Route_0[i][0][k]->widthStep);
				uchar* src_right = (uchar*)(Route_0[i][1][k]->imageData + m*Route_0[i][1][k]->widthStep);
				uchar* src_LR = (uchar*)(LRImage->imageData + m*LRImage->widthStep);
				for (n=0;n<SIZE;n++)
				{
					src_LR[n] = src_left[n];
				}
				for (n=SIZE;n<2*SIZE;n++)
				{
					src_LR[n] = src_right[n-SIZE];
				}

			}
			LRImage = Resize(LRImage);

			bool notAllBlack = false;
			bool notAllBlack0 = false;
			bool notAllBlack1 = false;
			notAllBlack0 = NotAllBlack(i,0,k,Route_0); //If the image are black, return false. Otherwise, true.
			notAllBlack1 = NotAllBlack(i,1,k,Route_0);
			notAllBlack = notAllBlack0 + notAllBlack1;
			int g = folderIndex%50;
			indicator[g][i][k][0] = notAllBlack0;
			indicator[g][i][k][1] = notAllBlack1;
			indicator[g][i][k][2] = !notAllBlack;
			outfile_LRBLabel<<folderIndex<<'\t'<<i<<'\t'<<k<<'\t'<<
				indicator[g][i][k][0]<<'\t'<<indicator[g][i][k][1]<<'\t'<<indicator[g][i][k][2]<<endl;


			if (!notAllBlack)
			{
				GetHOGHistogram_Patch(Route_0[i][2][k],HOG_0[i][0][k]);
				GetHOGHistogram_Patch(Route_0[i][2][k],HOG_LRB[i][2][k]);

				saveKeyFrame_separate(folderIndex,i,k,Route_0[i][2][k]);
				saveKeyFrame_LRB(folderIndex,i,k,Route_0[i][2][k],2);
				cvCopy(Route_0[i][2][k],Route_0[i][0][k]);  //They are at last stored in the Route_0[i][0][k]
			}
			else
			{
				GetHOGHistogram_Patch(LRImage,HOG_0[i][0][k]);
				if (notAllBlack0)
				{
					GetHOGHistogram_Patch(Route_0[i][0][k],HOG_LRB[i][0][k]);
					saveKeyFrame_LRB(folderIndex,i,k,Route_0[i][0][k],0);
				}
				if (notAllBlack1)
				{
					GetHOGHistogram_Patch(Route_0[i][1][k],HOG_LRB[i][1][k]);
					saveKeyFrame_LRB(folderIndex,i,k,Route_0[i][1][k],1);
				}
				saveKeyFrame_separate(folderIndex,i,k,LRImage);
				cvCopy(LRImage,Route_0[i][0][k]);  //They are at last stored in the Route_0[i][0][k]
			}

//			outfileData.write( (char *)&HOG_0[i][0][k][0],HOG_dimension*sizeof(double) );
// 			for (m=0; m<HOG_dimension; m++)
// 			{
// 				outfileData_csv<<HOG_0[i][0][k][m]<<",";
// 			}
			
			cvReleaseImage(&LRImage);
		}
//		outfileData_csv<<endl;

			// 
		for(int lrb=0;lrb<LRB;lrb++)
		{
			outfileLabel.write( (char *)&keyframe_count,sizeof(keyframe_count) );
			for(k=0;k<keyframe_count;k++)
			{
				int g = folderIndex%50;
				if (indicator[g][i][k][lrb] == 0)
				{
					for (m=0; m<HOG_dimension; m++)
					{
						HOG_LRB[i][lrb][k].push_back(0.0);
					}
				}
				outfileData.write( (char *)&HOG_LRB[i][lrb][k][0],HOG_dimension*sizeof(double) );
				for (m=0; m<HOG_dimension; m++)
				{
					outfileData_csv<<HOG_LRB[i][lrb][k][m]<<",";
				}
				outfileData_csv<<endl;
			}
		}
	}
	cout<<"("<<folderIndex<<")HOG features in file p"<<folderIndex<< " have been computed."<<endl;

// #ifdef OutPutImages
// 	saveKeyFrame_separate(posture_num,folderIndex,Route_0);
// #endif
	
}
	
/*************************************************
Function:       ReadDataFromGallery
Description:    Read separated galleries from saved file. It is fast for debugging.
Calls:          None. 
Input:          1. route: the route of saved gallery file.
				2. Gallery_num: the number of gallery. It is 5 temporally.
				3. ikeyFrameNo[Reference]: array of int. The frame number of all the galleries.
				4. HOG_X[Reference]: array of vector<double>. The HOG feature of all the galleries. 
Output:         1. ikeyFrameNo
				2. HOG_X
Return:         void
Others:         There must be saved files of separated galleries.
				The ikeyFrameNo and HOG_X are defined global.
*************************************************/
void ReadDataFromGallery(	CString route, 
							int Gallery_num, 
							int ikeyFrameNo[][Word_num],
							vector<double> HOG_LRB[][Word_num][LRB][MaxKeyNo],
							int  indicator[][Word_num][MaxKeyNo][LRB])
{
	int i, j, k, galleryIndex, m;
	int* Label_sequence;     //Original label sequence.
	int* Label_sequence_locate;
	double* p_gallery;           //Gallery HOG

	int labelSize = Gallery_num*Word_num*LRB;  //Label_size=5*370*3
	Label_sequence = new int[labelSize];
	Label_sequence_locate = new int[labelSize];

	ifstream infile1;
	infile1.open(route+"\\Gallery_Label.dat",ios::binary);
	infile1.read( (char *)Label_sequence, labelSize*sizeof(int) );//将Gallery_Label中的数据读到数组Label_sequence1中
	infile1.close();

	int keyFrameIntotal = 0;
	*(Label_sequence_locate+0) = *(Label_sequence + 0);
	for(i=0;i<labelSize;i++)
	{
		keyFrameIntotal += *(Label_sequence + i);
		if (i>0)
		{
			*(Label_sequence_locate+i) = *(Label_sequence_locate+i-1) + *(Label_sequence + i);
		}
	}
	cout<<"Label has been read into memory"<<endl;
	int HOGsize=keyFrameIntotal * HOG_dimension;//HOG_size
	p_gallery=new double[HOGsize];                         //p_gallery

	ifstream infile2;
	infile2.open(route+"\\Gallery_Data.dat",ios::binary);
	infile2.read((char*)p_gallery,HOGsize * sizeof(double));
	infile2.close();
	cout<<"Gallery has been read into the memory"<<endl;

	//////////////////////////////////////////////////////////////////////////
	// 	int testFlag[375];
	// 	for (i=0; i<Word_num; i++)
	// 	{
	// 		infileMask>>testFlag[i];
	// 	}

	for (galleryIndex = 0; galleryIndex<Gallery_num; galleryIndex++)
	{
		for(i=0; i<Word_num; i++)                             //Posture
		{
			for(j=0;j<LRB;j++)                                         //Left, right, both
			{
				ikeyFrameNo[galleryIndex][i] = *(Label_sequence + galleryIndex*LRB*Word_num + i*LRB + j);
				int frameLocation;
				if (galleryIndex == 0 && i == 0 && j == 0)
				{
					frameLocation = 0;
				}
				else
				{
					frameLocation = *(Label_sequence_locate + galleryIndex*LRB*Word_num + i*LRB + j-1);
				}

				for(k=0; k<ikeyFrameNo[galleryIndex][i]; k++)            //Key frame
				{
					double sumV = 0.0;
					for (m=0; m<HOG_dimension; m++)
					{
						sumV += *(p_gallery + HOG_dimension*(frameLocation+k) + m);
						HOG_LRB[galleryIndex][i][j][k].push_back(*(p_gallery + HOG_dimension*(frameLocation+k) + m));
					}
					if (sumV == 0)
					{
						indicator[galleryIndex][i][k][j] = 0;
					}
					else
					{
						indicator[galleryIndex][i][k][j] = 1;
					}
				}
			}
		}
		cout<<"Gallery "<<galleryIndex<<" has been read into array"<<endl;
	}



	delete[] p_gallery;
	delete[] Label_sequence;
}

void ReadTrajectoryFromDat(CString route, Tra myTra[])
{
	//////////////////////////////////////////////////////////////////////////
	//Read the head position
	int* headPosition;
	int headPosition_size = Word_num*3; //3 is for the x,y,z ordinate.
	headPosition = new int[headPosition_size];
	ifstream infile_headPosition;
	infile_headPosition.open(route+"\\HeadPosition.dat",ios::binary);
	infile_headPosition.read((char*)headPosition, headPosition_size*sizeof(int));//将Gallery_Label中的数据读到数组Label_sequence1中
	infile_headPosition.close();

	for (int i=0;i<Word_num; i++)
	{
		//cout<<"ID: "<<i<<*(headPosition + 3*i)<<" "<<*(headPosition + 3*i+1)<<" "<<*(headPosition + 3*i+2)<<endl;
		myTra[i].exist = 1;
		myTra[i].hx = *(headPosition + 3*i + 0);
		myTra[i].hy = *(headPosition + 3*i + 1);
		myTra[i].hz = *(headPosition + 3*i + 2);
	}
	delete[] headPosition;
	//////////////////////////////////////////////////////////////////////////
	//Read the label
	int* label;
	int label_size = Word_num * 20;   //At most 20 key frames in each sign.
	label = new int[label_size];
	ifstream infile_label;
	infile_label.open(route+"\\LabelForTra.dat",ios::binary);
	infile_label.read((char*)label,label_size*sizeof(int));

	int pointer_label = 0;
	for (int w=0; w<Word_num; w++)
	{
		myTra[w].signID = *(label + pointer_label + 0);
		myTra[w].frameNum = *(label + pointer_label + 1);
		myTra[w].segNum = *(label + pointer_label + 2);

		//cout<<myTra[w].signID<<" "<<myTra[w].frameNum<<" "<<myTra[w].segNum<<endl;
		for (int s=0; s<myTra[w].segNum; s++)
		{
			int frameID = *(label + pointer_label + 3 + s);
			myTra[w].segFrameID.push_back(frameID);
			//cout<<frameID<<" ";
		}
		//cout<<endl;
		pointer_label += 3+myTra[w].segNum;
	}
	delete[] label;
	//////////////////////////////////////////////////////////////////////////
	//Read the hand positions
	int* trajectory;
	int trajectory_size = Word_num * 6 * 500; //At most 500 frames average for each sign. 
	trajectory = new int[trajectory_size];
	ifstream infile_trajectory;
	infile_trajectory.open(route+"\\Trajectory.dat",ios::binary);
	infile_trajectory.read((char*)trajectory,trajectory_size*sizeof(int));

	int pointer_tra = 0;
	for (int w=0; w<Word_num; w++)
	{
		for (int s=0; s<myTra[w].frameNum; s++)
		{
			int lx = *(trajectory + pointer_tra + 0);
			int ly = *(trajectory + pointer_tra + 1);
			int lz = *(trajectory + pointer_tra + 2);
			int rx = *(trajectory + pointer_tra + 3);
			int ry = *(trajectory + pointer_tra + 4);
			int rz = *(trajectory + pointer_tra + 5);

			myTra[w].lx.push_back(lx);
			myTra[w].ly.push_back(ly);
			myTra[w].lz.push_back(lz);
			myTra[w].rx.push_back(rx);
			myTra[w].ry.push_back(ry);
			myTra[w].rz.push_back(rz);

			pointer_tra += 6;
		}
	}

	delete[] trajectory;
}

void GalleryCombine_new_initial(int keyFrameNo[], State myState[][MaxKeyNo])
{
	for (int w=0; w<Word_num; w++)
	{
		keyFrameNo_final[w] = keyFrameNo[w];
		for (int k=0; k<keyFrameNo_final[w]; k++)
		{
			myState_final[w][k] = myState[w][k];
		}
		for (int k=0; k<MaxCombine; k++)
		{
			if (k<keyFrameNo_final[w])
			{
				myState_final[w][k].frequency = 1.0;
			}
			else
			{
				myState_final[w][k].frequency = 0.0;
			}
			
// 			myState_final[w][k].previous = -1;
// 			myState_final[w][k].next = -2;
		}
		for (int k=0; k<MaxCombine+2; k++)
		{
			for (int l=0; l<MaxCombine+2; l++)
			{
				tranfer_final[w][k][l] = 0;
			}
		}
		for (int k=0; k<keyFrameNo_final[w]+1; k++)
		{
			tranfer_final[w][k][k+1] = 1; //The 0_th state is 1_th now since start is the first state. 
		}
	}

}

void GalleryCombine_new(int keyFrameNo[], State myState[][MaxKeyNo])
{
	//Initial isCombined.
	for(int i=0;i<Word_num;i++)
	{
		for(int j=0;j<LRB;j++)
		{
			for(int k=0;k<MaxCombine;k++)
			{
				isCombined[i][j][k] = false;
			}
		}
	}

	for (int w=0; w<Word_num; w++)
	{
		int finalSize = keyFrameNo_final[w];
		int currentSize = keyFrameNo[w];
		int pairNum = finalSize*currentSize;
		vector<Pair> myPair;

		//Compute the similarities
		for (int m=0; m<finalSize; m++)
		{
			for (int n=0; n<currentSize; n++)
			{
				Pair tempPair;
				tempPair.man = m;
				tempPair.woman = n;
				tempPair.love = states_similar(myState_final[w][m],myState[w][n],postureMatrix);
				tempPair.married = 2;
				myPair.push_back(tempPair);
			}
		}

		int Maybe_num = pairNum; //The potential married pair's number.
		while (Maybe_num > 0)
		{
			//Find the largest love and marry them.
			double max = 0.0;
			int maxIndex = 0;
			for (int k=0; k<pairNum; k++)
			{
				if (myPair[k].married==2 && myPair[k].love >= max)
				{
					max = myPair[k].love;
					maxIndex = k;
				}
			}

			if (myPair[maxIndex].love > MarryThre)
			{
				myPair[maxIndex].married = 1;

				//Unmarried the related others. 
				for (int k=0; k<pairNum; k++)
				{
					if (k != maxIndex)
					{
						bool sad = false;   //If sad is true, they will be unmarried (0).
						if (myPair[k].man == myPair[maxIndex].man)
						{
							sad = true;
						}
						if (myPair[k].woman == myPair[maxIndex].woman)
						{
							sad = true;
						}
						if (myPair[k].man > myPair[maxIndex].man && myPair[k].woman < myPair[maxIndex].woman)
						{
							sad = true;
						}
						if (myPair[k].man < myPair[maxIndex].man && myPair[k].woman > myPair[maxIndex].woman)
						{
							sad = true;
						}
						if (sad)
						{
							myPair[k].married = 0;  //They can not be married.
						}
					}
				}

				Maybe_num = 0;
				for (int k=0; k<pairNum; k++)
				{
					if (myPair[k].married == 2)
					{
						Maybe_num++;
					}
				}
			}
			else
			{
				for (int k=0; k<pairNum; k++)
				{
					if (myPair[k].married == 2)
					{
						myPair[k].married = 0;
					}
				}

				break;
			}
		}

		//Duplicate the keyFrameNo_final and myState_final to temp variables.
		State StateTemp[MaxCombine];
		int   keyFrameNoTemp = keyFrameNo_final[w];
		for (int k=0; k<MaxCombine; k++)
		{
			StateTemp[k] = myState_final[w][k];
		}


		int NoCount = 0;
		for (int k=0; k<pairNum; k++)
		{
			if (myPair[k].married == 1)
			{
				NoCount++;
			}
		}
		NoCount = finalSize + currentSize - NoCount;
		keyFrameNo_final[w] = NoCount;  //The key frame No after combining.
		NoCount = 0;

		int womanPoint = 0;
		int* oldMap;
		int* newMap;
		oldMap = new int[finalSize];    //update the oldMap.
		newMap = new int[currentSize];  //add the new map.
		int tranfer_test[MaxCombine+2][MaxCombine+2];
		for (int k=0; k<MaxCombine+2; k++) //Duplicate the transfer_final memory.
		{
			for (int l=0; l<MaxCombine+2; l++)
			{
				tranfer_test[k][l] = tranfer_final[w][k][l];
				tranfer_final[w][k][l] = 0.0;
			}
		}

		for (int k=0; k<keyFrameNoTemp; k++)
		{
			bool isMarry = false;
			int wife = 0;
			for (int m=0; m<pairNum; m++)
			{
				if (myPair[m].man == k && myPair[m].married == 1)
				{
					isMarry = true;
					wife = myPair[m].woman;
				}

			}
			if (!isMarry)
			{
				myState_final[w][NoCount] = StateTemp[k];
				oldMap[k] = NoCount;
				NoCount++;
			}
			else if (isMarry)
			{
				for (int n=womanPoint; n<wife; n++)
				{
					myState_final[w][NoCount] = myState[w][n];
					myState_final[w][NoCount].frequency += 1.0;
					newMap[n] = NoCount;
					NoCount++;
				}
				womanPoint = wife + 1;

				myState_final[w][NoCount] = StateTemp[k]; 
				myState_final[w][NoCount].frequency += 1.0;
				oldMap[k] = NoCount;
				newMap[wife] = NoCount;
				NoCount++;
			}
		}
// 		for (int k=0; k<keyFrameNoTemp; k++)
// 		{
// 			cout<<oldMap[k]<<endl;
// 		}
		//To process the rest.
		for (int k=womanPoint; k<currentSize; k++)
		{
			myState_final[w][NoCount] = myState[w][k];
			myState_final[w][NoCount].frequency += 1.0;
			newMap[k] = NoCount;
			NoCount++;
		}

		cout<<"For word: "<<w<<endl;
		for (int k=0; k<keyFrameNo_final[w]; k++)
		{
			for (int i=0; i<finalSize; i++)
			{
				if (oldMap[i] == k)
				{
					cout<<"Old: "<<i<<'\t'<<k<<endl;
				}
			}
			for (int j=0; j<currentSize; j++)
			{
				if (newMap[j] == k)
				{
					cout<<"New: "<<j<<'\t'<<k<<endl;
				}
			}
		}

			//Update the old transfer.
		for (int l=0; l<finalSize; l++)
		{
			int ll = oldMap[l];
			tranfer_final[w][0][ll+1] = tranfer_test[0][l+1];
			
		}
		for (int k=0; k<finalSize; k++)
		{
			int kk = oldMap[k];
			tranfer_final[w][kk+1][keyFrameNo_final[w]+1] = tranfer_test[k+1][finalSize+1];
		}

		for (int k=0; k<finalSize; k++)
		{
			for (int l=0; l<finalSize; l++)
			{
				int kk = oldMap[k];
				int ll = oldMap[l];
				tranfer_final[w][kk+1][ll+1] = tranfer_test[k+1][l+1]; 
			}
		}

			//Add new transfer.
		if (currentSize>0)
		{
			tranfer_final[w][0][newMap[0]+1] += 1;
			for (int k=0; k<currentSize-1; k++)
			{
				int kk = newMap[k];
				int ll = newMap[k+1];
				tranfer_final[w][kk+1][ll+1] += 1;

				if (newMap[k] == keyFrameNo_final[w]-1)
				{
					int kk = newMap[k]+1;
					tranfer_final[w][kk][keyFrameNo_final[w]+1] += 1;
				}
			}
			tranfer_final[w][newMap[currentSize-1]+1][keyFrameNo_final[w]+1] += 1;

// 			for (int k=0; k<keyFrameNo_final[w]+2; k++)
// 			{
// 				for (int l=0; l<keyFrameNo_final[w]+2; l++)
// 				{
// 					cout<<tranfer_final[w][k][l]<<" ";
// 				}
// 				cout<<endl;
// 			}
		}
		delete[] oldMap;
		delete[] newMap;
	}
}

void GalleryCombine_new_end()
{
	for (int w=0; w<Word_num; w++)
	{
		int keyFramNum = keyFrameNo_final[w];

		for (int k=0; k<keyFramNum+2; k++ )
		{
			double sumLineTr = 0.0;
			for (int l=0; l<keyFramNum+2; l++)
			{
				sumLineTr +=tranfer_final[w][k][l];
			}
			for (int l=0; l<keyFramNum+2; l++)
			{
				tranfer_final[w][k][l] /= (sumLineTr + ep);
			}
		}
		double sumFre = 0.0;
		for (int k=0; k<keyFramNum; k++)
		{
			sumFre += myState_final[w][k].frequency;
		}
		for (int k=0; k<keyFramNum; k++)
		{
			myState_final[w][k].frequency /= (sumFre + ep);
		}

		for (int k=0; k<keyFramNum+2; k++)
		{
			for (int l=0; l<keyFramNum+2; l++)
			{
				cout<<tranfer_final[w][k][l]<<" ";
			}
			cout<<endl;
		}

	}
}
/*************************************************
Function:       GalleryCombine
Description:    Fuse the galleries into one final gallery.
Calls:          1. Histogram.
				2. saveKeyFrame_combine.
Input:          1. HOG_X:     Array of vector. Data of five separated galleries.
				2. keyFrameNo_X: Array of int. Label of five separated galleries.
Output:         1. HOG_final: Array of vector. Data of fused gallery.
				2. keyFrameNo_final:Array of int. Label of fused gallery.
				3. isCombined: array of int. Indicator of combined gesture.
Return:         void
Others:         3 variants in output are all global.
*************************************************/
void GalleryCombine(vector<double> HOG0[][LRB][25],
					vector<double> HOG1[][LRB][25],
					vector<double> HOG2[][LRB][25],
					vector<double> HOG3[][LRB][25],
					vector<double> HOG4[][LRB][25],
					int keyFrameNo0[],
					int keyFrameNo1[],
					int keyFrameNo2[],
					int keyFrameNo3[],
					int keyFrameNo4[])
{
	int i,j,k,g,m,n;
	int  keyFrameNo[GalleryNum][Word_num];

		//Initial isCombined.
	for(i=0;i<Word_num;i++)
	{
		for(j=0;j<LRB;j++)
		{
			for(k=0;k<MaxCombine;k++)
			{
				isCombined[i][j][k] = false;
			}
		}
	}

	for (i=0; i<GalleryNum; i++)
	{
		for (j=0; j<Word_num; j++)
		{
			for (k=0; k<LRB; k++)
			{
				if (i==0) keyFrameNo[0][j] = keyFrameNo0[j];
				if (i==1) keyFrameNo[1][j] = keyFrameNo1[j];
				if (i==2) keyFrameNo[2][j] = keyFrameNo2[j];
				if (i==3) keyFrameNo[3][j] = keyFrameNo3[j];
				if (i==4) keyFrameNo[4][j] = keyFrameNo4[j];
			}
		}
	}
	
	//Initial of x_final. Input the HOG_0 into it.
	for (i=0; i<Word_num; i++)
	{
		int sizeTemp = keyFrameNo[0][i];
		keyFrameNo_final[i] = sizeTemp;
		for (j=0; j<LRB; j++)
		{
			for (k=0; k<sizeTemp; k++)
			{
				HOG_final[i][j][k] = HOG0[i][j][k];
			}
		}
	}
	//Gallery combination
	for (i=0; i<Word_num; i++)
	{
		cout<<"Posture"<<i<<endl;
		for (j=0; j<LRB; j++)
		{
			for (g=1; g<GalleryNum; g++)
			{
				int finalSize = keyFrameNo_final[i];
				int currentSize = keyFrameNo[g][i];
				int pairNum = finalSize*currentSize; //The total number of key frame pairs
				vector<Pair> myPair;
				for (m=0; m<finalSize; m++)
				{
					for (n=0; n<currentSize; n++)
					{
						Pair tempPair;
						tempPair.man = m;
						tempPair.woman = n;
						if (g==1)
						{
							tempPair.love = Histogram(HOG_final[i][j][m],HOG1[i][j][n]);
						}
						else if (g==2)
						{
							tempPair.love = Histogram(HOG_final[i][j][m],HOG2[i][j][n]);
						}
						else if (g==3)
						{
							tempPair.love = Histogram(HOG_final[i][j][m],HOG3[i][j][n]);
						}
						else if (g==4)
						{
							tempPair.love = Histogram(HOG_final[i][j][m],HOG4[i][j][n]);
						}
						tempPair.married = 2;
						myPair.push_back(tempPair);
					}
				}
				//////////////////////////////////////////////////////////////////////////
				//Find the hog_final from myPair.
				int Maybe_num = 0;
				for (k=0; k<pairNum; k++)
				{
					if (myPair[k].married == 2)
					{
						Maybe_num++;
					}
				}
				//Label the married.
				//int count = 0;
				while (Maybe_num > 0 /*&& count<pairNum*/)
				{
					//count++;
					//Find the largest love and marry them.
					double max = 0.0;
					int maxIndex = 0;
					for (k=0; k<pairNum; k++)
					{
						if (myPair[k].married==2 && myPair[k].love >= max)
						{
							max = myPair[k].love;
							maxIndex = k;
						}
					}
					if (myPair[maxIndex].love > MarryThre)
					{
						myPair[maxIndex].married = 1;

						//Unmarried the related others. 
						for (k=0; k<pairNum; k++)
						{
							if (k!=maxIndex)
							{
								bool sad = false;   //If sad is true, they will be unmarried (0).
								if (myPair[k].man == myPair[maxIndex].man)
								{
									sad = true;
								}
								if (myPair[k].woman == myPair[maxIndex].woman)
								{
									sad = true;
								}
								if (myPair[k].man > myPair[maxIndex].man && myPair[k].woman < myPair[maxIndex].woman)
								{
									sad = true;
								}
								if (myPair[k].man < myPair[maxIndex].man && myPair[k].woman > myPair[maxIndex].woman)
								{
									sad = true;
								}
								if (sad)
								{
									myPair[k].married = 0;  //They can not be married.
								}
							}
						}

						Maybe_num = 0;
						for (k=0; k<pairNum; k++)
						{
							if (myPair[k].married == 2)
							{
								Maybe_num++;
							}
						}
					}
					else
					{
						for (k=0; k<pairNum; k++)
						{
							if (myPair[k].married == 2)
							{
								myPair[k].married = 0;
							}
						}

						break;
					}
				}

				//Combine the gallery.
				vector<double> HOGTemp[MaxCombine];
				int keyFrameNoTemp = keyFrameNo_final[i];
				ForSaveImage forSaveTemp[MaxCombine];
				//bool isCombineTemp[MaxCombine];

				for (k=0; k<MaxCombine; k++)
				{
					HOGTemp[k] = HOG_final[i][j][k];
					//isCombineTemp[k] = isCombined[i][j][k];
					HOG_final[i][j][k].clear();
					isCombined[i][j][k] = 0;
					forSaveTemp[k].galleryIndex = myForSaveImage[i][j][k].galleryIndex;
					forSaveTemp[k].index = myForSaveImage[i][j][k].index;
					myForSaveImage[i][j][k].galleryIndex = -1;
					myForSaveImage[i][j][k].index = -1;
				}

				int NoCount = 0;
				for (k=0; k<pairNum; k++)
				{
					if (myPair[k].married == 1)
					{
						NoCount++;
					}
				}
				NoCount = finalSize + currentSize - NoCount;
				keyFrameNo_final[i] = NoCount;
				NoCount = 0;

				//int manPoint = 0;
				int womanPoint = 0;

				for (k=0; k<keyFrameNoTemp; k++)
				{
					bool isMarry = false;
					int wife = 0;
					for (m=0; m<pairNum; m++)
					{
						if (myPair[m].man == k && myPair[m].married == 1)
						{
							isMarry = true;
							wife = myPair[m].woman;
						}

					}
					if (!isMarry)
					{
						HOG_final[i][j][NoCount] = HOGTemp[k];
						if (g==1)
						{
							myForSaveImage[i][j][NoCount].galleryIndex = 0;
							myForSaveImage[i][j][NoCount].index = k;
						}
						else
						{
							myForSaveImage[i][j][NoCount].galleryIndex = forSaveTemp[k].galleryIndex;
							myForSaveImage[i][j][NoCount].index = forSaveTemp[k].index;
						}
						
						NoCount++;
					}
					else if (isMarry)
					{
						for (n=womanPoint; n<wife; n++)
						{
							if (g==1)
							{
								HOG_final[i][j][NoCount] = HOG1[i][j][n];
								myForSaveImage[i][j][NoCount].galleryIndex = 1;
								myForSaveImage[i][j][NoCount].index = n;
							}
							else if (g==2)
							{
								HOG_final[i][j][NoCount] = HOG2[i][j][n];
								myForSaveImage[i][j][NoCount].galleryIndex = 2;
								myForSaveImage[i][j][NoCount].index = n;
							}
							else if (g==3)
							{
								HOG_final[i][j][NoCount] = HOG3[i][j][n];
								myForSaveImage[i][j][NoCount].galleryIndex = 3;
								myForSaveImage[i][j][NoCount].index = n;
							}
							else if (g==4)
							{
								HOG_final[i][j][NoCount] = HOG4[i][j][n];
								myForSaveImage[i][j][NoCount].galleryIndex = 4;
								myForSaveImage[i][j][NoCount].index = n;
							}
							NoCount++;
						}
						womanPoint = wife + 1;
						//////////////////////////////////////////////////////////////////////////
						//Choose one 
							//without weight
						//HOG_final[i][j][NoCount] = HOGTemp[k];  

							//with weight
						double weight = 0.5;
						for (n=0; n<HOG_dimension; n++)
						{
							double hog = 0.0;
							double hogtemp = 0.0;

							if (g==1) 
							{
								hog = HOG1[i][j][wife][n]*weight;
								hogtemp = HOGTemp[k][n]*(1-weight) + hog;
							}
							else if (g==2) 
							{
								hog = HOG2[i][j][wife][n]*weight;
								hogtemp = HOGTemp[k][n]*(1-weight) + hog;
							}
							else if (g==3) 
							{
								hog = HOG3[i][j][wife][n]*weight;
								hogtemp = HOGTemp[k][n]*(1-weight) + hog;
							}
							else if (g==4) 
							{
								hog = HOG4[i][j][wife][n]*weight;
								hogtemp = HOGTemp[k][n]*(1-weight) + hog;
							}
							HOG_final[i][j][NoCount].push_back(hogtemp);
						}

						isCombined[i][j][NoCount] = 1;
						
						
						//////////////////////////////////////////////////////////////////////////
						if (g==1)
						{
							myForSaveImage[i][j][NoCount].galleryIndex = 0;
							myForSaveImage[i][j][NoCount].index = k;
						}
						else
						{
							myForSaveImage[i][j][NoCount].galleryIndex = forSaveTemp[k].galleryIndex;
							myForSaveImage[i][j][NoCount].index = forSaveTemp[k].index;
						}
						NoCount++;
					}
				}

				for (k=womanPoint; k<currentSize; k++)
				{
					if (g==1)
					{
						HOG_final[i][j][NoCount] = HOG1[i][j][k];
						myForSaveImage[i][j][NoCount].galleryIndex = 1;
						myForSaveImage[i][j][NoCount].index = k;
					}
					else if (g==2)
					{
						HOG_final[i][j][NoCount] = HOG2[i][j][k];
						myForSaveImage[i][j][NoCount].galleryIndex = 2;
						myForSaveImage[i][j][NoCount].index = k;
					}
					else if (g==3)
					{
						HOG_final[i][j][NoCount] = HOG3[i][j][k];
						myForSaveImage[i][j][NoCount].galleryIndex = 3;
						myForSaveImage[i][j][NoCount].index = k;
					}
					else if (g==4)
					{
						HOG_final[i][j][NoCount] = HOG4[i][j][k];
						myForSaveImage[i][j][NoCount].galleryIndex = 4;
						myForSaveImage[i][j][NoCount].index = k;
					}
					NoCount++;

				}
				
			}
		}
	}


	//For outputting images
	int keyFrameTemp;
	for(i=0;i<Word_num;i++)
	{
		for(j=0;j<LRB;j++)
		{
			keyFrameTemp = keyFrameNo_final[i];
			for(k=0;k<keyFrameTemp;k++)
			{
				//int posture, int gallery, int lrb, int indexNow, int indexOri, IplImage* myImage
				int gallery = myForSaveImage[i][j][k].galleryIndex;
				int indexOri = myForSaveImage[i][j][k].index;
				saveKeyFrame_combine(i,gallery,j,k,indexOri,Route[gallery][i][j][indexOri]);
			}
		}
	}

}

void labelPosture(                                      //To get the "label" in this function
	vector<double> HOG_LRB[][LRB][MaxKeyNo], 
	int            label[][MaxKeyNo][LRB],
	int			   classNum[], 
	int            keyFrameNo[], 
	int            indicator[][MaxKeyNo][LRB],
	float          postureC[][maxClassNum][HOG_dimension])
{
	for (int p=0; p<Word_num; p++)
	{
		int keyFrameNum = keyFrameNo[p];
		for (int k=0; k<keyFrameNum; k++)
		{
			for (int lrb=0; lrb<LRB; lrb++)
			{
				int classNum_in = classNum[lrb];
				int maxClass = 0;
				float maxSimilar = 1000.0;
				int indicator_in = indicator[p][k][lrb];
				if (indicator_in == 1)
				{
					for (int c=0; c<classNum_in; c++)
					{
						vector<double> classCenter;
						for (int i=0; i<HOG_dimension; i++)
						{
							classCenter.push_back(postureC[lrb][c][i]);
						}
						//float temp = Histogram(classCenter, HOG_LRB[p][lrb][k]);
						float temp = Histogram_minD(classCenter, HOG_LRB[p][lrb][k]);

						if (maxSimilar > temp)
						{
							maxSimilar = temp;
							maxClass = c;
						}
					}
					label[p][k][lrb] = maxClass;
				}
				else
				{
					label[p][k][lrb] = -1;  //-1 means no images. Actually, it will not be visited under the control of "indicator".
				}
				
			}
		}
	}
}

double states_similar(State myState1, State myState2, 
	float postureMatrix[][maxClassNum][maxClassNum])
{
	double similarity = 0.0;
	if (myState1.l == myState2.l 
		&& myState1.r == myState2.r
		&& myState1.b == myState2.b)
	{
		//if ()   //Position is used here.
		{
			if(myState1.l == 1 && myState1.r == 1)
			{
				int left1 = myState1.L<myState2.L?myState1.L:myState2.L;
				int left2 = myState1.L>myState2.L?myState1.L:myState2.L;

				int right1 = myState1.R<myState2.R?myState1.R:myState2.R;
				int right2 = myState1.R>myState2.R?myState1.R:myState2.R;

				similarity = sqrt(postureMatrix[0][left1][left2]*postureMatrix[1][right1][right2]);
			}
			else if (myState1.l == 1 && myState1.r == 0)
			{
				int left1 = myState1.L<myState2.L?myState1.L:myState2.L;
				int left2 = myState1.L>myState2.L?myState1.L:myState2.L;
				similarity = postureMatrix[0][left1][left2];
			}
			else if (myState1.l == 0 && myState1.r == 1)
			{
				int right1 = myState1.R<myState2.R?myState1.R:myState2.R;
				int right2 = myState1.R>myState2.R?myState1.R:myState2.R;
				similarity = postureMatrix[1][right1][right2];
			}
			else if (myState1.b == 1)
			{
				int both1 = myState1.B<myState2.B?myState1.B:myState2.B;
				int both2 = myState1.B>myState2.B?myState1.B:myState2.B;
				similarity = postureMatrix[2][both1][both2];
			}
		}
	}


	return similarity;

}

void stateGenerate(int keyFrameNo, int label[][LRB],int indicator[][LRB], State myState[])
{
	for (int i=0; i<keyFrameNo; i++)
	{
		myState[i].l = indicator[i][0];
		myState[i].r = indicator[i][1];
		myState[i].b = indicator[i][2];

		myState[i].L = label[i][0];
		myState[i].R = label[i][1];
		myState[i].B = label[i][2];
	}
}


int main()
{
	int i,j,k,g,m,n;
	CString TestdataFolder;
	TestdataFolder = "20130616";
	cout<<"**********Use data in file folder "<<TestdataFolder<<"**********"<<endl;

	readInPostureC("..\\input\\postureC_0.txt",0); //Left posture
	readInPostureC("..\\input\\postureC_1.txt",1); //Right posture
	readInPostureC("..\\input\\postureC_2.txt",2); //Both posture
	readInPostureMatrix("..\\input\\postureMatrix_0.txt",0);
	readInPostureMatrix("..\\input\\postureMatrix_1.txt",1);
	readInPostureMatrix("..\\input\\postureMatrix_2.txt",2);

		//Read the trajectories
	Tra myTra[GalleryNum][Word_num];
	CString root;
	for (int g=0; g<5; g++)
	{
		cout<<"Read trajectory data P5"<<g<<"..."<<endl;
		root.Format("..\\input\\trajectory\\P5%d",g);
		ReadTrajectoryFromDat(root, myTra[g]);
	}

#ifndef ReadFromDat
	outfileData.open("..\\output\\Gallery_Data.dat",ios::binary | ios::out);
	outfileLabel.open("..\\output\\Gallery_Label.dat",ios::binary | ios::out);
	outfileData_csv.open("..\\output\\Gallery_Data.csv",ios::out);
	outfile_LRBLabel.open("..\\output\\LRBLabel.txt",ios::out);

	for (i=0; i<GalleryNum; i++)
	{
		dataTofeature(TestdataFolder,Word_num,keyFrameNo[i],Route[i],HOG[i],HOG_LRB[i],50+i);
		//TestdataFolder and 50+i are used for creating route. 
	}

	outfileData.close();
	outfileLabel.close();
	outfileData_csv.close();
	outfile_LRBLabel.close();

	//Label the posture.
	cout<<"Label the posture..."<<endl;
	for (i=0; i<GalleryNum; i++)
	{
		cout<<"Sample: "<<i<<endl;
		labelPosture(HOG_LRB[i], label[i], classNum, keyFrameNo[i], indicator[i], postureC);
	}

	//Generate the state. Memory "myState[GalleryNum][Word_num][MaxKeyNo]". 
	cout<<"Generate the state..."<<endl;
	for (g=0; g<GalleryNum; g++)
	{
		for (int w=0; w<Word_num; w++)
		{
			stateGenerate(keyFrameNo[g][w],label[g][w],indicator[g][w],myState[g][w]);
		}
	}


	//keyFrameNo_final, myState_final, tranfer_final are the public variances.
	GalleryCombine_new_initial(keyFrameNo[0], myState[0]);
	for (int g=1; g<5; g++)
	{
		GalleryCombine_new(keyFrameNo[g], myState[g]);
	}
	GalleryCombine_new_end();

// 		//Label the posture.
// 	for (i=0; i<GalleryNum; i++)
// 	{
// 		labelPosture(HOG_LRB[i], label[i], classNum, keyFrameNo[i], indicator[i], postureC);
// 	}
// 
// 		//Generate the state. Memory "myState[GalleryNum][Word_num][MaxKeyNo]". 
// 	for (g=0; g<GalleryNum; g++)
// 	{
// 		for (int w=0; w<Word_num; w++)
// 		{
// 			stateGenerate(keyFrameNo[g][w],label[g][w],indicator[g][w],myState[g][w]);
// 		}
// 	}
// 
// 	int totalKeyFrameNum = 0;
// 	for (g=0; g<GalleryNum; g++)
// 	{
// 		for (int w=0; w<Word_num; w++)
// 		{
// 			totalKeyFrameNum += keyFrameNo[g][w];
// 		}
// 	}
// 
// 		//Output the state matrix for clustering in Matlab. 
// 		//The matrix contains the distance between each state pair. 
// 		//However, I think it is a weak idea to cluster states. 
// 	ofstream outfile_stateMatrix;
// 	outfile_stateMatrix.open("..\\output\\stateMatrix.csv",ios::out);
// 	for (g=0; g<GalleryNum; g++)
// 	{
// 		for (int w=0; w<Word_num; w++)
// 		{
// 			for (k=0; k<keyFrameNo[g][w]; k++)
// 			{
// 				for (int g2=0; g2<GalleryNum; g2++)
// 				{
// 					for (int w2=0; w2<Word_num; w2++)
// 					{
// 						for (int k2=0; k2<keyFrameNo[g2][w2]; k2++)
// 						{
// 							//normalize is needed here.
// 							outfile_stateMatrix<<1 - states_similar(myState[g][w][k],myState[g2][w2][k2],postureMatrix)<<",";
// 						}
// 						
// 					}
// 				}
// 				outfile_stateMatrix<<endl;
// 			}
// 			
// 		}
// 	}
// 	outfile_stateMatrix.close();
// 
// 
// 
// 	//The combine step. The parameters can not be changed.
// 	GalleryCombine(HOG[0],HOG[1],HOG[2],HOG[3],HOG[4],
// 		keyFrameNo[0],keyFrameNo[1],keyFrameNo[2],keyFrameNo[3],keyFrameNo[4]);
#endif

#ifdef ReadFromDat
		//For gallery combination. Fast reading.
	CString routeGallery;
	routeGallery="..\\input";
	ReadDataFromGallery(routeGallery, GalleryNum, keyFrameNo,HOG_LRB, indicator);

	//Label the posture.
	cout<<"Label the posture..."<<endl;
	for (i=0; i<GalleryNum; i++)
	{
		cout<<"Sample: "<<i<<endl;
		labelPosture(HOG_LRB[i], label[i], classNum, keyFrameNo[i], indicator[i], postureC);
	}

	//Generate the state. Memory "myState[GalleryNum][Word_num][MaxKeyNo]". 
	cout<<"Generate the state..."<<endl;
	for (g=0; g<GalleryNum; g++)
	{
		for (int w=0; w<Word_num; w++)
		{
			stateGenerate(keyFrameNo[g][w],label[g][w],indicator[g][w],myState[g][w]);
		}
	}


	//keyFrameNo_final, myState_final, tranfer_final are the public variances.
	GalleryCombine_new_initial(keyFrameNo[3], myState[3]);
	for (int g=2; g>=0; g--)
	{
		GalleryCombine_new(keyFrameNo[g], myState[g]);
	}
	GalleryCombine_new_end();


#endif


	//int           keyFrameNo_final[Word_num];
	//State         myState_final[Word_num][MaxCombine];         //This is the state before gallery generating.
	//double        tranfer_final[Word_num][MaxCombine+2][MaxCombine+2]; 

	ofstream outfileKeyFrameNo;
	ofstream outfileMyState;
	ofstream outfileTranfer;
	int keyFrameTemp;
	outfileKeyFrameNo.open("..\\output\\keyFrameNo.dat",ios::binary | ios::out);
	outfileMyState.open("..\\output\\myState.dat",ios::binary | ios::out);
	outfileTranfer.open("..\\output\\tranfer.dat",ios::binary | ios::out);
	for (int w=0; w<Word_num; w++)
	{
		keyFrameTemp = keyFrameNo_final[w];
		outfileKeyFrameNo.write((char*)&keyFrameTemp,sizeof(keyFrameTemp));
		for (int m=0; m<keyFrameTemp; m++)
		{
			float r = (float)myState_final[w][m].r;             
			float l = (float)myState_final[w][m].l;
			float b = (float)myState_final[w][m].b;
			float R = (float)myState_final[w][m].R;       
			float L = (float)myState_final[w][m].L;
			float B = (float)myState_final[w][m].B;
			float PLx = (float)myState_final[w][m].PL.x;  
			float PLy = (float)myState_final[w][m].PL.y;
			float PLz = (float)myState_final[w][m].PL.z;
			float PRx = (float)myState_final[w][m].PR.x;  
			float PRy = (float)myState_final[w][m].PR.y;  
			float PRz = (float)myState_final[w][m].PR.z;  
			float TL = (float)myState_final[w][m].TL;            
			float TR = (float)myState_final[w][m].TR;
			float frequency = (float)myState_final[w][m].frequency;

			outfileMyState.write((char*)&r,sizeof(r));
			outfileMyState.write((char*)&l,sizeof(l));
			outfileMyState.write((char*)&b,sizeof(b));
			outfileMyState.write((char*)&R,sizeof(R));
			outfileMyState.write((char*)&L,sizeof(L));
			outfileMyState.write((char*)&B,sizeof(B));
			outfileMyState.write((char*)&PLx,sizeof(PLx));
			outfileMyState.write((char*)&PLy,sizeof(PLy));
			outfileMyState.write((char*)&PLz,sizeof(PLz));
			outfileMyState.write((char*)&PRx,sizeof(PRx));
			outfileMyState.write((char*)&PRy,sizeof(PRy));
			outfileMyState.write((char*)&PRz,sizeof(PRz));
			outfileMyState.write((char*)&TL,sizeof(TL));
			outfileMyState.write((char*)&TR,sizeof(TR));
			outfileMyState.write((char*)&frequency,sizeof(frequency));
		}
		for (int k=0; k<MaxCombine+2; k++)
		{
			for (int l=0; l<MaxCombine+2; l++)
			{
				double tempTransfer = tranfer_final[w][k][l];
				outfileTranfer.write((char*)&tempTransfer,sizeof(tempTransfer));

			}
		}
	}


	outfileKeyFrameNo.close();
	outfileMyState.close();
	outfileTranfer.close();

		
// 	ofstream outfileDataCombine;
// 	ofstream outfileLabelCombine;
// 	ofstream outfileFlagCombine;
// 	outfileDataCombine.open("..\\output\\Gallery_Data_Combine.dat",ios::binary | ios::out);
// 	outfileLabelCombine.open("..\\output\\Gallery_Label_Combine.dat",ios::binary | ios::out);
// 	outfileFlagCombine.open("..\\output\\Gallery_Flag_Combine.dat",ios::binary | ios::out);
// 	int keyFrameTemp;
// 	for(i=0;i<Word_num;i++)
// 	{
// 		for(j=0;j<FusedLRB;j++)
// 		{
// 			keyFrameTemp = keyFrameNo_final[i];
// 			outfileLabelCombine.write((char*)&keyFrameTemp,sizeof(keyFrameTemp));
// 			for(k=0;k<keyFrameTemp;k++)
// 			{
// 				outfileDataCombine.write((char*)&HOG_final[i][j][k][0],HOG_dimension*sizeof(double));
// 				outfileFlagCombine.write((char*)&isCombined[i][j][k],sizeof(int));
// 			}
// 		}
// 	}
// 	outfileDataCombine.close();
// 	outfileLabelCombine.close();
// 	outfileFlagCombine.close();
	
	cout<<"Done"<<endl;
	getchar();
	return 0;
}
