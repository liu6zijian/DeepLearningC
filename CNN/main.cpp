#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <time.h>
#include <mpi.h>
#include "cnn.h"
#include "minst.h"
#include <ctime>
#include <iostream>
#include <cstdlib>
using namespace std;
clock_t t1,t2;
// ���¶��ǲ��Ժ��������Բ��ù�
// ����Minstģ���Ƿ�������
// void test_minst(){
// 	LabelArr testLabel=read_Lable("E:\\Code\\VS2010 code\\CNN\\Minst\\test-labels.idx1-ubyte");
// 	ImgArr testImg=read_Img("E:\\Code\\VS2010 code\\CNN\\Minst\\test-images.idx3-ubyte");
// 	save_Img(testImg,"E:\\Code\\VS2010 code\\CNN\\Minst\\testImgs\\");
// }
// ����Matģ���Ƿ�������
void test_mat(){
	int i,j;
	nSize srcSize={6,6};
	nSize mapSize={4,4};
	srand((unsigned)time(NULL));
	float** src=(float**)malloc(srcSize.r*sizeof(float*));
	for(i=0;i<srcSize.r;i++){
		src[i]=(float*)malloc(srcSize.c*sizeof(float));
		for(j=0;j<srcSize.c;j++){
			src[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}
	float** map=(float**)malloc(mapSize.r*sizeof(float*));
	for(i=0;i<mapSize.r;i++){
		map[i]=(float*)malloc(mapSize.c*sizeof(float));
		for(j=0;j<mapSize.c;j++){
			map[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}

	nSize cov1size={srcSize.c+mapSize.c-1,srcSize.r+mapSize.r-1};
	float** cov1=cov(map,mapSize,src,srcSize,full);
	//nSize cov2size={srcSize.c,srcSize.r};
	//float** cov2=cov(map,mapSize,src,srcSize,same);
	nSize cov3size={srcSize.c-(mapSize.c-1),srcSize.r-(mapSize.r-1)};
	float** cov3=cov(map,mapSize,src,srcSize,valid);

	savemat(src,srcSize,"E:\\Code\\Matlab\\PicTrans\\src.ma");
	savemat(map,mapSize,"E:\\Code\\Matlab\\PicTrans\\map.ma");
	savemat(cov1,cov1size,"E:\\Code\\Matlab\\PicTrans\\cov1.ma");
	//savemat(cov2,cov2size,"E:\\Code\\Matlab\\PicTrans\\cov2.ma");
	savemat(cov3,cov3size,"E:\\Code\\Matlab\\PicTrans\\cov3.ma");

	float** sample=UpSample(src,srcSize,2,2);
	nSize samSize={srcSize.c*2,srcSize.r*2};
	savemat(sample,samSize,"E:\\Code\\Matlab\\PicTrans\\sam.ma");
}
void test_mat1()
{
	int i,j;
	nSize srcSize={12,12};
	nSize mapSize={5,5};
	float** src=(float**)malloc(srcSize.r*sizeof(float*));
	for(i=0;i<srcSize.r;i++){
		src[i]=(float*)malloc(srcSize.c*sizeof(float));
		for(j=0;j<srcSize.c;j++){
			src[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}
	float** map1=(float**)malloc(mapSize.r*sizeof(float*));
	for(i=0;i<mapSize.r;i++){
		map1[i]=(float*)malloc(mapSize.c*sizeof(float));
		for(j=0;j<mapSize.c;j++){
			map1[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}
	float** map2=(float**)malloc(mapSize.r*sizeof(float*));
	for(i=0;i<mapSize.r;i++){
		map2[i]=(float*)malloc(mapSize.c*sizeof(float));
		for(j=0;j<mapSize.c;j++){
			map2[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}
	float** map3=(float**)malloc(mapSize.r*sizeof(float*));
	for(i=0;i<mapSize.r;i++){
		map3[i]=(float*)malloc(mapSize.c*sizeof(float));
		for(j=0;j<mapSize.c;j++){
			map3[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2; 
		}
	}

	float** cov1=cov(map1,mapSize,src,srcSize,valid);
	float** cov2=cov(map2,mapSize,src,srcSize,valid);
	nSize covsize={srcSize.c-(mapSize.c-1),srcSize.r-(mapSize.r-1)};
	float** cov3=cov(map3,mapSize,src,srcSize,valid);
	addmat(cov1,cov1,covsize,cov2,covsize);
	addmat(cov1,cov1,covsize,cov3,covsize);


	savemat(src,srcSize,"E:\\Code\\Matlab\\PicTrans\\src.ma");
	savemat(map1,mapSize,"E:\\Code\\Matlab\\PicTrans\\map1.ma");
	savemat(map2,mapSize,"E:\\Code\\Matlab\\PicTrans\\map2.ma");
	savemat(map3,mapSize,"E:\\Code\\Matlab\\PicTrans\\map3.ma");
	savemat(cov1,covsize,"E:\\Code\\Matlab\\PicTrans\\cov1.ma");
	savemat(cov2,covsize,"E:\\Code\\Matlab\\PicTrans\\cov2.ma");
	savemat(cov3,covsize,"E:\\Code\\Matlab\\PicTrans\\cov3.ma");

}
// ����cnnģ���Ƿ�������
/*
void test_cnn()
{

	LabelArr testLabel=read_Lable("Minst/train-labels.idx1-ubyte");
	ImgArr testImg=read_Img("Minst/train-images.idx3-ubyte");

	nSize inputSize={testImg->ImgPtr[0].c,testImg->ImgPtr[0].r};
	int outSize=testLabel->LabelPtr[0].l;

	CNN* cnn=(CNN*)malloc(sizeof(CNN));
	cnnsetup(cnn,inputSize,outSize);

	CNNOpts opts;
	opts.numepochs=1;
	opts.alpha=1;
	int trainNum=5000;
	cnntrain(cnn,testImg,testLabel,opts,trainNum);

	FILE  *fp=NULL;
	fp=fopen("E:\\Code\\Matlab\\PicTrans\\cnnL.ma","wb");
	if(fp==NULL)
		printf("write file failed\n");
	fwrite(cnn->L,sizeof(float),trainNum,fp);
	fclose(fp);
}*/


/*������*/
int main(int argc, char *argv[])
{
	LabelArr trainLabel=read_Lable("../Minst/train-labels.idx1-ubyte");
	ImgArr trainImg=read_Img("../Minst/train-images.idx3-ubyte");
	// cout << "hello" << endl;
	LabelArr testLabel=read_Lable("../Minst/t10k-labels.idx1-ubyte");
	ImgArr testImg=read_Img("../Minst/t10k-images.idx3-ubyte");

	nSize inputSize={testImg->ImgPtr[0].c,testImg->ImgPtr[0].r};
	int outSize=testLabel->LabelPtr[0].l;
	cout << "Data has been loaded..." << endl;
	t1 = clock();
	// CNN�ṹ�ĳ�ʼ��
	CNN* cnn=(CNN*)malloc(sizeof(CNN));
	cnnsetup(cnn,inputSize,outSize);
	cout << "Model initialization"<< endl;
	cout << "image size: (" << testImg->ImgPtr[0].c << ","<< testImg->ImgPtr[0].r << ")"<<endl;
	cout << outSize <<" classification issue" << endl;
	// CNNѵ��
	int myid, numprocs;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	
	CNNOpts opts;
	opts.numepochs=10;
	opts.alpha=1.0;
	// int trainNum=10000;
	// cout << "Model training"<< endl;
	// cnntrain(cnn,trainImg,trainLabel,opts,trainNum, myid, numprocs);
	// printf("train finished!!\n");
	// savecnn(cnn,"minst.cnn");
	// // ����ѵ�����?
	// FILE  *fp=NULL;
	// fp=fopen("../CNNData/cnnL.ma","wb");
	// if(fp==NULL)
	// 	printf("write file failed\n");
	// fwrite(cnn->L,sizeof(float),trainNum,fp);
	// fclose(fp);
	// //cout << "hello" << endl;
    //     t2 = clock();
    //     double endtime = (double)(t2-t1)/CLOCKS_PER_SEC;
    //     cout <<"train time: " <<endtime<<endl;
	// // CNN����
	importcnn(cnn,"minst.cnn");
	int testNum=5000;
	float incorrectRatio=0.0;
	t1 = clock();
	incorrectRatio=cnntest(cnn,testImg,testLabel,testNum, myid, numprocs);
	printf("test finished!! Error is %.4f\n",incorrectRatio);
	
	t2 = clock();
	double endtime = (double)(t2-t1)/CLOCKS_PER_SEC;
	if(myid==0)
		cout <<"train time: " <<endtime<<endl;
	MPI_Finalize();
	return 0;
}
