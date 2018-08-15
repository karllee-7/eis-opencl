#define _SLOG_FILE_ "slog"
// #define _CAM_UNDISTORT_
/*==============================================================================================*/
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <utility>
#include <string>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "unistd.h"
#include <cmath>
#include <assert.h>
#include <sys/types.h>
#include "sys/time.h"
#include <time.h>
#include "eis.h"
#include "slog.hpp"
#include "threadPool.hxx"

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::array;
using std::memcpy;
using std::move;
using std::rand;
using std::thread;
using std::atomic;
using std::mutex;
using std::unique_lock;
using std::condition_variable;

#ifndef __arm__
#include "opencv/cv.hpp"
using cv::Mat;
using cv::imshow;
using cv::imread;
using cv::imwrite;
using cv::waitKey;
#endif
/*==============================================================================================*/
int main(void) {
	slog_t slog("main");
	atomic<bool> run_(true);
        if(eis_init()){
		slog.err("init eis error.");
		return -1;
	}
#ifndef __arm__
	karl::msgQueue<Mat> imgQueue;
#endif

#ifdef __arm__
        thread t_loadImage([&run_](){
		slog_t slog("loadImage");
		auto file_buffer = (uint8_t*)malloc(IMAGE_IN_W*IMAGE_IN_H*2);
		FILE* fd = fopen("temp/img.dat", "rb");
		if(fd == NULL){
			slog.err("can not open file img.dat");
			free(file_buffer);
			run_ = false;
			return ;
		}
		int cnt = fread(file_buffer, sizeof(uint8_t), IMAGE_IN_W*IMAGE_IN_H*2, fd);
		if(cnt != IMAGE_IN_W*IMAGE_IN_H*2){
			slog.err("file img.dat length error.");
			fclose(fd);
			free(file_buffer);
			run_ = false;
			return ;
		}
		fclose(fd);
		int frame_cnt = 0;
		slog.info("thread loadImage start run.");
		while(run_){
			uint8_t *ptr0, *ptr1;
			int err;
			err = eis_input_command(EIS_COMMAND_GET_BUFFER, NULL, NULL, (void**)&ptr0, (void**)&ptr1);
			assert(err >= 0);
			memcpy(ptr0, file_buffer, sizeof(uint8_t)*IMAGE_IN_W*IMAGE_IN_H*2);
			memcpy(ptr1, file_buffer, sizeof(uint8_t)*IMAGE_IN_W*IMAGE_IN_H*2);
			err = eis_input_command(EIS_COMMAND_RELEASE_BUFFER, ptr0, ptr1);
			assert(err >= 0);
			frame_cnt++;
		}
		slog.info("thread loadImage exit.");
		free(file_buffer);
        });
/*=========================================================*/
#else
        thread t_loadImage([&run_, &imgQueue](){
		slog_t slog("loadImage");
		Mat img, img_resize, img_rgb565, img_rgba;
		// cv::VideoCapture capture("temp/IMG_0235.MOV");
		cv::VideoCapture capture("temp/15530444.h264");
		// cv::VideoCapture capture("temp/20180525-14-15.mp4");
		// cv::VideoCapture capture("temp/H249_0.mp4");
		if(!capture.isOpened()){
			slog.err("can't capture the video file");
			run_ = false;
			return ;
		}
		slog.info("thread loadImage start run.");
		while(run_){
			auto ret = capture.read(img);
			if(!ret){
				slog.war("capture read error.");
				run_ = false;
				continue ;
			}

#ifdef _CAM_UNDISTORT_
			{
				Mat camera_matrix(3,3,CV_32F);
				Mat distort_coeffs(5,1,CV_32F);
				camera_matrix.ptr<float>(0)[0] = 1398.62903365191;
				camera_matrix.ptr<float>(0)[1] = 0;
				camera_matrix.ptr<float>(0)[2] = 1050.91363112597;
				camera_matrix.ptr<float>(1)[0] = 0;
				camera_matrix.ptr<float>(1)[1] = 1365.71260963290;
				camera_matrix.ptr<float>(1)[2] = 496.326169224848;
				camera_matrix.ptr<float>(2)[0] = 0;
				camera_matrix.ptr<float>(2)[1] = 0;
				camera_matrix.ptr<float>(2)[2] = 1;
				distort_coeffs.ptr<float>(0)[0] = -0.393348138522701;
				distort_coeffs.ptr<float>(1)[0] = 0.152668588560805;
				distort_coeffs.ptr<float>(2)[0] = 0;
				distort_coeffs.ptr<float>(3)[0] = 0;
				distort_coeffs.ptr<float>(4)[0] = 0;
				Mat dst;
				cv::undistort(img, dst, camera_matrix, distort_coeffs);
				img = dst;
			}
#endif

			cv::resize(img, img_resize, cv::Size(IMAGE_IN_W, IMAGE_IN_H));
			cv::cvtColor(img_resize, img_rgb565, CV_BGR2BGR565);
			cv::cvtColor(img_resize, img_rgba, CV_BGR2RGBA);

			uint8_t *ptr0, *ptr1;
			int err;
			err = eis_input_command(EIS_COMMAND_GET_BUFFER, NULL, NULL, (void**)&ptr0, (void**)&ptr1);
			assert(err >= 0);
			for(int row=0;row<IMAGE_IN_H;row++)
				memcpy(ptr0 + row*IMAGE_IN_W*2, img_rgb565.ptr<uint8_t>(row), sizeof(uint8_t)*IMAGE_IN_W*2);
			for(int row=0;row<IMAGE_IN_H;row++)
				memcpy(ptr1 + row*IMAGE_IN_W*4, img_rgba.ptr<uint8_t>(row), sizeof(uint8_t)*IMAGE_IN_W*4);
			err = eis_input_command(EIS_COMMAND_RELEASE_BUFFER, ptr0, ptr1);
			assert(err >= 0);
#ifndef __arm__
			Mat img_resize_1;
			cv::resize(img_resize, img_resize_1, cv::Size(IMAGE_IN_W/2, IMAGE_IN_H/2));
			imgQueue.push(img_resize_1);
#endif
		}
		slog.info("thread loadImage exit.");
		capture.release();
        });
#endif
/*=========================================================*/
#ifndef __arm__
	cv::VideoWriter writer("temp/eis_out.avi", cv::VideoWriter::fourcc('D','I','V','X'), 30.0, cv::Size(IMAGE_OUT_W, IMAGE_OUT_H));
        if(!writer.isOpened()){
		slog.err("error: cant creat the output file");
		run_ = false;
        }
#endif
/*=========================================================*/
	int frame_cnt = 0;
	struct timeval t_s, t_e;
#ifndef __arm__
	Mat img_0, img_1, img_2;
#endif

	slog.info("main loop start.");
	while(run_){
		uint8_t* ptr0;
		int err;
		err = eis_output_command(EIS_COMMAND_GET_BUFFER, NULL, (void**)&ptr0);
		assert(err >= 0);
#ifndef __arm__
		{
			Mat temp_image_0(IMAGE_OUT_H, IMAGE_OUT_W, CV_8UC4);
			Mat temp_image_1;
			for(int row=0;row<IMAGE_OUT_H;row++){
				uint8_t* src_ptr = ptr0 + row*IMAGE_OUT_W*4;
				uint8_t* dst_ptr = temp_image_0.ptr<uint8_t>(row);
				for(int col=0;col<IMAGE_OUT_W*4;col++){
					dst_ptr[col] = src_ptr[col];
				}
			}
			cv::cvtColor(temp_image_0, img_0, CV_RGBA2BGR);
			cv::resize(img_0, img_1, cv::Size(IMAGE_OUT_W/2, IMAGE_OUT_H/2));
		}
		imshow("1", img_1);
		writer.write(img_0);
#endif
#if 0
		{
			array<size_t, 3> image_origin = {0, 0, 0};
			array<size_t, 3> image_region = {IMAGE_OUT_W, IMAGE_OUT_H, 1};
			size_t image_row_pitch = 0;
			auto ptr0 = (uint8_t*)queue.enqueueMapImage(msgIn.m_img, CL_TRUE, CL_MAP_READ, image_origin, image_region, &image_row_pitch, NULL);
			printf("img data: %d\n", ptr0[(IMAGE_OUT_W*IMAGE_OUT_H)/2*2]);
			queue.enqueueUnmapMemObject(msgIn.m_img, ptr0);
		}
#endif
#if 0
		Mat img;
		{
			Mat gray(GRAY_H, GRAY_W, CV_8UC1);
			auto ptr0 = (uint8_t*)queue.enqueueMapBuffer(msgIn.m_gray, CL_TRUE, CL_MAP_READ, 0, sizeof(uint8_t)*GRAY_W*GRAY_H);
			for(int row=0;row<GRAY_H;row++)
				memcpy(gray.ptr<uint8_t>(row), ptr0 + row*GRAY_W, sizeof(uint8_t)*GRAY_W);
			queue.enqueueUnmapMemObject(msgIn.m_gray, ptr0);
			cv::cvtColor(gray, img, CV_GRAY2BGR);
		}
		//  imshow("2", gray);
#endif
#if 0
		{
			auto ptr0 = (int*)queue.enqueueMapBuffer(msgIn.m_kps, CL_TRUE, CL_MAP_READ, 0, sizeof(int)*KPS_MAXNUM*2);
			auto ptr1 = (float*)queue.enqueueMapBuffer(msgIn.m_angle, CL_TRUE, CL_MAP_READ, 0, sizeof(float)*KPS_MAXNUM*2);
			for(int i=0;i<msgIn.kps_num;i++){
				auto x0 = cv::Point(ptr0[i*2+0], ptr0[i*2+1]);
				auto x1 = cv::Point(ptr0[i*2+0]+ptr1[i*2+0]*16, ptr0[i*2+1]+ptr1[i*2+1]*16);
				cv::circle(img, x0, 16, cv::Scalar(0,0,255), 1, cv::LINE_8);
				// cv::circle(img, x0, 2, cv::Scalar(0,0,255), 1, cv::LINE_8);
				cv::line(img, x0, x1, cv::Scalar(0,255,0));
			}
			queue.enqueueUnmapMemObject(msgIn.m_kps, ptr0);
		}
		imshow("3", img);
#endif
#if 0
		{
			auto ptr0 = (float*)queue.enqueueMapBuffer(msgIn.m_matches, CL_TRUE, CL_MAP_READ, 0, sizeof(float)*MATCHES_MAXNUM*4);
			for(int i=0;i<msgIn.matches_num;i++){
				auto x0 = cv::Point(ptr0[i*4+0]+GRAY_W/2, ptr0[i*4+1]+GRAY_H/2);
				auto x1 = cv::Point(ptr0[i*4+2]+GRAY_W/2, ptr0[i*4+3]+GRAY_H/2);
				cv::circle(img, x0, 16, cv::Scalar(0,0,255), 1, cv::LINE_8);
				cv::line(img, x0, x1, cv::Scalar(0,255,0));
			}
			queue.enqueueUnmapMemObject(msgIn.m_matches, ptr0);
		}
		imshow("4", img);
#endif
		err = eis_output_command(EIS_COMMAND_RELEASE_BUFFER, ptr0);
		assert(err >= 0);
#ifndef __arm__
		while(imgQueue.size() > 0)
			img_2 = imgQueue.pop();

		
		imshow("main 1", img_2);
#endif
		slog.info("main frame_cnt: %d", frame_cnt);
		if(frame_cnt % 30 == 0){
			gettimeofday(&t_e,NULL);
			slog.info_v0("time used is %fs", (float)(t_e.tv_sec - t_s.tv_sec) + (float)(t_e.tv_usec - t_s.tv_usec)/1000000.0f);
			t_s = t_e;
		}
		if(frame_cnt > 600)
			run_ = false;
		frame_cnt ++;
		
#ifndef __arm__
		int key = waitKey(1);
		if(key == 27){
			run_ = false;
		}else if(key == 112){
			do{
				key = waitKey(500);
			}while(key != 112);
		}
#endif
	}
	slog.info("main loop exit.");

#ifndef __arm__
	if(writer.isOpened())
		writer.release();
	imgQueue.discon();
#endif
	t_loadImage.join();

	slog.info("t_loadImage.join.");
	eis_exit();
	return 0;
}

