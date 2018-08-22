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
#include "json/json.h"


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


#include "opencv/cv.hpp"
using cv::Mat;
using cv::imshow;
using cv::imread;
using cv::imwrite;
using cv::waitKey;
/*==============================================================================================*/
int main(void) {
	slog_t slog("main");
	atomic<bool> run_(true);
        assert(eis_init() == 0);

	karl::msgQueue<Mat> imgQueue;
        thread t_loadImage([&run_, &imgQueue](){
		slog_t slog("loadImage");
		Mat img, img_resize, img_rgb565, img_rgba;
		cv::VideoCapture capture("temp/eis_output_thread.h264");
		assert(capture.isOpened());
		std::ifstream mpu_ifs;
		mpu_ifs.open ("temp/mpu_output_thread", std::ifstream::in);
		assert(mpu_ifs.is_open());
		Json::Value mpu_root;
		Json::Reader mpu_reader;
		assert(mpu_reader.parse(mpu_ifs, mpu_root));
		slog.info("thread loadImage start run.");
		while(run_){
			auto ret = capture.read(img);
			//cout << "CV_CAP_PROP_POS_FRAMES " << capture.get(CV_CAP_PROP_POS_FRAMES) << endl;
			//cout << "CV_CAP_PROP_POS_MSEC " << capture.get(CV_CAP_PROP_POS_MSEC) << endl;
			if(!ret){
				slog.war("capture read error.");
				run_ = false;
				continue ;
			}
			int frame_index = capture.get(CV_CAP_PROP_POS_FRAMES)-1;
			quaternion_cls pos(
				(double)mpu_root["quaternion"][frame_index][0].asInt() / 1073741824.0,
				(double)mpu_root["quaternion"][frame_index][1].asInt() / 1073741824.0,
				(double)mpu_root["quaternion"][frame_index][2].asInt() / 1073741824.0,
				(double)mpu_root["quaternion"][frame_index][3].asInt() / 1073741824.0
			);
			printf("frame index %d, quaternion %f, %f, %f, %f\n", frame_index, pos.get_q0(), pos.get_q1(), pos.get_q2(), pos.get_q3());
			cv::resize(img, img_resize, cv::Size(IMAGE_IN_W, IMAGE_IN_H));
			cv::cvtColor(img_resize, img_rgb565, CV_BGR2BGR565);
			cv::cvtColor(img_resize, img_rgba, CV_BGR2RGBA);
			uint8_t *ptr0;
			int err = eis_input_command(EIS_COMMAND_GET_BUFFER, NULL, NULL, (void**)&ptr0);
			assert(err >= 0);
			for(int row=0;row<IMAGE_IN_H;row++)
				memcpy(ptr0 + row*IMAGE_IN_W*4, img_rgba.ptr<uint8_t>(row), sizeof(uint8_t)*IMAGE_IN_W*4);
			err = eis_input_command(EIS_COMMAND_RELEASE_BUFFER, ptr0, &pos);
			assert(err >= 0);
			Mat img_resize_1;
			cv::resize(img_resize, img_resize_1, cv::Size(IMAGE_IN_W/2, IMAGE_IN_H/2));
			imgQueue.push(img_resize_1);
		}
		slog.info("thread loadImage exit.");
		mpu_ifs.close();
		capture.release();
        });
/*=========================================================*/
	int frame_cnt = 0;
	slog.info("main loop start.");
	Mat img_1, img_2;
	while(run_){
		uint8_t* ptr0;
		int err = eis_output_command(EIS_COMMAND_GET_BUFFER, NULL, (void**)&ptr0);
		assert(err >= 0);
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
			cv::cvtColor(temp_image_0, temp_image_1, CV_BGRA2RGB);
			cv::resize(temp_image_1, img_1, cv::Size(IMAGE_OUT_W/2, IMAGE_OUT_H/2));
		}
		err = eis_output_command(EIS_COMMAND_RELEASE_BUFFER, ptr0);
		assert(err >= 0);
		imshow("img_1", img_1);
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
#ifndef __arm__
		while(imgQueue.size() > 0)
			img_2 = imgQueue.pop();

		
		imshow("img_2", img_2);
#endif
		slog.info("main frame_cnt: %d", frame_cnt);
		frame_cnt ++;
		
#ifndef __arm__
		int key = waitKey(10);
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

#if 0
	if(writer.isOpened())
		writer.release();
	imgQueue.discon();
#endif
	t_loadImage.join();

	slog.info("t_loadImage.join.");
	eis_exit();
	return 0;
}

