#define _SLOG_FILE_ "slog"
#define _CHECK_TIME  1
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
#include "json/json.h"
#include "opencv/cv.hpp"
using cv::Mat;
using cv::imshow;
using cv::imread;
using cv::imwrite;
using cv::waitKey;
#endif
/*==============================================================================================*/
int main(void) {
	int ret;
	slog_t slog("main");
	atomic<bool> run_(true);
	ret = eis_init();
	if(ret < 0){
		slog.err("eis_init error");
		return -1;
	}
#ifdef __arm__
        thread t_loadImage([&run_, &slog](){
		slog.info("thread loadImage start run.");
		while(run_){
			uint8_t *ptr0;
			int err = eis_input_command(EIS_COMMAND_GET_BUFFER, NULL, NULL, (void**)&ptr0);
			assert(err >= 0);
			quaternion_cls pos(1, 0, 0, 0);
			err = eis_input_command(EIS_COMMAND_RELEASE_BUFFER, ptr0, &pos);
			assert(err >= 0);
		}
		slog.info("thread loadImage exit.");
        });
#else
	karl::msgQueue<Mat> imgQueue;
        thread t_loadImage([&run_, &imgQueue](){
		int ret;
		slog_t slog("loadImage");
		Mat img, img_resize, img_rgb565, img_rgba;
		cv::VideoCapture capture("temp/eis_output_thread.h264");
		assert(capture.isOpened());
		std::ifstream mpu_ifs;
		mpu_ifs.open ("temp/mpu_output_thread", std::ifstream::in);
		assert(mpu_ifs.is_open());
		Json::Value mpu_root;
		Json::Reader mpu_reader;
		ret = mpu_reader.parse(mpu_ifs, mpu_root);
		assert(ret);
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
#endif
/*=========================================================*/
	int frame_cnt = 0;
#ifdef _CHECK_TIME
	timeval tv_s, tv_e;
	gettimeofday(&tv_s, NULL);
	float time_min = std::numeric_limits<float>::max();
#endif
	slog.info("main loop start.");
	while(run_){
		uint8_t* ptr0;
		int err = eis_output_command(EIS_COMMAND_GET_BUFFER, NULL, (void**)&ptr0);
		assert(err >= 0);
#ifndef __arm__
		Mat img_1, img_2;
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
			// img_1 = temp_image_1;
			cv::resize(temp_image_1, img_1, cv::Size(IMAGE_OUT_W/2, IMAGE_OUT_H/2));
		}
#endif
		err = eis_output_command(EIS_COMMAND_RELEASE_BUFFER, ptr0);
		assert(err >= 0);
#ifndef __arm__
		while(imgQueue.size() > 0){
			img_2 = imgQueue.pop();
			imshow("img_2", img_2);
		}
		// if(img_1.rows>0)
			imshow("img_1", img_1);

		int key = waitKey(10);
		if(key == 27){
			run_ = false;
		}else if(key == 112){
			do{
				key = waitKey(500);
			}while(key != 112);
		}
#endif
		// slog.info("main frame_cnt: %d", frame_cnt);
#ifdef _CHECK_TIME
		gettimeofday(&tv_e, NULL);
		time_min = std::min(time_min, (float)(tv_e.tv_sec - tv_s.tv_sec) + (float)(tv_e.tv_usec - tv_s.tv_usec) * 1.0e-6f);
		slog.info("main loop min period %fms", time_min*1000.0f);
		tv_s = tv_e;
#endif
		frame_cnt ++;
	}
	slog.info("main loop exit.");

#ifndef __arm__
	imgQueue.discon();
	t_loadImage.join();
#endif
	slog.info("t_loadImage.join.");
	eis_exit();
	return 0;
}

