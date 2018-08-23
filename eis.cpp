#ifdef __arm__
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#else
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#endif

#define CL_HPP_ENABLE_EXCEPTIONS

#define _SLOG_FILE_ "slog"

#ifndef __arm__
#define _DRAW_KPS_
#endif
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
#include <map>
#include <random>
#include <mutex>
#include <condition_variable>
#include <limits>
#include "unistd.h"
#include <cmath>
#include <assert.h>
#include <sstream>
#include <sys/types.h>
#include "sys/time.h"
#include <time.h>
#include "eis.h"
#include "slog.hpp"
#include "pError.hpp"
#include <CL/cl2.hpp>
#include "threadPool.hxx"
#include "eigen3/Eigen/Dense"


#include "bit_pattern_31.h"
// #include "matchRand4_table.h"

using std::string;
using std::cout;
using std::endl;
using std::vector;
using std::map;
using std::array;
using std::memcpy;
using std::move;
using std::rand;
using std::thread;
using std::atomic;
using std::mutex;
using std::stringstream;
using std::unique_lock;
using std::condition_variable;
using Eigen::Matrix;
using Eigen::JacobiSVD;
using Eigen::BDCSVD;
using Eigen::MatrixXf;
using Eigen::MatrixXd;
using Eigen::Matrix3f;
using Eigen::VectorXf;
using std::fstream;
using std::nth_element;

#ifndef __arm__
#include "opencv/cv.hpp"
using cv::Mat;
using cv::imshow;
using cv::imread;
using cv::imwrite;
using cv::waitKey;
#endif

#ifdef __arm__
#include<arm_neon.h>
#endif

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXd;
/*==============================================================================================*/
class loadImageMsg_t{
public:
        cl::Image2D m_img;
	quaternion_cls pos;
	loadImageMsg_t(){}
	loadImageMsg_t(const loadImageMsg_t& a){
		m_img = a.m_img;
		pos = a.pos;
	}
	loadImageMsg_t(cl::Image2D& m_img, quaternion_cls& pos):
		m_img(m_img),
		pos(pos){
	}
};
class postImageMsg_t{
public:
        cl::Image2D  m_img;
	postImageMsg_t(){}
	postImageMsg_t(cl::Image2D& m_img): 
		m_img(m_img){
	}
};
/*==============================================================================================*/
template < typename T >
inline T square(T a){
	return a * a;
}
cl::Program createProgramWithFile(vector<cl::Device> &devices, cl::Context& context, const char* buildOptions, const char* fileName){
	slog_t slog;

	std::ifstream programFile(fileName);
	if(!programFile.is_open())
		throw cl::Error(-1, "can't open kernel file.");
	std::stringstream buffer;
	buffer << programFile.rdbuf();
	std::string source(buffer.str());

	cl::Program program(context, source);
	program.build(devices, buildOptions); 

	return program;
}
inline void cl_fillBuffer(cl::CommandQueue& queue, cl::Buffer& buffer, int data, size_t offset, size_t size, const vector<cl::Event>* events = NULL, cl::Event* event = NULL){
	auto ptr = (uint8_t*)queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, offset, size, events);
	memset(ptr, data, size);
	queue.enqueueUnmapMemObject(buffer, ptr, NULL, event);
}
inline void cl_clearBufferDistribute(cl::CommandQueue& queue, cl::Buffer& buffer, size_t offset, size_t size, size_t stride, size_t num, const vector<cl::Event>* events = NULL, cl::Event* event = NULL){
	assert(stride >= size);
	auto ptr = (uint8_t*)queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, offset, stride * num, events);
	for(size_t i=0;i<num;i++)
		memset(ptr+stride*i, 0, size);
	queue.enqueueUnmapMemObject(buffer, ptr, NULL, event);
}
Matrix<float, 3, 3> pos2array(const quaternion_cls& pos){
	Matrix<float, 3, 3> retv;
	double q0 = pos.get_q0();
	double q1 = pos.get_q1();
	double q2 = pos.get_q2();
	double q3 = pos.get_q3();
	retv << 1.0-2.0*(q2*q2+q3*q3), 2.0*(q1*q2-q0*q3), 2.0*(q1*q3+q0*q2),
	        2.0*(q1*q2+q0*q3), 1-2.0*(q1*q1+q3*q3), 2.0*(q2*q3-q0*q1),
	        2.0*(q1*q3-q0*q2), 2.0*(q2*q3+q0*q1), 1-2.0*(q1*q1+q2*q2);
	return retv;
}
Matrix<float, 3, 1> pos2rotate(const quaternion_cls& pos){
	Matrix<float, 3, 1> retv;
	double q0 = pos.get_q0();
	double q1 = pos.get_q1();
	double q2 = pos.get_q2();
	double q3 = pos.get_q3();
        retv << atan2(2*(q0*q1+q2*q3), 1-2*(q1*q1+q2*q2)) * 57.3,
                asin(2*(q0*q2-q1*q3)) * 57.3,
                atan2(2*(q0*q3+q1*q2), 1-2*(q2*q2+q3*q3)) * 57.3;
	return retv;
}
/*==============================================================================================*/
atomic<bool> run_(false);
stringstream clBuildOptions;
vector<cl::Platform> platforms;
vector<cl::Device> devices;
cl::Context context;
#ifdef __arm__
static cl::ImageFormat image_format(CL_RGB, CL_UNORM_SHORT_565);
#else
static cl::ImageFormat image_format(CL_RGBA, CL_UNORM_INT8);
#endif
//static cl::ImageFormat gauss_format(CL_R, CL_UNORM_INT8);

thread t_wrapImage;

karl::msgQueue<loadImageMsg_t> loadImageQueue(3);
karl::msgQueue<postImageMsg_t> postImageQueue(2);
bool isDebug = true;
/*==============================================================================================*/
int eis_input_command(int command, void* ptr_image2d, quaternion_cls *pos, void** pptr_image2d){
	static slog_t slog("eis_input_command");
	static cl::CommandQueue queue;
	static map<void*, cl::Image2D> image2d_container;
	static array<cl::size_type, 3> image_origin = {0, 0, 0};
	static array<cl::size_type, 3> image_region = {IMAGE_IN_W, IMAGE_IN_H, 1};
	static size_t image_row_pitch = 0;
	int cmd = command & EIS_COMMAND_MASK;
	int flg = command & EIS_FLAG_MASK;
	if(cmd == EIS_COMMAND_CLEAR_ALL){
		if(image2d_container.size() > 0){
			for(auto iter=image2d_container.begin(); iter!=image2d_container.end();iter++)
				queue.enqueueUnmapMemObject(iter->second, iter->first);
			image2d_container.erase(image2d_container.begin(), image2d_container.end());
		}
		return 0;
	}
	if(run_ == false || platforms.size() == 0 || devices.size() == 0 || context.get() == NULL){	
		slog.war("platforms|devices|context is empty, or pipline is not running, please call eis_init() first.");
		return -1;
	}
	if(queue.get() == NULL){
		queue = cl::CommandQueue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
	}
	if(cmd == EIS_COMMAND_GET_BUFFER){
		if(pptr_image2d == NULL){
			slog.war("pptr_buffer | pptr_image2d cannot be NULL with command EIS_COMMAND_GET_BUFFER.");
			return -1;
		}
		if(image2d_container.size() > 0){
			slog.war("input image2d has been created. do not request again.");
			return -1;
		}
		cl::Image2D image2d(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, image_format, IMAGE_IN_W, IMAGE_IN_H);
		void* ptr1 = queue.enqueueMapImage(image2d, CL_TRUE, CL_MAP_WRITE, image_origin, image_region, &image_row_pitch, NULL);
		image2d_container[ptr1] = image2d;
		*pptr_image2d = ptr1;
		return 1;
	}
	if(cmd == EIS_COMMAND_RELEASE_BUFFER){
		if(ptr_image2d == NULL){
			slog.war("image2d ptr is missing.");
			return -1;
		}
		if(image2d_container.find(ptr_image2d) == image2d_container.end()){
			slog.war("no image2d found.");
			return -1;
		}
		loadImageMsg_t msgOut(image2d_container[ptr_image2d], *pos);
		image2d_container.erase(ptr_image2d);
		queue.enqueueUnmapMemObject(msgOut.m_img, ptr_image2d);
		if(flg == EIS_FLAG_NOLOCK && loadImageQueue.is_full())
			return 0;
		loadImageQueue.push(msgOut);
		return 1;
	}
	slog.war("unknown command");
	return -2;
}
/*==============================================================================================*/
int eis_output_command(int command, void* ptr_image2d, void** pptr_image2d){
	static slog_t slog("eis_output_command");
	static cl::CommandQueue queue;
	static map<void*, cl::Image2D> image2d_container;
	static array<cl::size_type, 3> image_origin = {0, 0, 0};
	static array<cl::size_type, 3> image_region = {IMAGE_OUT_W, IMAGE_OUT_H, 1};
	static size_t image_row_pitch = 0;
	int cmd = command & EIS_COMMAND_MASK;
	int flg = command & EIS_FLAG_MASK;
	if(command == EIS_COMMAND_CLEAR_ALL){
		if(image2d_container.size() > 0){
			for(auto iter=image2d_container.begin(); iter!=image2d_container.end();iter++)
				queue.enqueueUnmapMemObject(iter->second, iter->first);
			image2d_container.erase(image2d_container.begin(), image2d_container.end());
		}
		return 0;
	}
	if(run_ == false || platforms.size() == 0 || devices.size() == 0 || context.get() == NULL){	
		slog.war("platforms|devices|context is empty, or pipline is not running, please call eis_init() first.");
		return -1;
	}
	if(queue.get() == NULL){
		queue = cl::CommandQueue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
	}
	if(cmd == EIS_COMMAND_GET_BUFFER){
		if(pptr_image2d == NULL){
			slog.war("pptr_image2d cannot be NULL with command EIS_COMMAND_GET_BUFFER.");
			return -1;
		}
		if(image2d_container.size() > 0){
			slog.war("output image2d has been created. do not request again.");
			return -1;
		}
		if(flg == EIS_FLAG_NOLOCK && postImageQueue.is_empty())
			return 0;
		auto msgIn = postImageQueue.pop();
		void* ptr0 = queue.enqueueMapImage(msgIn.m_img, CL_TRUE, CL_MAP_READ, image_origin, image_region, &image_row_pitch, NULL);
		image2d_container[ptr0] = msgIn.m_img;
		*pptr_image2d = ptr0;
		return 1;
	}
	if(cmd == EIS_COMMAND_RELEASE_BUFFER){
		if(ptr_image2d == NULL){
			slog.war("image2d ptr is missing.");
			return -1;
		}
		if(image2d_container.find(ptr_image2d) == image2d_container.end()){
			slog.war("no image2d found.");
			return -1;
		}
		queue.enqueueUnmapMemObject(image2d_container[ptr_image2d], ptr_image2d);
		image2d_container.erase(ptr_image2d);
		return 1;
	}
	slog.war("unknown command");
	return -2;
}
/*==============================================================================================*/
int eis_init() {
	cl_int cl_err;
	std::srand(1);
	slog_t slog("eis main");
	clBuildOptions \
		<< "-cl-no-signed-zeros "\
		<< "-DIMAGE_IN_W=" << IMAGE_IN_W << " " \
		<< "-DIMAGE_IN_H=" << IMAGE_IN_H << " " \
		<< "-DIMAGE_OUT_W=" << IMAGE_OUT_W << " " \
		<< "-DIMAGE_OUT_H=" << IMAGE_OUT_H << " " \
		<< "-DCAM_PAM_F=" << CAM_PAM_F << " " \
		<< "-DGRAY_W=" << GRAY_W << " " \
		<< "-DGRAY_H=" << GRAY_H << " " \
		<< "-DBLOCK_EST_MAXNUM=" << BLOCK_EST_MAXNUM << " " \
		<< "-DBLOCK_NOMAX_MAXNUM=" << BLOCK_NOMAX_MAXNUM << " " \
		<< "-DKPS_EST_THRESHOLD=" << KPS_EST_THRESHOLD << " " \
		<< "-DMATCHES_MAXNUM=" << MATCHES_MAXNUM << " " \
		<< "-DMATCHES_DISTANCE=" << MATCHES_DISTANCE << " " \
		<< "-DSOLVE_MATRIXA_RANSAC_PRECISE=" << SOLVE_MATRIXA_RANSAC_PRECISE << " " ;
/*=========================================================*/
	if(run_ == true){
		slog.war("pipline is running already, do not init again.");
		return 0;
	}
	run_ = true;
/*=========================================================*/
	assert(!cl::Platform::get(&platforms));
	assert(!platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices));
	for(size_t i=0; i<devices.size(); i++) 
		slog.info("Device: %s Version: %s", devices[i].getInfo<CL_DEVICE_NAME>().c_str(), devices[i].getInfo<CL_DRIVER_VERSION>().c_str());

	context = cl::Context(devices, NULL, NULL, NULL, &cl_err);
	assert(!cl_err);
	slog.info("creat context success.");
/*=========================================================*/
#if 0
	vector<cl::ImageFormat> deviceImageFormats;
	context.getSupportedImageFormats(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, CL_MEM_OBJECT_IMAGE2D, &deviceImageFormats);
	for(auto elem : deviceImageFormats)
		printf("support image_channel_order: 0x%04x, image_channel_data_type: 0x%04x\n", elem.image_channel_order, elem.image_channel_data_type);
#endif
/*=========================================================*/
        t_wrapImage = thread([](){
		slog_t slog("normalizeImage");
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		cl::Program program;
		string fileName = "kernels/eis.cl";
		try {
			program = createProgramWithFile(devices, context, clBuildOptions.str().c_str(), fileName.c_str());
		} catch (cl::BuildError& e){
			slog.err("buld program (%s) error. info: %s, log: %s", fileName.c_str(), e.what(), e.getBuildLog()[0].second.c_str());
			run_ = false;
			return ;
		}
		cl::Sampler sampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_LINEAR);
		cl::Kernel kernel_0(program, "prepareImage");
		cl::Kernel kernel_1(program, "gaussFiler");
		cl::Kernel kernel_2(program, "gaussFiler");
		cl::Kernel kernel_3(program, "gaussFiler");
		cl::Kernel kernel_4(program, "gaussFiler");
		cl::Kernel kernel_5(program, "cvtColor_gray");

                cl::NDRange k0_offset(0, 0);
                cl::NDRange k0_global(IMAGE_IN_W, IMAGE_IN_H);
                cl::NDRange k0_local(32, 8);

                cl::NDRange k1_offset(0, 0);
                cl::NDRange k1_global(IMAGE_IN_W/2, IMAGE_IN_H/2);
                cl::NDRange k1_local(32, 8);

                cl::NDRange k2_offset(0, 0);
                cl::NDRange k2_global(IMAGE_IN_W/4, IMAGE_IN_H/4);
                cl::NDRange k2_local(32, 4);

                cl::NDRange k3_offset(0, 0);
                cl::NDRange k3_global(IMAGE_IN_W/8, IMAGE_IN_H/8);
                cl::NDRange k3_local(16, 2);

                cl::NDRange k4_offset(0, 0);
                cl::NDRange k4_global(IMAGE_IN_W/16, IMAGE_IN_H/16);
                cl::NDRange k4_local(8, 1);

		loadImageMsg_t msgPre;
                cl::Buffer m_matrixA(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*16);
                cl::Buffer m_org_gray(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W*IMAGE_IN_H);
                cl::Buffer m_gray_L1(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W/2*IMAGE_IN_H/2);
                cl::Buffer m_gray_L2(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W/4*IMAGE_IN_H/4);
                cl::Buffer m_gray_L3(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W/8*IMAGE_IN_H/8);
                cl::Buffer m_gray_L4(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W/16*IMAGE_IN_H/16);
                cl::Buffer m_pre_org_gray(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W*IMAGE_IN_H);
                cl::Buffer m_pre_gray_L1(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W/2*IMAGE_IN_H/2);
                cl::Buffer m_pre_gray_L2(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W/4*IMAGE_IN_H/4);
                cl::Buffer m_pre_gray_L3(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W/8*IMAGE_IN_H/8);
                cl::Buffer m_pre_gray_L4(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W/16*IMAGE_IN_H/16);
		
		assert(k0_global.get()[0] % k0_local.get()[0] == 0);
		assert(k0_global.get()[1] % k0_local.get()[1] == 0);
		assert(k1_global.get()[0] % k1_local.get()[0] == 0);
		assert(k1_global.get()[1] % k1_local.get()[1] == 0);
		assert(k2_global.get()[0] % k2_local.get()[0] == 0);
		assert(k2_global.get()[1] % k2_local.get()[1] == 0);
		assert(k3_global.get()[0] % k3_local.get()[0] == 0);
		assert(k3_global.get()[1] % k3_local.get()[1] == 0);
		assert(k4_global.get()[0] % k4_local.get()[0] == 0);
		assert(k4_global.get()[1] % k4_local.get()[1] == 0);
                slog.info("thread normalize Image start run.");
		bool isFirst = true;
		quaternion_cls pos_curr(1.0, 0.0, 0.0, 0.0);
                while(run_){
			vector<cl::Event> event_0(1), event_1(1), event_2(1), event_3(1), event_4(1), event_5(1), event_6(1), event_7(1);
			vector<cl::Event> event_8(1), event_9(1), event_10(1), event_11(1);
			cl::Image2D m_debugImg(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, image_format, IMAGE_OUT_W, IMAGE_OUT_H);
			cl::Image2D m_debugImg_aux(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, image_format, IMAGE_OUT_W, IMAGE_OUT_H);
#if 0
			cl::Image2D m_outImg;
			try{
				m_outImg = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, image_format, IMAGE_OUT_W, IMAGE_OUT_H);
			} catch(cl::Error& e) { 
				slog.err("malloc buffer: %s | %d", e.what(), e.err()); 
				run_ = false; 
				continue; 
			}
#endif
                        loadImageMsg_t msgIn = loadImageQueue.pop();
			if(isFirst){
				msgPre = msgIn;
				isFirst = false;
				continue;
			}
			try{
				auto ptr0 = (float*)queue.enqueueMapBuffer(m_matrixA, CL_FALSE, CL_MAP_WRITE, 0, sizeof(float)*16, NULL, &event_0[0]);
				quaternion_cls pos_diff = msgIn.pos * msgPre.pos.inv();
				pos_curr = pos_diff * pos_curr;
				auto matrixA = pos2array(pos_curr.inv());
				event_0[0].wait();
				for(int i=0;i<9;i++)
					ptr0[i] = matrixA(i/3, i%3);
				queue.enqueueUnmapMemObject(m_matrixA,  ptr0, &event_0, &event_1[0]);
			} catch(cl::Error& e){
				slog.err("fill matrixA err | %s | %d", e.what(), e.err()); 
				if(event_0[0]() != NULL) slog.err("event_0: %d", event_0[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_1[0]() != NULL) slog.err("event_1: %d", event_1[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
			try{ 
				kernel_0.setArg(0, msgIn.m_img);
				kernel_0.setArg(1, m_org_gray);
				kernel_0.setArg(2, sampler);
				kernel_0.setArg(3, m_matrixA);
				kernel_0.setArg(4, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_0, k0_offset, k0_global, k0_local, &event_1, &event_2[0]);
				kernel_1.setArg(0, m_org_gray);
				kernel_1.setArg(1, m_gray_L1);
				kernel_1.setArg(2, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_1, k1_offset, k1_global, k1_local, &event_2, &event_3[0]);
				kernel_2.setArg(0, m_gray_L1);
				kernel_2.setArg(1, m_gray_L2);
				kernel_2.setArg(2, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_2, k2_offset, k2_global, k2_local, &event_3, &event_4[0]);
				kernel_3.setArg(0, m_gray_L2);
				kernel_3.setArg(1, m_gray_L3);
				kernel_3.setArg(2, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_3, k3_offset, k3_global, k3_local, &event_4, &event_5[0]);
				kernel_4.setArg(0, m_gray_L3);
				kernel_4.setArg(1, m_gray_L4);
				kernel_4.setArg(2, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_4, k4_offset, k4_global, k4_local, &event_5, &event_6[0]);
				kernel_5.setArg(0, msgPre.m_img);
				kernel_5.setArg(1, m_pre_org_gray);
				kernel_5.setArg(2, sampler);
				kernel_5.setArg(3, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_5, k0_offset, k0_global, k0_local, &event_6, &event_7[0]);
				kernel_1.setArg(0, m_pre_org_gray);
				kernel_1.setArg(1, m_pre_gray_L1);
				kernel_1.setArg(2, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_1, k1_offset, k1_global, k1_local, &event_7, &event_8[0]);
				kernel_2.setArg(0, m_pre_gray_L1);
				kernel_2.setArg(1, m_pre_gray_L2);
				kernel_2.setArg(2, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_2, k2_offset, k2_global, k2_local, &event_8, &event_9[0]);
				kernel_3.setArg(0, m_pre_gray_L2);
				kernel_3.setArg(1, m_pre_gray_L3);
				kernel_3.setArg(2, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_3, k3_offset, k3_global, k3_local, &event_9, &event_10[0]);
				kernel_4.setArg(0, m_pre_gray_L3);
				kernel_4.setArg(1, m_pre_gray_L4);
				kernel_4.setArg(2, m_debugImg);
				queue.enqueueNDRangeKernel(kernel_4, k4_offset, k4_global, k4_local, &event_10, &event_11[0]);
				event_11[0].wait(); 
			} catch(cl::Error& e) { 
				slog.err("enqueue kernel | %s | %d", e.what(), e.err()); 
				if(event_2[0]() != NULL) slog.err("event_2: %d", event_2[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_3[0]() != NULL) slog.err("event_3: %d", event_3[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_4[0]() != NULL) slog.err("event_4: %d", event_4[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_5[0]() != NULL) slog.err("event_5: %d", event_5[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_6[0]() != NULL) slog.err("event_6: %d", event_6[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_7[0]() != NULL) slog.err("event_7: %d", event_7[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_8[0]() != NULL) slog.err("event_8: %d", event_8[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_9[0]() != NULL) slog.err("event_9: %d", event_9[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_10[0]() != NULL) slog.err("event_10: %d", event_10[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_11[0]() != NULL) slog.err("event_11: %d", event_11[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
                        postImageQueue.push(postImageMsg_t(m_debugImg));
			msgPre = msgIn;
                }
		slog.info("thread normalize exit.");
        });
	return 0;
/*=========================================================*/
}
int eis_exit(){
	slog_t slog("eis_exit");
	if(run_ == false){
		slog.war("pipline is stop already, do not exit again.");
		return 0;
	}
	run_ = false;

	eis_input_command(EIS_COMMAND_CLEAR_ALL);
	eis_output_command(EIS_COMMAND_CLEAR_ALL);

	loadImageQueue.discon();
	postImageQueue.discon();
	
	t_wrapImage.join();
	slog.info("t_wrapImage.join");

	return 0;
}
