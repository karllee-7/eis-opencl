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
class keyPoint_t{
public:
        int x;
        int y;
        float response;
        keyPoint_t(int x=0, int y=0, float response=0.0f):x(x), y(y), response(response){}
};
class matches_t{
public:
        int x0;
        int y0;
        int x1;
        int y1;
        float response;
        matches_t(int x0=0, int y0=0, int x1=0, int y1=0, float response=0.0f):x0(x0), y0(y0), x1(x1), y1(y1), response(response){}
};
class matches1_t{
public:
        float x0;
        float y0;
        float x1;
        float y1;
        float response;
        matches1_t(float x0=0, float y0=0, float x1=0, float y1=0, float response=0.0f):x0(x0), y0(y0), x1(x1), y1(y1), response(response){}
};
class BFMatch_buff_t{
public:
        cl::Buffer m_kps;
        cl::Buffer m_desc;
        cl_int kps_num;
	bool valid;
        BFMatch_buff_t():kps_num(0), valid(false){}
        BFMatch_buff_t(cl::Buffer m_kps, cl::Buffer m_desc, cl_int kps_num, bool valid):
		m_kps(m_kps), 
		m_desc(m_desc), 
		kps_num(kps_num), 
		valid(valid){
        }
};
/*==============================================================================================*/
class loadImageMsg_t{
public:
        cl::Image2D m_img;
        cl::Buffer  m_img_aux;
        bool valid;
	loadImageMsg_t(): valid(false){}
};
class normalizeImageMsg_t{
public:
        cl::Image2D m_img;
        cl::Buffer  m_gray;
        cl::Image2D m_gauss;
        bool        valid;
	normalizeImageMsg_t(): valid(false){}
	normalizeImageMsg_t(cl::Image2D m_img, cl::Buffer m_gray, cl::Image2D m_gauss, bool valid): 
		m_img(m_img),
		m_gray(m_gray),
		m_gauss(m_gauss),
		valid(valid){
	}
};
class estKpsMsg_t{
public:
	cl::Image2D m_img;
        cl::Buffer  m_gray;
        cl::Image2D m_gauss;
        cl::Buffer  m_kps;
        bool        valid;
	estKpsMsg_t(): valid(false){}
	estKpsMsg_t(cl::Image2D m_img, cl::Buffer m_gray, cl::Image2D m_gauss, cl::Buffer m_kps, bool valid): 
		m_img(m_img),
		m_gray(m_gray),
		m_gauss(m_gauss),
		m_kps(m_kps), 
		valid(valid){
	}
};
class nomaxKpsMsg_t{
public:
	cl::Image2D m_img;
        cl::Buffer  m_gray;
        cl::Image2D m_gauss;
        cl::Buffer  m_kps;
        bool        valid;
	nomaxKpsMsg_t(): valid(false){}
	nomaxKpsMsg_t(cl::Image2D m_img, cl::Buffer m_gray, cl::Image2D m_gauss, cl::Buffer m_kps, bool valid): 
		m_img(m_img),
		m_gray(m_gray),
		m_gauss(m_gauss),
		m_kps(m_kps), 
		valid(valid){
	}
};
class bestKpsMsg_t{
public:
	cl::Image2D m_img;
        cl::Buffer  m_gray;
        cl::Image2D m_gauss;
        cl::Buffer  m_kps;
	cl_int      kps_num;
        bool        valid;
	bestKpsMsg_t(): kps_num(0), valid(false){}
	bestKpsMsg_t(cl::Image2D m_img, cl::Buffer m_gray, cl::Image2D m_gauss, cl::Buffer m_kps, cl_int kps_num, bool valid): 
		m_img(m_img),
		m_gray(m_gray),
		m_gauss(m_gauss),
		m_kps(m_kps), 
		kps_num(kps_num), 
		valid(valid){
	}
};
class BFMatchMsg_t{
public:
        cl::Image2D m_img;
        cl::Buffer  m_matches;
        cl_int      matches_num;
        bool        valid;
	BFMatchMsg_t(): matches_num(0), valid(false){}
	BFMatchMsg_t(cl::Image2D m_img, cl::Buffer m_matches, cl_int matches_num, bool valid): 
		m_img(m_img),
		m_matches(m_matches),
		matches_num(matches_num),
		valid(valid){
	}
};
class solveHMsg_t{
public:
        cl::Image2D m_img;
        cl::Buffer  m_matrixA;
        bool        valid;
	solveHMsg_t(): valid(false){}
	solveHMsg_t(cl::Image2D m_img, cl::Buffer m_matrixA, bool valid): m_img(m_img), m_matrixA(m_matrixA), valid(valid){}
};
class adjAMsg_t{
public:
        cl::Image2D m_img;
        cl::Buffer  m_matrixA;
        bool        valid;
	adjAMsg_t(): valid(false){}
	adjAMsg_t(cl::Image2D m_img, cl::Buffer m_matrixA, bool valid): m_img(m_img), m_matrixA(m_matrixA), valid(valid){}
};
class wrapImageMsg_t{
public:
        cl::Image2D m_img;
        bool        valid;
	wrapImageMsg_t(): valid(false){}
	wrapImageMsg_t(cl::Image2D m_img, bool valid): m_img(m_img), valid(valid){}
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

bool check_solveH_ransac(vector<int>& rnds){
	for(auto iter0 = rnds.begin(); iter0 != rnds.end(); iter0++){
		for(auto iter1 = iter0 + 1; iter1 != rnds.end(); iter1++){
			if(*iter0 == *iter1){
				return false;
			}
		}
	}
	return true;
}

bool check_in_area(MatrixXf A){
	MatrixXf p0(3,4);
	p0 << 0, IMAGE_IN_W, IMAGE_IN_W, 0,
	      0, 0,          IMAGE_IN_H, IMAGE_IN_H,
	      1, 1,          1,          1;
	static int max_distance = square((IMAGE_IN_W - IMAGE_OUT_W)/2) + square((IMAGE_IN_H - IMAGE_OUT_H)/2);
	MatrixXf p1 = A * p0;
	for(int i=0; i<p1.cols(); i++)
		p1.block<3,1>(0,i) = p1.block<3,1>(0,i) / p1(2,i);
	MatrixXf p_sub = p1.block<2,4>(0,0) - p0.block<2,4>(0,0);
	MatrixXf p_product = p_sub.cwiseProduct(p_sub);
	MatrixXf p_distance = p_product.colwise().sum();
	
	if( p_distance(0,0) <  max_distance && 
	    p_distance(0,1) <  max_distance && 
	    p_distance(0,2) <  max_distance && 
	    p_distance(0,3) <  max_distance)
		return true;

	return false;
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
static cl::ImageFormat gauss_format(CL_R, CL_UNORM_INT8);

thread t_normalizeImage;
thread t_estKps;
thread t_nomaxKps;
thread t_bestKps;
thread t_BFMatch;
thread t_solveH;
thread t_adjA;
thread t_wrapImage;

karl::msgQueue<loadImageMsg_t> loadImageQueue(3);
karl::msgQueue<normalizeImageMsg_t> normalizeImageQueue(2);
karl::msgQueue<estKpsMsg_t> estKpsQueue(2);
karl::msgQueue<nomaxKpsMsg_t> nomaxKpsQueue(2);
karl::msgQueue<bestKpsMsg_t> bestKpsQueue(2);
karl::msgQueue<BFMatchMsg_t> BFMatchQueue(2);
karl::msgQueue<solveHMsg_t> solveHQueue(2);
karl::msgQueue<adjAMsg_t> adjAQueue(2);
karl::msgQueue<wrapImageMsg_t> wrapImageQueue(2);
/*==============================================================================================*/
int eis_input_command(int command, void* ptr_buffer, void* ptr_image2d, void** pptr_buffer, void** pptr_image2d){
	static slog_t slog("eis_input_command");
	static cl::CommandQueue queue;
	static map<void*, cl::Buffer> buffer_container;
	static map<void*, cl::Image2D> image2d_container;
	static array<cl::size_type, 3> image_origin = {0, 0, 0};
	static array<cl::size_type, 3> image_region = {IMAGE_IN_W, IMAGE_IN_H, 1};
	static size_t image_row_pitch = 0;
	int cmd = command & EIS_COMMAND_MASK;
	int flg = command & EIS_FLAG_MASK;
	if(cmd == EIS_COMMAND_CLEAR_ALL){
		if(buffer_container.size() > 0){
			for(auto iter=buffer_container.begin(); iter!=buffer_container.end();iter++)
				queue.enqueueUnmapMemObject(iter->second, iter->first);
			buffer_container.erase(buffer_container.begin(), buffer_container.end());
		}
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
		if(pptr_buffer == NULL || pptr_image2d == NULL){
			slog.war("pptr_buffer | pptr_image2d cannot be NULL with command EIS_COMMAND_GET_BUFFER.");
			return -1;
		}
		if(buffer_container.size() > 0){
			slog.war("input buffer has been created. do not request again.");
			return -1;
		}
		if(image2d_container.size() > 0){
			slog.war("input image2d has been created. do not request again.");
			return -1;
		}
		cl::Buffer buffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*IMAGE_IN_W*IMAGE_IN_H*2);
		void* ptr0 = queue.enqueueMapBuffer(buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(uint8_t)*IMAGE_IN_W*IMAGE_IN_H*2);
		cl::Image2D image2d(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, image_format, IMAGE_IN_W, IMAGE_IN_H);
		void* ptr1 = queue.enqueueMapImage(image2d, CL_TRUE, CL_MAP_WRITE, image_origin, image_region, &image_row_pitch, NULL);
		buffer_container[ptr0] = buffer;
		image2d_container[ptr1] = image2d;
		*pptr_buffer = ptr0;
		*pptr_image2d = ptr1;
		return 1;
	}
	if(cmd == EIS_COMMAND_RELEASE_BUFFER){
		if(ptr_buffer == NULL){
			slog.war("buffer ptr is missing.");
			return -1;
		}
		if(ptr_image2d == NULL){
			slog.war("image2d ptr is missing.");
			return -1;
		}
		if(buffer_container.find(ptr_buffer) == buffer_container.end()){
			slog.war("no buffer found.");
			return -1;
		}
		if(image2d_container.find(ptr_image2d) == image2d_container.end()){
			slog.war("no image2d found.");
			return -1;
		}
		loadImageMsg_t msgOut;
		msgOut.m_img_aux = buffer_container[ptr_buffer];
		msgOut.m_img     = image2d_container[ptr_image2d];
		msgOut.valid = true;
		buffer_container.erase(ptr_buffer);
		image2d_container.erase(ptr_image2d);
		queue.enqueueUnmapMemObject(msgOut.m_img_aux, ptr_buffer);
		queue.enqueueUnmapMemObject(msgOut.m_img,     ptr_image2d);
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
		if(flg == EIS_FLAG_NOLOCK && wrapImageQueue.is_empty())
			return 0;
		auto msgIn = wrapImageQueue.pop();
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
	std::srand(1);
	slog_t slog("eis main");
	clBuildOptions \
		<< "-cl-no-signed-zeros "\
		<< "-DIMAGE_IN_W=" << IMAGE_IN_W << " " \
		<< "-DIMAGE_IN_H=" << IMAGE_IN_H << " " \
		<< "-DIMAGE_OUT_W=" << IMAGE_OUT_W << " " \
		<< "-DIMAGE_OUT_H=" << IMAGE_OUT_H << " " \
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
	cl_int cl_err = cl::Platform::get(&platforms);
	if(cl_err){
		slog.err("get platform error.");
		return -1;
	}

	cl_err = platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if(cl_err){
		slog.err("get device error.");
		return -1;
	}

	for(size_t i=0; i<devices.size(); i++) 
		slog.info("Device: %s Version: %s", devices[i].getInfo<CL_DEVICE_NAME>().c_str(), devices[i].getInfo<CL_DRIVER_VERSION>().c_str());

	context = cl::Context(devices, NULL, NULL, NULL, &cl_err);
	if(cl_err){
		slog.err("create context error.");
		return -1;
	}

	slog.info("creat context success.");
/*=========================================================*/
#if 0
	vector<cl::ImageFormat> deviceImageFormats;
	context.getSupportedImageFormats(CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, CL_MEM_OBJECT_IMAGE2D, &deviceImageFormats);
	for(auto elem : deviceImageFormats)
		printf("support image_channel_order: 0x%04x, image_channel_data_type: 0x%04x\n", elem.image_channel_order, elem.image_channel_data_type);
#endif
/*=========================================================*/
        t_normalizeImage = thread([](){
		slog_t slog("normalizeImage");
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		cl::Program program;
		string fileName = "kernels/cvtColor_gray.cl";
		try {
			program = createProgramWithFile(devices, context, clBuildOptions.str().c_str(), fileName.c_str());
		} catch (cl::BuildError& e){
			slog.err("buld program (%s) error. info: %s, log: %s", fileName.c_str(), e.what(), e.getBuildLog()[0].second.c_str());
			run_ = false;
			return ;
		}
		cl::Kernel kernel_0(program, "cvtColor_gray");
		cl::Kernel kernel_1(program, "cvtColor_gauss");

                cl::NDRange k0_offset(0, 0);
                cl::NDRange k0_global(GRAY_W/4, GRAY_H);
                cl::NDRange k0_local(16, 8);
                cl::NDRange k1_offset(0, 0);
                cl::NDRange k1_global(GRAY_W/3, GRAY_H);
                cl::NDRange k1_local(32, 8);

		cl::Sampler sampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_LINEAR);

		assert(GRAY_W % 4 == 0);
		assert(GRAY_W % 3 == 0);
		assert(k0_global.get()[0] % k0_local.get()[0] == 0);
		assert(k0_global.get()[1] % k0_local.get()[1] == 0);
		assert((k0_local.get()[0] * k0_local.get()[1] == 128) || (k0_local.get()[0] * k0_local.get()[1] == 256));
		assert(k1_global.get()[0] % k1_local.get()[0] == 0);
		assert(k1_global.get()[1] % k1_local.get()[1] == 0);
		assert((k1_local.get()[0] * k1_local.get()[1] == 128) || (k1_local.get()[0] * k1_local.get()[1] == 256));

#if _CKECK_TIME_
		timeval tv_s, tv_e;
		float time_min = std::numeric_limits<float>::max();
#endif
                slog.info("thread normalize Image start run.");
                while(run_){
			vector<cl::Event> event_0(1), event_1(1);
			cl::Buffer m_gray;
			cl::Image2D m_gauss;
			try{
				m_gray = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(uint8_t)*GRAY_W*GRAY_H);
				m_gauss = cl::Image2D(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, gauss_format, GRAY_W, GRAY_H);
			} catch(cl::Error& e) { 
				slog.err("malloc buffer: %s | %d", e.what(), e.err()); 
				run_ = false; 
				continue; 
			}

                        loadImageMsg_t msgIn = loadImageQueue.pop();
			if(!msgIn.valid){
				slog.war("msgIn.valid false.");
                        	normalizeImageQueue.push(normalizeImageMsg_t(msgIn.m_img, m_gray, m_gauss, false));
				continue;
			}
#if _CKECK_TIME_
			gettimeofday(&tv_s, NULL);
#endif
			try{ 
				kernel_0.setArg(0, msgIn.m_img_aux);
				kernel_0.setArg(1, m_gray);
				queue.enqueueNDRangeKernel(kernel_0, k0_offset, k0_global, k0_local, NULL, &event_0[0]);
				kernel_1.setArg(0, m_gray);
				kernel_1.setArg(1, m_gauss);
				kernel_1.setArg(2, sampler);
				queue.enqueueNDRangeKernel(kernel_1, k1_offset, k1_global, k1_local, &event_0, &event_1[0]);
				event_1[0].wait(); 
			} catch(cl::Error& e) { 
				slog.err("enqueue kernel | %s | %d", e.what(), e.err()); 
				if(event_0[0]() != NULL) slog.err("event_0: %d", event_0[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_1[0]() != NULL) slog.err("event_1: %d", event_1[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
#if 0
			{
				static array<cl::size_type, 3> image_origin = {0, 0, 0};
				static array<cl::size_type, 3> image_region = {GRAY_W, GRAY_H, 1};
				static size_t image_row_pitch = 0;
				vector<cl::Event> event_2(1), event_3(1);
				Mat img0(GRAY_H, GRAY_W, CV_8UC1);
				Mat img1(GRAY_H, GRAY_W, CV_8UC1);
				auto ptr0 = (uint8_t*)queue.enqueueMapBuffer(m_gray,  CL_TRUE, CL_MAP_READ,  0, sizeof(uint8_t)*GRAY_W*GRAY_H);
				auto ptr1 = (uint8_t*)queue.enqueueMapImage(m_gauss, CL_TRUE, CL_MAP_READ, image_origin, image_region, &image_row_pitch, NULL);
				for(int i=0; i<GRAY_H; i++)
					memcpy(img0.ptr<uint8_t>(i), ptr0 + i*GRAY_W, sizeof(uint8_t)*GRAY_W);
				for(int i=0; i<GRAY_H; i++)
					memcpy(img1.ptr<uint8_t>(i), ptr1 + i*GRAY_W, sizeof(uint8_t)*GRAY_W);
				queue.enqueueUnmapMemObject(m_gray,  ptr0, NULL,      &event_2[0]);
				queue.enqueueUnmapMemObject(m_gauss,  ptr1, &event_2, &event_3[0]);
				event_3[0].wait();
				imwrite("temp/0.bmp", img0);
				imwrite("temp/1.bmp", img1);
			}
#endif
#if _CKECK_TIME_
			gettimeofday(&tv_e, NULL);
			time_min = std::min(time_min, (float)(tv_e.tv_sec - tv_s.tv_sec) + (float)(tv_e.tv_usec - tv_s.tv_usec) * 1.0e-6f);
#endif
                        normalizeImageQueue.push(normalizeImageMsg_t(msgIn.m_img, m_gray, m_gauss, true));
                }
#if _CKECK_TIME_
                slog.info("thread normalize exit. min period %fs", time_min);
#else
		slog.info("thread normalize exit.");
#endif
        });
/*=========================================================*/
        t_estKps = thread([](){
		slog_t slog("estKps");
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		cl::Program program;
		string fileName = "kernels/estKps.cl";
		try {
			program = createProgramWithFile(devices, context, clBuildOptions.str().c_str(), fileName.c_str());
		} catch (cl::BuildError& e){
			slog.err("buld program (%s) error. info: %s, log: %s", fileName.c_str(), e.what(), e.getBuildLog()[0].second.c_str());
			run_ = false;
			return ;
		}
		cl::Kernel kernel_0(program, "estKps");

                cl::NDRange k0_local(16, 16, 1);
                cl::NDRange k0_global(std::floor((GRAY_W-64)/(k0_local.get()[0]*BLOCK_W))*k0_local.get()[0], std::floor((GRAY_H-64)/(k0_local.get()[1]*BLOCK_H))*k0_local.get()[1], BLOCK_NUM);
                cl::NDRange k0_offset(0, 0, 0);

		cl::Buffer m_offset(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*2*BLOCK_NUM);
		{
			int x0 = (GRAY_W - k0_global.get()[0] * BLOCK_W) / 2;
			int y0 = (GRAY_H - k0_global.get()[1] * BLOCK_H) / 2;
			auto ptr0 = (int*)queue.enqueueMapBuffer(m_offset, CL_TRUE, CL_MAP_WRITE, 0, sizeof(int)*2*BLOCK_NUM);
			for(int j=0;j<BLOCK_H;j++){
				for(int i=0;i<BLOCK_W;i++){
					int n=j*BLOCK_W+i;
					ptr0[n*2+0] = x0 + k0_global.get()[0] * i;
					ptr0[n*2+1] = y0 + k0_global.get()[1] * j;
				}
			}
			queue.enqueueUnmapMemObject(m_offset, ptr0);
		}
		assert(k0_global.get()[0] % k0_local.get()[0] == 0);
		assert(k0_global.get()[1] % k0_local.get()[1] == 0);

#if _CKECK_TIME_
		timeval tv_s, tv_e;
		float time_min = std::numeric_limits<float>::max();
#endif
		slog.info("thread findKps start run.");
		while(run_){
                        vector<cl::Event> event_0(1), event_1(1);
			cl::Buffer m_kps;
			try{
				m_kps = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*(BLOCK_EST_MAXNUM+1)*2*BLOCK_NUM);
				cl_clearBufferDistribute(queue, m_kps, 0, sizeof(int)*2, sizeof(int)*(BLOCK_EST_MAXNUM+1)*2, BLOCK_NUM, NULL, &event_0[0]);
			} catch (cl::Error& e) {
				slog.err("pre %s | %d", e.what(), e.err());
				if(event_0[0]() != NULL) slog.err("event_0: %d", event_0[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
                        auto msgIn = normalizeImageQueue.pop();
			if(!msgIn.valid){
				event_0[0].wait();
				estKpsQueue.push(estKpsMsg_t(msgIn.m_img, msgIn.m_gray, msgIn.m_gauss, m_kps, false));
				slog.war("msgIn.valid false.");
				continue;
			}
#if _CKECK_TIME_
			gettimeofday(&tv_s, NULL);
#endif
			try{
				kernel_0.setArg(0, msgIn.m_gray);
				kernel_0.setArg(1, m_offset);
				kernel_0.setArg(2, m_kps);
				queue.enqueueNDRangeKernel(kernel_0, k0_offset, k0_global, k0_local, &event_0, &event_1[0]);
			} catch(cl::Error& e) { 
				slog.err("enqueue kernel | %s | %d", e.what(), e.err()); 
				if(event_1[0]() != NULL) slog.err("event_1: %d", event_1[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
			event_1[0].wait();
#if _CKECK_TIME_
			gettimeofday(&tv_e, NULL);
			time_min = std::min(time_min, (float)(tv_e.tv_sec - tv_s.tv_sec) + (float)(tv_e.tv_usec - tv_s.tv_usec) * 1.0e-6f);
#endif
			estKpsQueue.push(estKpsMsg_t(msgIn.m_img, msgIn.m_gray, msgIn.m_gauss, m_kps, true));
		}
#if _CKECK_TIME_
		slog.info("thread estKps exit. min period %fs", time_min);
#else
		slog.info("thread estKps exit.");
#endif
		
	});
/*=========================================================*/
        t_nomaxKps = thread([](){
		slog_t slog("nomaxResponse");
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		cl::Program program;
		string fileName = "kernels/nomaxResponse.cl";
		try {
			program = createProgramWithFile(devices, context, clBuildOptions.str().c_str(), fileName.c_str());
		} catch (cl::BuildError& e){
			slog.err("buld program (%s) error. info: %s, log: %s", fileName.c_str(), e.what(), e.getBuildLog()[0].second.c_str());
			run_ = false;
			return ;
		}
		cl::Kernel kernel_0(program, "nomaxResponse");

		cl::Buffer m_nomaxKps(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*(BLOCK_EST_MAXNUM+1)*3*BLOCK_NUM);

                cl::NDRange k0_local(256, 1);
                cl::NDRange k0_global(BLOCK_EST_MAXNUM, BLOCK_NUM);
                cl::NDRange k0_offset(0, 0);

#if _CKECK_TIME_
		timeval tv_s, tv_e;
		float time_min = std::numeric_limits<float>::max();
#endif
		slog.info("thread findKps start run.");
		while(run_){
                        vector<cl::Event> event_0(1), event_1(1), event_2(1), event_3(1), event_4(1), event_5(1), event_6(1);
			cl::Buffer m_kps;
			try{
				m_kps = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*(BLOCK_NOMAX_MAXNUM+1)*2*BLOCK_NUM);
				cl_clearBufferDistribute(queue, m_nomaxKps, 0, sizeof(int)*3, sizeof(int)*(BLOCK_EST_MAXNUM+1)*3, BLOCK_NUM, NULL, &event_0[0]);
			} catch (cl::Error& e) {
				slog.err("pre %s | %d", e.what(), e.err());
				if(event_0[0]() != NULL) slog.err("event_0: %d", event_0[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
                        auto msgIn = estKpsQueue.pop();
			if(!msgIn.valid){
				event_0[0].wait();
				nomaxKpsQueue.push(nomaxKpsMsg_t(msgIn.m_img, msgIn.m_gray, msgIn.m_gauss, m_kps, false));
				slog.war("msgIn.valid false.");
				continue;
			}
#if _CKECK_TIME_
			gettimeofday(&tv_s, NULL);
#endif
			try{
				kernel_0.setArg(0, msgIn.m_gray);
				kernel_0.setArg(1, msgIn.m_kps);
				kernel_0.setArg(2, m_nomaxKps);
				queue.enqueueNDRangeKernel(kernel_0, k0_offset, k0_global, k0_local, &event_0, &event_1[0]);
			} catch(cl::Error& e) { 
				slog.err("enqueue kernel | %s | %d", e.what(), e.err()); 
				if(event_1[0]() != NULL) slog.err("event_1: %d", event_1[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
			try{
				auto ptr0 = (int*)queue.enqueueMapBuffer(m_nomaxKps,  CL_FALSE, CL_MAP_READ,  0, sizeof(int)*(BLOCK_EST_MAXNUM+1)*3*BLOCK_NUM, &event_1, &event_2[0]);
				auto ptr1 = (int*)queue.enqueueMapBuffer(m_kps, CL_FALSE, CL_MAP_WRITE, 0, sizeof(int)*(BLOCK_NOMAX_MAXNUM+1)*2*BLOCK_NUM, &event_2, &event_3[0]);
				event_3[0].wait();
				for(int j=0;j<BLOCK_NUM;j++){
					static vector<keyPoint_t> v_kps(BLOCK_EST_MAXNUM);
					v_kps.resize(0);
					int* offset0 = ptr0 + (BLOCK_EST_MAXNUM + 1) * 3 * j;
					int* offset1 = ptr1 + (BLOCK_NOMAX_MAXNUM + 1) * 2 * j;
					for(int i=0; i<fmin(offset0[0], BLOCK_EST_MAXNUM); i++){
						v_kps.emplace_back(offset0[(i+1)*3+0], offset0[(i+1)*3+1], (float)offset0[(i+1)*3+2]);
					}
					if(v_kps.size() > BLOCK_NOMAX_MAXNUM){
						nth_element(
							v_kps.begin(), 
							v_kps.begin()+BLOCK_NOMAX_MAXNUM, 
							v_kps.end(), 
							[](keyPoint_t& a, keyPoint_t& b){return a.response > b.response;});
						v_kps.erase(v_kps.begin()+BLOCK_NOMAX_MAXNUM, v_kps.end());
					}
					offset1[0] = v_kps.size();
					for(unsigned i=0; i<v_kps.size(); i++){
						offset1[(i+1)*2+0] = v_kps[i].x;
						offset1[(i+1)*2+1] = v_kps[i].y;
					}
					// slog.info("nomax block %d find kps num: %d", j, v_kps.size());
				}
				queue.enqueueUnmapMemObject(m_nomaxKps,  ptr0, NULL, &event_4[0]);
				queue.enqueueUnmapMemObject(m_kps, ptr1, &event_4, &event_5[0]);
				event_5[0].wait();
			} catch (cl::Error& e) {
				slog.err("sort response: %s | %d", e.what(), e.err());
				if(event_2[0]() != NULL) slog.err("event_2: %d", event_2[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_3[0]() != NULL) slog.err("event_3: %d", event_3[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_4[0]() != NULL) slog.err("event_4: %d", event_4[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_5[0]() != NULL) slog.err("event_5: %d", event_5[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
#if _CKECK_TIME_
			gettimeofday(&tv_e, NULL);
			time_min = std::min(time_min, (float)(tv_e.tv_sec - tv_s.tv_sec) + (float)(tv_e.tv_usec - tv_s.tv_usec) * 1.0e-6f);
#endif
			nomaxKpsQueue.push(nomaxKpsMsg_t(msgIn.m_img, msgIn.m_gray, msgIn.m_gauss, m_kps, true));
		}
#if _CKECK_TIME_
		slog.info("thread nomaxKps exit. min period %fs", time_min);
#else
		slog.info("thread nomaxKps exit.");
#endif
		
	});
/*=========================================================*/
        t_bestKps = thread([](){
		slog_t slog("best");
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		cl::Program program;
		string fileName = "kernels/computeResponses.cl";
		try {
			program = createProgramWithFile(devices, context, clBuildOptions.str().c_str(), fileName.c_str());
		} catch (cl::BuildError& e){
			slog.err("buld program (%s) error. info: %s, log: %s", fileName.c_str(), e.what(), e.getBuildLog()[0].second.c_str());
			run_ = false;
			return ;
		}
		cl::Kernel kernel_0(program, "computeResponses");

                cl::NDRange k0_offset(0, 0);
                cl::NDRange k0_global(BLOCK_NOMAX_MAXNUM, BLOCK_NUM);
                cl::NDRange k0_local(128, 2);
		assert(BLOCK_NOMAX_MAXNUM % k0_local.get()[0] == 0);
		assert(BLOCK_NUM % k0_local.get()[1] == 0);
		assert(KPS_MAXNUM >= BLOCK_KPS_MAXNUM*BLOCK_NUM);

		cl::Buffer m_response(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*(BLOCK_NOMAX_MAXNUM+1)*BLOCK_NUM);

#if _CKECK_TIME_
		timeval tv_s, tv_e;
		float time_min = std::numeric_limits<float>::max();
#endif
		slog.info("thread findKps start run.");
		while(run_){
                        vector<cl::Event> event_0(1), event_1(1), event_2(1), event_3(1), event_4(1), event_5(1), event_6(1);
			cl::Buffer m_kps;
			try{
				m_kps = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*KPS_MAXNUM*2);
			} catch (cl::Error& e) {
				slog.err("malloc m_kps %s | %d", e.what(), e.err());
				run_ = false; 
				continue; 
			}
                        auto msgIn = nomaxKpsQueue.pop();
			if(!msgIn.valid){
				bestKpsQueue.push(bestKpsMsg_t(msgIn.m_img, msgIn.m_gray, msgIn.m_gauss, m_kps, 0, false));
				slog.war("msgIn.valid false.");
				continue;
			}
#if _CKECK_TIME_
			gettimeofday(&tv_s, NULL);
#endif
			try{
				kernel_0.setArg(0, msgIn.m_gray);
				kernel_0.setArg(1, msgIn.m_kps);
				kernel_0.setArg(2, m_response);
				queue.enqueueNDRangeKernel(kernel_0, k0_offset, k0_global, k0_local, NULL, &event_0[0]);
			} catch(cl::Error& e) { 
				slog.err("enqueue kernel | %s | %d", e.what(), e.err()); 
				if(event_0[0]() != NULL) slog.err("event_0: %d", event_0[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
			static int kps_num = 0;
			try{
				auto ptr0 = (int*)  queue.enqueueMapBuffer(msgIn.m_kps,  CL_FALSE, CL_MAP_READ,  0, sizeof(int)*(BLOCK_NOMAX_MAXNUM+1)*2*BLOCK_NUM,   &event_0, &event_1[0]);
				auto ptr1 = (float*)queue.enqueueMapBuffer(m_response,   CL_FALSE, CL_MAP_READ,  0, sizeof(float)*(BLOCK_NOMAX_MAXNUM+1)*1*BLOCK_NUM, &event_1, &event_2[0]);
				auto ptr2 = (int*)  queue.enqueueMapBuffer(m_kps,        CL_FALSE, CL_MAP_WRITE, 0, sizeof(int)*KPS_MAXNUM*2,                         &event_2, &event_3[0]);
				event_3[0].wait();
				static vector<keyPoint_t> v_kps(BLOCK_NOMAX_MAXNUM);
				kps_num = 0;
				for(int j=0;j<BLOCK_NUM;j++){
					v_kps.resize(0);
					int* offset0 = ptr0 + (BLOCK_NOMAX_MAXNUM+1)*2*j;
					float* offset1 = ptr1 + (BLOCK_NOMAX_MAXNUM+1)*1*j;
					for(int i=0;i<std::min(offset0[0], BLOCK_NOMAX_MAXNUM);i++)
						v_kps.emplace_back(offset0[(i+1)*2+0], offset0[(i+1)*2+1], offset1[i+1]);
					if(v_kps.size() > BLOCK_KPS_MAXNUM){
						std::nth_element(v_kps.begin(), v_kps.begin()+BLOCK_KPS_MAXNUM, v_kps.end(), [](keyPoint_t& a, keyPoint_t& b){return a.response > b.response;});
						v_kps.erase(v_kps.begin()+BLOCK_KPS_MAXNUM, v_kps.end());
					}
					for(unsigned i=0; i<v_kps.size(); i++){
						ptr2[kps_num*2+0] = v_kps[i].x;
						ptr2[kps_num*2+1] = v_kps[i].y;
						kps_num++;
					}
				}
				slog.info("best kps num: %d", kps_num);
				queue.enqueueUnmapMemObject(msgIn.m_kps, ptr0, NULL,     &event_4[0]);
				queue.enqueueUnmapMemObject(m_response,  ptr1, &event_4, &event_5[0]);
				queue.enqueueUnmapMemObject(m_kps,       ptr2, &event_5, &event_6[0]);
				event_6[0].wait();
			} catch (cl::Error& e) {
				slog.err("sort response: %s | %d", e.what(), e.err());
				if(event_1[0]() != NULL) slog.err("event_1: %d", event_1[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_2[0]() != NULL) slog.err("event_2: %d", event_2[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_3[0]() != NULL) slog.err("event_3: %d", event_3[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_4[0]() != NULL) slog.err("event_4: %d", event_4[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_5[0]() != NULL) slog.err("event_5: %d", event_5[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_6[0]() != NULL) slog.err("event_6: %d", event_6[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
#if _CKECK_TIME_
			gettimeofday(&tv_e, NULL);
			time_min = std::min(time_min, (float)(tv_e.tv_sec - tv_s.tv_sec) + (float)(tv_e.tv_usec - tv_s.tv_usec) * 1.0e-6f);
#endif
			bestKpsQueue.push(bestKpsMsg_t(msgIn.m_img, msgIn.m_gray, msgIn.m_gauss, m_kps, kps_num, true));
		}
#if _CKECK_TIME_
		slog.info("thread computeKpsResponse exit. min period %fs", time_min);
#else
		slog.info("thread computeKpsResponse exit.");
#endif
	});
/*=========================================================*/
        t_BFMatch = thread([](){
		slog_t slog("match");
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		cl::Program program;
		string fileName = "kernels/BFMatch.cl";
		try {
			program = createProgramWithFile(devices, context, clBuildOptions.str().c_str(), fileName.c_str());
		} catch (cl::BuildError& e){
			slog.err("buld program (%s) error. info: %s, log: %s", fileName.c_str(), e.what(), e.getBuildLog()[0].second.c_str());
			run_ = false;
			return ;
		}
		cl::Kernel kernel_0(program, "ComputeAngle");
		cl::Kernel kernel_1(program, "ComputeDescriptor");
		cl::Kernel kernel_2(program, "BFMatch");

		cl::Buffer m_bitPattern(context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, sizeof(int8_t) * 4 * 256);
		cl::Buffer m_angle     (context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float) * KPS_MAXNUM * 2);
		cl::Buffer m_matches_a (context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * KPS_MAXNUM * 2);
		cl::Buffer m_matches_b (context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * KPS_MAXNUM * 2);

                cl::NDRange k0_offset(0);
                cl::NDRange k0_global(KPS_MAXNUM);
                cl::NDRange k0_local(256);
                cl::NDRange k1_offset(0, 0);
                cl::NDRange k1_global(8, KPS_MAXNUM);
                cl::NDRange k1_local(8, 32);
                cl::NDRange k2_offset(0, 0);
                cl::NDRange k2_global(KPS_MAXNUM, KPS_MAXNUM);
                cl::NDRange k2_local(256, 1);

		queue.enqueueWriteBuffer(m_bitPattern, CL_TRUE, 0, sizeof(char)*4*256, bit_pattern_31);

                BFMatch_buff_t buffer(
			cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * KPS_MAXNUM * 2),
			cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * KPS_MAXNUM),
			0, false);

		cl::Sampler sampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_LINEAR);
                vector<matches_t> v_matches(MATCHES_MAXNUM);

#if _CKECK_TIME_
		timeval tv_s, tv_e;
		float time_min = std::numeric_limits<float>::max();
#endif
                slog.info("thread BFMatch start run.");
		while(run_){
			vector<cl::Event> event_0(1), event_1(1), event_2(1), event_3(1), event_4(1), event_5(1), event_6(1), event_7(1);
			vector<cl::Event> event_8(1), event_9(1), event_10(1), event_11(1), event_12(1), event_13(1), event_14(1);
			cl::Buffer m_desc, m_matches;
			try{
				m_desc = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*KPS_MAXNUM*8);
				m_matches = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*KPS_MAXNUM*4);
			} catch (cl::Error& e) {
				slog.err("malloc m_desc %s | %d", e.what(), e.err());
				run_ = false; 
				continue; 
			}
			try{
				cl_fillBuffer(queue, m_matches_a, 0, 0, sizeof(int)*KPS_MAXNUM*2, NULL, &event_0[0]);
				cl_fillBuffer(queue, m_matches_b, 0, 0, sizeof(int)*KPS_MAXNUM*2, &event_0, &event_1[0]);
			} catch (cl::Error& e) {
				slog.err("cl_fillBuffer %s | %d", e.what(), e.err());
				if(event_0[0]() != NULL) slog.err("event_0: %d", event_0[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_1[0]() != NULL) slog.err("event_1: %d", event_1[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
                        auto msgIn = bestKpsQueue.pop();
			if(!msgIn.valid){
				event_1[0].wait();
				slog.war("BFMatch valid false.");
        			buffer = BFMatch_buff_t(msgIn.m_kps, m_desc, 0, false);
                        	BFMatchQueue.push(BFMatchMsg_t(msgIn.m_img, m_matches, 0, false));
				continue;
			}
#if _CKECK_TIME_
			gettimeofday(&tv_s, NULL);
#endif
			try{
				kernel_0.setArg(0, msgIn.m_gray);
				kernel_0.setArg(1, msgIn.m_kps);
				kernel_0.setArg(2, msgIn.kps_num);
				kernel_0.setArg(3, m_angle);
				queue.enqueueNDRangeKernel(kernel_0, k0_offset, k0_global, k0_local, &event_1, &event_2[0]);
				kernel_1.setArg(0, msgIn.m_gauss);
				kernel_1.setArg(1, sampler);
				kernel_1.setArg(2, m_bitPattern);
				kernel_1.setArg(3, msgIn.m_kps);
				kernel_1.setArg(4, msgIn.kps_num);
				kernel_1.setArg(5, m_angle);
				kernel_1.setArg(6, m_desc);
				queue.enqueueNDRangeKernel(kernel_1, k1_offset, k1_global, k1_local, &event_2, &event_3[0]);
				kernel_2.setArg(0, buffer.m_kps);
				kernel_2.setArg(1, buffer.m_desc);
				kernel_2.setArg(2, buffer.kps_num);
				kernel_2.setArg(3, msgIn.m_kps);
				kernel_2.setArg(4, m_desc);
				kernel_2.setArg(5, msgIn.kps_num);
				kernel_2.setArg(6, m_matches_a);
				kernel_2.setArg(7, m_matches_b);
				queue.enqueueNDRangeKernel(kernel_2, k2_offset, k2_global, k2_local, &event_3, &event_4[0]);
			} catch(cl::Error& e) { 
				slog.err("enqueue kernel | %s | %d", e.what(), e.err()); 
				if(event_2[0]() != NULL) slog.err("event_2: %d", event_2[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_3[0]() != NULL) slog.err("event_3: %d", event_3[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_4[0]() != NULL) slog.err("event_4: %d", event_4[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
			try{
				auto ptr0 = (int*)  queue.enqueueMapBuffer(m_matches_a,      CL_FALSE, CL_MAP_READ,  0, sizeof(int)*KPS_MAXNUM,         &event_4, &event_5[0]);
				auto ptr1 = (int*)  queue.enqueueMapBuffer(m_matches_b,      CL_FALSE, CL_MAP_READ,  0, sizeof(int)*KPS_MAXNUM,         &event_5, &event_6[0]);
				auto ptr2 = (int*)  queue.enqueueMapBuffer(buffer.m_kps,     CL_FALSE, CL_MAP_READ,  0, sizeof(int)*2*KPS_MAXNUM,       &event_6, &event_7[0]);
				auto ptr3 = (int*)  queue.enqueueMapBuffer(msgIn.m_kps,      CL_FALSE, CL_MAP_READ,  0, sizeof(int)*2*KPS_MAXNUM,       &event_7, &event_8[0]);
				auto ptr4 = (float*)queue.enqueueMapBuffer(m_matches,        CL_FALSE, CL_MAP_WRITE, 0, sizeof(float)*4*MATCHES_MAXNUM, &event_8, &event_9[0]);
				event_9[0].wait();

				v_matches.resize(0);
				for(int i=0;i<buffer.kps_num;i++){
					float response_a0 = (float)(ptr0[i*2+0] >> 16);
					float response_a1 = (float)(ptr0[i*2+1] >> 16);
					if(response_a0 <= (response_a1 * 2.0f))
						continue;
					int shoot_b = ptr0[i*2+0] & 0x0000FFFF;
					if(shoot_b >= msgIn.kps_num)
						continue;
					float response_b0 = (float)(ptr1[shoot_b*2+0] >> 16);
					float response_b1 = (float)(ptr1[shoot_b*2+1] >> 16);
					if(response_b0 <= (response_b1*2.0f))
						continue;
					int shoot_a = ptr1[shoot_b*2+0] & 0x0000FFFF;
					if(shoot_a != i)
						continue;
					if(square(ptr2[shoot_a*2+0]-ptr3[shoot_b*2+0])+square(ptr2[shoot_a*2+1]-ptr3[shoot_b*2+1]) > (MATCHES_DISTANCE*MATCHES_DISTANCE))
						continue;
					v_matches.emplace_back(ptr2[shoot_a*2+0], ptr2[shoot_a*2+1], ptr3[shoot_b*2+0], ptr3[shoot_b*2+1], std::min(response_a0, response_b0));
				}
				if(v_matches.size() > MATCHES_MAXNUM){
					nth_element(v_matches.begin(), v_matches.begin()+MATCHES_MAXNUM, v_matches.end(), [](matches_t& a, matches_t& b){ return a.response > b.response; });
					v_matches.erase(v_matches.begin()+MATCHES_MAXNUM, v_matches.end());
				}
				for(unsigned i=0; i<v_matches.size(); i++){
					ptr4[i*4+0] = v_matches[i].x0*2;
					ptr4[i*4+1] = v_matches[i].y0*2;
					ptr4[i*4+2] = v_matches[i].x1*2;
					ptr4[i*4+3] = v_matches[i].y1*2;
				}
				queue.enqueueUnmapMemObject(m_matches_a,      ptr0, NULL,      &event_10[0]);
				queue.enqueueUnmapMemObject(m_matches_b,      ptr1, &event_10, &event_11[0]);
				queue.enqueueUnmapMemObject(buffer.m_kps,     ptr2, &event_11, &event_12[0]);
				queue.enqueueUnmapMemObject(msgIn.m_kps,      ptr3, &event_12, &event_13[0]);
				queue.enqueueUnmapMemObject(m_matches, ptr4, &event_13, &event_14[0]);
				event_14[0].wait();
			} catch(cl::Error& e) { 
				slog.err("sort matches | %s | %d", e.what(), e.err()); 
				if(event_5[0]() != NULL) slog.err("event_5: %d", event_5[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_6[0]() != NULL) slog.err("event_6: %d", event_6[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_7[0]() != NULL) slog.err("event_7: %d", event_7[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_8[0]() != NULL) slog.err("event_8: %d", event_8[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_9[0]() != NULL) slog.err("event_9: %d", event_9[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_10[0]() != NULL) slog.err("event_10: %d", event_10[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_11[0]() != NULL) slog.err("event_11: %d", event_11[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_12[0]() != NULL) slog.err("event_12: %d", event_12[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_13[0]() != NULL) slog.err("event_13: %d", event_13[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_14[0]() != NULL) slog.err("event_14: %d", event_14[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
#ifdef _DRAW_KPS_
			{
				array<cl::size_type, 3> image_origin = {0, 0, 0};
				array<cl::size_type, 3> image_region = {IMAGE_IN_W, IMAGE_IN_H, 1};
				size_t image_row_pitch = 0;
				auto ptr3 = (uint8_t*)queue.enqueueMapImage(msgIn.m_img,        CL_TRUE, CL_MAP_READ, image_origin, image_region, &image_row_pitch, NULL);
				auto ptr4 = (int*)    queue.enqueueMapBuffer(msgIn.m_kps,       CL_TRUE, CL_MAP_READ, 0, sizeof(int)*KPS_MAXNUM*2);
				auto ptr6 = (float*)  queue.enqueueMapBuffer(m_angle,           CL_TRUE, CL_MAP_READ, 0, sizeof(float) * KPS_MAXNUM * 2);

				Mat img_0(IMAGE_IN_H, IMAGE_IN_W, CV_8UC4);
				Mat img_1(IMAGE_IN_H, IMAGE_IN_W, CV_8UC3);
				for(int row=0;row<IMAGE_IN_H;row++)
					memcpy(img_0.ptr<uint8_t>(row), ptr3 + row*IMAGE_IN_W*4, sizeof(uint8_t)*IMAGE_IN_W*4);
				cv::cvtColor(img_0, img_1, CV_RGBA2BGR);

				for(int i=0;i<msgIn.kps_num;i++){
					cv::Point p0(ptr4[i*2+0]*2, ptr4[i*2+1]*2);
					cv::Point p1(ptr4[i*2+0]*2 + 30 * ptr6[i*2+0], ptr4[i*2+1]*2 + 30 * ptr6[i*2+1]);
					cv::circle(img_1, p0, 30, cv::Scalar(255,0,0), 1);
					cv::circle(img_1, p0, 1, cv::Scalar(255,0,0), 1);
					cv::line(img_1, p0, p1, cv::Scalar(255,0,0), 2);
				}
				for(auto elem : v_matches){
					cv::Point p0(elem.x0*2, elem.y0*2);
					cv::Point p1(elem.x1*2, elem.y1*2);
					cv::circle(img_1, p0, 30, cv::Scalar(0,255,0), 2);
					cv::line(img_1, p0, p1, cv::Scalar(0,255,0), 2);
				}
				cv::cvtColor(img_1, img_0, CV_RGB2BGRA);
				for(int row=0;row<IMAGE_IN_H;row++)
					memcpy(ptr3 + row*IMAGE_IN_W*4, img_0.ptr<uint8_t>(row), sizeof(uint8_t)*IMAGE_IN_W*4);

				queue.enqueueUnmapMemObject(msgIn.m_img, ptr3);
				queue.enqueueUnmapMemObject(msgIn.m_kps, ptr4);
				queue.enqueueUnmapMemObject(m_angle, ptr6);
			}
#endif
			slog.info("matches num: %d", v_matches.size());
        		buffer = BFMatch_buff_t(msgIn.m_kps, m_desc, msgIn.kps_num, msgIn.valid);
#if _CKECK_TIME_
			gettimeofday(&tv_e, NULL);
			time_min = std::min(time_min, (float)(tv_e.tv_sec - tv_s.tv_sec) + (float)(tv_e.tv_usec - tv_s.tv_usec) * 1.0e-6f);
#endif
                        BFMatchQueue.push(BFMatchMsg_t(msgIn.m_img, m_matches, v_matches.size(), true));
		}
#if _CKECK_TIME_
		slog.info("thread BFMatch exit. min period %fs", time_min);
#else
		slog.info("thread BFMatch exit.");
#endif
	});
/*=========================================================*/
        t_solveH = thread([](){
		slog_t slog("solveH");
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		cl::Program program;
		string fileName = "kernels/solveH.cl";
		try {
			program = createProgramWithFile(devices, context, clBuildOptions.str().c_str(), fileName.c_str());
		} catch (cl::BuildError& e){
			slog.err("buld program (%s) error. info: %s, log: %s", fileName.c_str(), e.what(), e.getBuildLog()[0].second.c_str());
			run_ = false;
			return ;
		}
		cl::Kernel kernel_0(program, "solveH");

		cl::Buffer m_matrixA_ransac (context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*SOLVE_MATRIXA_RANSAC_NUM*16);
		cl::Buffer m_evaluate(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int)*SOLVE_MATRIXA_RANSAC_NUM*2);

                cl::NDRange k0_offset(0, 0);
                cl::NDRange k0_global(MATCHES_MAXNUM, SOLVE_MATRIXA_RANSAC_NUM);
                cl::NDRange k0_local(128, 2);

                MatrixXf T(3,3), T_inv(3,3);
                vector<matches1_t> v_matches(MATCHES_MAXNUM);
		vector<matches1_t> v_matches_inside(MATCHES_MAXNUM);
#ifdef _DRAW_KPS_
		vector<matches1_t> v_matches_inside1(MATCHES_MAXNUM);
#endif
		std::mt19937 random_e;

#if _CKECK_TIME_
		timeval tv_s, tv_e;
		float time_min = std::numeric_limits<float>::max();
#endif
		slog.info("thread solveH start run.");
		int frame_cnt = 0;
		while(run_){
			vector<cl::Event> event_0(1), event_1(1), event_2(1), event_3(1), event_4(1), event_5(1), event_6(1), event_7(1);
			vector<cl::Event> event_8(1), event_9(1), event_10(1), event_11(1), event_12(1), event_13(1);
                        cl::Buffer m_matrixA;
			try{
                        	m_matrixA = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*16);
			} catch (cl::Error& e) {
				slog.err("malloc m_matrixA: %s | %d", e.what(), e.err());
				run_ = false;
				continue; 
			}
			try {
				cl_fillBuffer(queue, m_evaluate, 0, 0, sizeof(int)*2*SOLVE_MATRIXA_RANSAC_NUM, NULL, &event_0[0]);
			} catch (cl::Error& e) {
				slog.err("cl_fillBuffer %s | %d", e.what(), e.err());
				if(event_0[0]() != NULL) slog.err("event_0: %d", event_0[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
                        auto msgIn = BFMatchQueue.pop();
			if(!msgIn.valid || msgIn.matches_num < SOLVE_MATRIXA_RANSAC_POINT_NUM){
				event_0[0].wait();
				solveHQueue.push(solveHMsg_t(msgIn.m_img, m_matrixA, false));
				slog.war("solveH valid false.");
				continue;
			}
#if _CKECK_TIME_
			gettimeofday(&tv_s, NULL);
#endif
                        try{
				auto ptr0 = (float*)queue.enqueueMapBuffer(msgIn.m_matches,  CL_FALSE, CL_MAP_READ,  0, sizeof(float)*4*MATCHES_MAXNUM,           &event_0, &event_1[0]);
				auto ptr1 = (float*)queue.enqueueMapBuffer(m_matrixA_ransac, CL_FALSE, CL_MAP_WRITE, 0, sizeof(float)*16*SOLVE_MATRIXA_RANSAC_NUM, &event_1, &event_2[0]);
				event_2[0].wait();

				std::uniform_int_distribution<unsigned> random_d(0, msgIn.matches_num-1);
				v_matches.resize(0);
				for(int i=0; i<msgIn.matches_num; i++){
					v_matches.emplace_back(ptr0[i*4+0], ptr0[i*4+1], ptr0[i*4+2], ptr0[i*4+3]);
				}
                                for(int ransc_cnt=0; ransc_cnt<SOLVE_MATRIXA_RANSAC_NUM; ransc_cnt++){
					static vector<int> rnds(4);
					for(int k=0;k<10;k++){
						rnds.resize(0);
						for(int m=0;m<4;m++)
							rnds.push_back(random_d(random_e));
						if(check_solveH_ransac(rnds))
							break;
					}
					static MatrixXf A(4,1);
					static MatrixXf B(4,1);
					for(int k=0; k<4; k++){
						int id = rnds[k];
						float x0 = v_matches[id].x0;
						float y0 = v_matches[id].y0;
						float x1 = v_matches[id].x1;
						float y1 = v_matches[id].y1;
						A.block<1,1>(k, 0) << x1 - x0;
						B.block<1,1>(k, 0) << y1 - y0;
					}
					float t0 = A.mean();
					float t1 = B.mean();
                                        ptr1[ransc_cnt*16+0] = 1;
                                        ptr1[ransc_cnt*16+1] = 0;
                                        ptr1[ransc_cnt*16+2] = t0;
                                        ptr1[ransc_cnt*16+3] = 0;
                                        ptr1[ransc_cnt*16+4] = 1;
                                        ptr1[ransc_cnt*16+5] = t1;
                                        ptr1[ransc_cnt*16+6] = 0;
                                        ptr1[ransc_cnt*16+7] = 0;
                                        ptr1[ransc_cnt*16+8] = 1;
                                };
				queue.enqueueUnmapMemObject(msgIn.m_matches,  ptr0, NULL,     &event_3[0]);
				queue.enqueueUnmapMemObject(m_matrixA_ransac, ptr1, &event_3, &event_4[0]);
			} catch (cl::Error& e) {
				slog.err("ransac take %s | %d", e.what(), e.err());
				if(event_1[0]() != NULL) slog.err("event_1: %d", event_1[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_2[0]() != NULL) slog.err("event_2: %d", event_2[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_3[0]() != NULL) slog.err("event_3: %d", event_3[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_4[0]() != NULL) slog.err("event_4: %d", event_4[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
			try{
                        	kernel_0.setArg(0, msgIn.m_matches);
                        	kernel_0.setArg(1, msgIn.matches_num);
                        	kernel_0.setArg(2, m_matrixA_ransac);
                        	kernel_0.setArg(3, m_evaluate);
				queue.enqueueNDRangeKernel(kernel_0, k0_offset, k0_global, k0_local, &event_4, &event_5[0]);
			} catch (cl::Error& e) {
				slog.err("kernel0 %s | %d", e.what(), e.err());
				if(event_5[0]() != NULL) slog.err("event_5: %d", event_5[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
                        try{
				auto ptr0 = (float*)queue.enqueueMapBuffer(m_matrixA_ransac, CL_FALSE, CL_MAP_WRITE,  0, sizeof(float)*16*SOLVE_MATRIXA_RANSAC_NUM, &event_5, &event_6[0]);
				auto ptr1 = (int*)  queue.enqueueMapBuffer(m_evaluate,       CL_FALSE, CL_MAP_WRITE,  0, sizeof(int)*2*SOLVE_MATRIXA_RANSAC_NUM,   &event_6, &event_7[0]);
				auto ptr2 = (float*)queue.enqueueMapBuffer(m_matrixA,        CL_FALSE, CL_MAP_WRITE,  0, sizeof(float)*16,                         &event_7, &event_8[0]);
				auto ptr3 = (float*)queue.enqueueMapBuffer(msgIn.m_matches,  CL_FALSE, CL_MAP_READ,   0, sizeof(float)*4*MATCHES_MAXNUM,           &event_8, &event_9[0]);
				event_9[0].wait();

				static vector<int> v_evaluate(SOLVE_MATRIXA_RANSAC_NUM);
				v_evaluate.resize(0);
                                for(int i=0;i<SOLVE_MATRIXA_RANSAC_NUM;i++)
					v_evaluate.emplace_back(ptr1[i]);
				auto evaluate_iter = std::max_element(v_evaluate.begin(), v_evaluate.end());
				int evaluate_id = distance(v_evaluate.begin(), evaluate_iter);
                                static MatrixXf h0(3,3), h1(3,3);
                                h0 << ptr0[evaluate_id*16+0], ptr0[evaluate_id*16+1], ptr0[evaluate_id*16+2],
                                      ptr0[evaluate_id*16+3], ptr0[evaluate_id*16+4], ptr0[evaluate_id*16+5],
                                      ptr0[evaluate_id*16+6], ptr0[evaluate_id*16+7], ptr0[evaluate_id*16+8];
				// cout << "aft index " << evaluate_id << " h" << endl << h0 << endl;

				v_matches_inside.resize(0);
#ifdef _DRAW_KPS_
				v_matches_inside1.resize(0);
#endif
				for(int i=0; i<msgIn.matches_num; i++){
					MatrixXf p0(3,1), p1(3,1), p1_(3,1);
					p0 << ptr3[i*4+0], ptr3[i*4+1], 1;
					p1 << ptr3[i*4+2], ptr3[i*4+3], 1;
					p1_ = h0 * p0;
					if((square(p1_(0,0)/p1_(2,0)-p1(0,0)) + square(p1_(1,0)/p1_(2,0)-p1(1,0))) < (SOLVE_MATRIXA_RANSAC_PRECISE*SOLVE_MATRIXA_RANSAC_PRECISE)){
						v_matches_inside.push_back(v_matches[i]);
#ifdef _DRAW_KPS_
						v_matches_inside1.emplace_back(p0(0,0), p0(1,0), p1(0,0), p1(1,0));
#endif
					}
				}
				slog.info("solve H matches %d, inside %d(%d)", msgIn.matches_num, v_matches_inside.size(), *evaluate_iter);
				float t0 = 0, t1 = 0;
				if(v_matches_inside.size() > 4){
					MatrixXf A(v_matches_inside.size(), 1);
					MatrixXf B(v_matches_inside.size(), 1);
					for(unsigned i=0; i<v_matches_inside.size(); i++){
						float x0 = v_matches_inside[i].x0;
						float y0 = v_matches_inside[i].y0;
						float x1 = v_matches_inside[i].x1;
						float y1 = v_matches_inside[i].y1;
						A.block<1,1>(i, 0) << x1 - x0;
						B.block<1,1>(i, 0) << y1 - y0;
					}
					t0 = A.mean();
					t1 = B.mean();
				}
#if 1
                                ptr2[0] = 1;
                                ptr2[1] = 0;
                                ptr2[2] = t0;
                                ptr2[3] = 0;
                                ptr2[4] = 1;
                                ptr2[5] = t1;
                                ptr2[6] = 0;
                                ptr2[7] = 0;
                                ptr2[8] = 1;
#else
                                ptr2[0] = h0(0,0);
                                ptr2[1] = h0(0,1);
                                ptr2[2] = h0(0,2);
                                ptr2[3] = h0(1,0);
                                ptr2[4] = h0(1,1);
                                ptr2[5] = h0(1,2);
                                ptr2[6] = h0(2,0);
                                ptr2[7] = h0(2,1);
                                ptr2[8] = h0(2,2);
#endif
				queue.enqueueUnmapMemObject(m_matrixA_ransac, ptr0, NULL,      &event_10[0]);
				queue.enqueueUnmapMemObject(m_evaluate,       ptr1, &event_10, &event_11[0]);
				queue.enqueueUnmapMemObject(m_matrixA,        ptr2, &event_11, &event_12[0]);
				queue.enqueueUnmapMemObject(msgIn.m_matches,  ptr3, &event_12, &event_13[0]);
				event_13[0].wait();
			} catch (cl::Error& e) {
				slog.err("sort evaluate %s | %d", e.what(), e.err());
				if(event_6[0]() != NULL) slog.err("event_6: %d", event_6[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_7[0]() != NULL) slog.err("event_7: %d", event_7[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_8[0]() != NULL) slog.err("event_8: %d", event_8[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_9[0]() != NULL) slog.err("event_9: %d", event_9[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_10[0]() != NULL) slog.err("event_10: %d", event_10[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_11[0]() != NULL) slog.err("event_11: %d", event_11[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
#ifdef _DRAW_KPS_
			{
				array<cl::size_type, 3> image_origin = {0, 0, 0};
				array<cl::size_type, 3> image_region = {IMAGE_IN_W, IMAGE_IN_H, 1};
				size_t image_row_pitch = 0;
				auto ptr0 = (uint8_t*)queue.enqueueMapImage(msgIn.m_img,        CL_TRUE, CL_MAP_READ, image_origin, image_region, &image_row_pitch, NULL);
				auto ptr1 = (float*)  queue.enqueueMapBuffer(msgIn.m_matches,   CL_TRUE, CL_MAP_READ, 0, sizeof(float)*MATCHES_MAXNUM*4);
				Mat img_0(IMAGE_IN_H, IMAGE_IN_W, CV_8UC4);
				Mat img_1(IMAGE_IN_H, IMAGE_IN_W, CV_8UC3);
				for(int row=0;row<IMAGE_IN_H;row++)
					memcpy(img_0.ptr<uint8_t>(row), ptr0 + row*IMAGE_IN_W*4, sizeof(uint8_t)*IMAGE_IN_W*4);
				cv::cvtColor(img_0, img_1, CV_RGBA2BGR);
				
				for(int i=0; i<msgIn.matches_num; i++){
					cv::Point p0(ptr1[i*4+0], ptr1[i*4+1]);
					cv::Point p1(ptr1[i*4+2], ptr1[i*4+3]);
					cv::circle(img_1, p1, 28, cv::Scalar(0,0,255), 2);
					cv::line(img_1, p0, p1, cv::Scalar(0,0,255), 1);
				}
				for(unsigned i=0; i<v_matches_inside1.size(); i++){
					cv::Point p0(v_matches_inside1[i].x0, v_matches_inside1[i].y0);
					cv::Point p1(v_matches_inside1[i].x1, v_matches_inside1[i].y1);
					cv::circle(img_1, p1, 25, cv::Scalar(0,255,255), 2);
					cv::line(img_1, p0, p1, cv::Scalar(0,255,255), 1);
				}
				cv::cvtColor(img_1, img_0, CV_RGB2BGRA);
				for(int row=0;row<IMAGE_IN_H;row++)
					memcpy(ptr0 + row*IMAGE_IN_W*4, img_0.ptr<uint8_t>(row), sizeof(uint8_t)*IMAGE_IN_W*4);

				queue.enqueueUnmapMemObject(msgIn.m_img, ptr0);
				queue.enqueueUnmapMemObject(msgIn.m_matches, ptr1);
			}
#endif
#if _CKECK_TIME_
			gettimeofday(&tv_e, NULL);
			time_min = std::min(time_min, (float)(tv_e.tv_sec - tv_s.tv_sec) + (float)(tv_e.tv_usec - tv_s.tv_usec) * 1.0e-6f);
#endif
			solveHQueue.push(solveHMsg_t(msgIn.m_img, m_matrixA, true));
			frame_cnt++;
		}
#if _CKECK_TIME_
		slog.info("thread solveH exit. min period %fs", time_min);
#else
		slog.info("thread solveH exit.");
#endif
	});
/*=========================================================*/
        t_adjA = thread([](){
		slog_t slog("adjA");
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
                Matrix<float,3,3> A_mac, A;
                A_mac << 1, 0, 0, 0, 1, 0, 0, 0, 1;
		// T << 1, 0, -(IMAGE_IN_W - IMAGE_OUT_W)/2, 0, 1, -(IMAGE_IN_H - IMAGE_OUT_H)/2, 0, 0, 1;

#if _CKECK_TIME_
		timeval tv_s, tv_e;
		float time_min = std::numeric_limits<float>::max();
#endif
		slog.info("thread adjA start run.");
		int frame_cnt = 0;
                while(run_){
			vector<cl::Event> event_0(1), event_1(1), event_2(1), event_3(1);
                        cl::Buffer m_matrixA;
			try{
                        	m_matrixA = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(float)*16);
			} catch (cl::Error& e) {
				slog.err("malloc m_matrixA: %s | %d", e.what(), e.err());
				run_ = false;
				continue; 
			}
                        auto msgIn = solveHQueue.pop();
#if _CKECK_TIME_
			gettimeofday(&tv_s, NULL);
#endif
                        try{
				if(msgIn.valid){
					auto ptr0 = (float*)queue.enqueueMapBuffer(msgIn.m_matrixA, CL_FALSE, CL_MAP_WRITE, 0, sizeof(float)*16, NULL, &event_0[0]);
					event_0[0].wait();
					A << ptr0[0], ptr0[1], ptr0[2], ptr0[3], ptr0[4], ptr0[5], ptr0[6], ptr0[7], ptr0[8];
					queue.enqueueUnmapMemObject(msgIn.m_matrixA, ptr0, NULL, &event_1[0]);
				}else{
					slog.war("no matrix A found.");
					A << 1, 0, 0, 0, 1, 0, 0, 0, 1;
				}
				A_mac = A_mac * A;
				A_mac = A_mac / std::sqrt((A_mac.transpose()*A_mac).trace());
				// cout << "a mac" << endl << A_mac << endl;
				auto ptr1 = (float*)queue.enqueueMapBuffer(m_matrixA, CL_FALSE, CL_MAP_WRITE, 0, sizeof(float)*16, msgIn.valid? &event_1 : NULL, &event_2[0]);
				for(float i=0.99; i>=0; i-=0.01){
					Matrix<float,3,3> a;
					a = (A_mac - MatrixXf::Identity(3,3)*A_mac(2,2)) * i + MatrixXf::Identity(3,3)*A_mac(2,2);
					// cout << "iter " << i << " a " << endl << a << endl;
					if(check_in_area(a)){
						A_mac = a;
						break;
					}
				}
				// cout << "a mac after" << endl << A_mac << endl;
				event_2[0].wait();
				ptr1[0] = A_mac(0,0);
				ptr1[1] = A_mac(0,1);
				ptr1[2] = A_mac(0,2);
				ptr1[3] = A_mac(1,0);
				ptr1[4] = A_mac(1,1);
				ptr1[5] = A_mac(1,2);
				ptr1[6] = A_mac(2,0);
				ptr1[7] = A_mac(2,1);
				ptr1[8] = A_mac(2,2);
				queue.enqueueUnmapMemObject(m_matrixA, ptr1, NULL, &event_3[0]);
				event_3[0].wait();
			} catch (cl::Error& e) {
				slog.err("m_matrixA: %s | %d", e.what(), e.err());
				if(event_0[0]() != NULL) slog.err("event_0: %d", event_0[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_1[0]() != NULL) slog.err("event_1: %d", event_1[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_2[0]() != NULL) slog.err("event_2: %d", event_2[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				if(event_3[0]() != NULL) slog.err("event_3: %d", event_3[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false;
				continue; 
			}
#if _CKECK_TIME_
			gettimeofday(&tv_e, NULL);
			time_min = std::min(time_min, (float)(tv_e.tv_sec - tv_s.tv_sec) + (float)(tv_e.tv_usec - tv_s.tv_usec) * 1.0e-6f);
#endif
                        adjAQueue.push(adjAMsg_t(msgIn.m_img, m_matrixA, true));
			frame_cnt++;
                }
#if _CKECK_TIME_
		slog.info("thread adjA exit. min period %fs", time_min);
#else
		slog.info("thread adjA exit.");
#endif
	});
/*=========================================================*/
        t_wrapImage = thread([](){
		slog_t slog("wrapImage");
		cl::CommandQueue queue(context, devices[0], CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
		cl::Program program;
		string fileName = "kernels/wrapImage.cl";
		try {
			program = createProgramWithFile(devices, context, clBuildOptions.str().c_str(), fileName.c_str());
		} catch (cl::BuildError& e){
			slog.err("buld program (%s) error. info: %s, log: %s", fileName.c_str(), e.what(), e.getBuildLog()[0].second.c_str());
			run_ = false;
			return ;
		}
		cl::Kernel kernel_0(program, "wrapImage");
		cl::Sampler sampler(context, CL_FALSE, CL_ADDRESS_CLAMP, CL_FILTER_LINEAR);

                cl::NDRange k0_offset(0, 0);
                cl::NDRange k0_global(IMAGE_OUT_W/15, IMAGE_OUT_H);
                cl::NDRange k0_local(32, 8);
		
		assert(IMAGE_OUT_W % 15 == 0);
		assert(k0_global.get()[0] % k0_local.get()[0] == 0);
		assert(k0_global.get()[1] % k0_local.get()[1] == 0);

#if _CKECK_TIME_
		timeval tv_s, tv_e;
		float time_min = std::numeric_limits<float>::max();
#endif
		slog.info("thread wrap image start run.");
                while(run_){
			vector<cl::Event> event_0(1), event_1(1);
			cl::Image2D m_img;
			try{
				m_img = cl::Image2D(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, image_format, IMAGE_OUT_W, IMAGE_OUT_H);
			} catch (cl::Error& e) {
				slog.err("malloc m_img: %s | %d", e.what(), e.err());
				run_ = false;
				continue; 
			}
                        auto msgIn = adjAQueue.pop();
			if(!msgIn.valid){
				slog.war("wrapImage valid false.");
				continue;
			}
#if _CKECK_TIME_
			gettimeofday(&tv_s, NULL);
#endif
			try{
                        	kernel_0.setArg(0, msgIn.m_img);
                        	kernel_0.setArg(1, m_img);
                        	kernel_0.setArg(2, sampler);
                        	kernel_0.setArg(3, msgIn.m_matrixA);
				queue.enqueueNDRangeKernel(kernel_0, k0_offset, k0_global, k0_local, NULL, &event_0[0]);
				event_0[0].wait();
			} catch (cl::Error& e) {
				slog.err("kernel0 %s | %d", e.what(), e.err());
				if(event_0[0]() != NULL) slog.err("event_0: %d", event_0[0].getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>());
				run_ = false; 
				continue; 
			}
#if _CKECK_TIME_
			gettimeofday(&tv_e, NULL);
			time_min = std::min(time_min, (float)(tv_e.tv_sec - tv_s.tv_sec) + (float)(tv_e.tv_usec - tv_s.tv_usec) * 1.0e-6f);
#endif
			wrapImageQueue.push(wrapImageMsg_t(m_img, true));
                }
#if _CKECK_TIME_
		slog.info("thread wrap image exit. min period %fs", time_min);
#else
		slog.info("thread wrap image exit.");
#endif
        });
	slog.info("eis init complete.");
	return 0;
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
	normalizeImageQueue.discon();
	estKpsQueue.discon();
	nomaxKpsQueue.discon();
	bestKpsQueue.discon();
	BFMatchQueue.discon();
	solveHQueue.discon();
	adjAQueue.discon();
	wrapImageQueue.discon();
	
	// if(t_normalizeImage.joinable())
		t_normalizeImage.join();
	slog.info("t_normalizeImage.join");
	// if(t_estKps.joinable())
		t_estKps.join();
	slog.info("t_estKps.join");
	// if(t_nomaxKps.joinable())
		t_nomaxKps.join();
	slog.info("t_nomaxKps.join");
	// if(t_bestKps.joinable())
		t_bestKps.join();
	slog.info("t_bestKps.join");
	// if(t_BFMatch.joinable())
		t_BFMatch.join();
	slog.info("t_BFMatch.join");
	// if(t_solveH.joinable())
		t_solveH.join();
	slog.info("t_solveH.join");
	// if(t_adjA.joinable())
		t_adjA.join();
	slog.info("t_adjA.join");
	// if(t_wrapImage.joinable())
		t_wrapImage.join();
	slog.info("t_wrapImage.join");

	slog.info("eis pipline clean.");

	return 0;
}
