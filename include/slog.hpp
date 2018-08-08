/*==============================================================================================*/
#ifndef __SLOG_H__
#define __SLOG_H__

#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <map>
#include <time.h>
#include <stdarg.h>
#include "unistd.h"
#include <sys/syscall.h>

using std::string;
using std::cout;
using std::endl;
using std::thread;
using std::mutex;
using std::unique_lock;
using std::map;

#ifndef _SLOG_FILE_
#define _SLOG_FILE_ "slog"
#endif

#define _SLOG_MAX_SLEN_ 256

//#define _SLOG_NO_INFO_ 
//#define _SLOG_NO_INFO_V0_ 

class slog_t{
private:
	static mutex _lock;
	static std::once_flag _flag; 
	static string fileName_i;
	static string fileName_w;
	static string fileName_e;
	static map<string, string> id_container;
	string mark_id;
	string mark_name;
public:
	slog_t(const char* name = NULL){
		unique_lock<mutex> lock(_lock);

		char* tmp = new char[64];
		sprintf(tmp, "[%ld/%ld]", syscall(SYS_getpid), syscall(SYS_gettid));
		mark_id = string(tmp);
		delete [] tmp;

		if(id_container.find(mark_id) != id_container.end()){
			mark_name = id_container[mark_id];
		} else {
			if(name == NULL){
				mark_name = "</temp>";
			} else {
				mark_name = string("<") + name + string(">");
			}
			id_container[mark_id] = mark_name;
		}

		std::call_once(_flag, init, name);
	}
	~slog_t(){
		unique_lock<mutex> lock(_lock);
		id_container.erase(mark_id);
	}
	static void init(const char* name){
		cout << "slog info: start." << endl;

		fileName_i = string(_SLOG_FILE_) + ".info";
		fileName_w = string(_SLOG_FILE_) + ".warning";
		fileName_e = string(_SLOG_FILE_) + ".error";

		std::ofstream f_i(fileName_i, std::ofstream::out | std::ofstream::trunc);
		std::ofstream f_w(fileName_w, std::ofstream::out | std::ofstream::trunc);
		std::ofstream f_e(fileName_e, std::ofstream::out | std::ofstream::trunc);
	}
	string get_time(){
		time_t t = time(0); 
		char tmp[64]; 
		strftime(tmp, sizeof(tmp), "[%Y/%m/%d %X]", localtime(&t)); 
		return string(tmp);
	}
	void info(const char* format, ...){
#ifndef _SLOG_NO_INFO_
		unique_lock<mutex> lock(_lock);
		char buffer[_SLOG_MAX_SLEN_];
		va_list args;
		va_start(args, format);
		vsnprintf(buffer, _SLOG_MAX_SLEN_-1, format, args);
		va_end(args);
		string s(buffer);
		std::ofstream f(fileName_i, std::ofstream::out | std::ofstream::app);
		f << get_time() << mark_id << "info: " << s << mark_name << endl;
		cout << get_time() << mark_id << "info: " << s << mark_name << endl;
#endif
	}
	void info_v0(const char* format, ...){
#ifndef _SLOG_NO_INFO_V0_
		unique_lock<mutex> lock(_lock);
		char buffer[_SLOG_MAX_SLEN_];
		va_list args;
		va_start(args, format);
		vsnprintf(buffer, _SLOG_MAX_SLEN_-1, format, args);
		va_end(args);
		string s(buffer);
		std::ofstream f(fileName_i, std::ofstream::out | std::ofstream::app);
		f << get_time() << mark_id << "info: " << s << mark_name << endl;
		cout << get_time() << mark_id << "info: " << s << mark_name << endl;
#endif
	}
	void war(const char* format, ...){
		unique_lock<mutex> lock(_lock);
		char buffer[_SLOG_MAX_SLEN_];
		va_list args;
		va_start(args, format);
		vsnprintf(buffer, _SLOG_MAX_SLEN_-1, format, args);
		va_end(args);
		string s(buffer);
		std::ofstream f(fileName_w, std::ofstream::out | std::ofstream::app);
		f << get_time() << mark_id << "warning: " << s << mark_name << endl;
		cout << get_time() << mark_id << "warning: " << s << mark_name << endl;
	}
	void err(const char* format, ...){
		unique_lock<mutex> lock(_lock);
		char buffer[_SLOG_MAX_SLEN_];
		va_list args;
		va_start(args, format);
		vsnprintf(buffer, _SLOG_MAX_SLEN_-1, format, args);
		va_end(args);
		string s(buffer);
		std::ofstream f(fileName_e, std::ofstream::out | std::ofstream::app);
		f << get_time() << mark_id << "error: " << s << mark_name << endl;
		cout << get_time() << mark_id << "error: " << s << mark_name << endl;
	}
};

mutex slog_t::_lock;
std::once_flag slog_t::_flag;
string slog_t::fileName_i;
string slog_t::fileName_w;
string slog_t::fileName_e;
map<string, string> slog_t::id_container;


#endif
