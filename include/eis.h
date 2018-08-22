/*==============================*/
#define IMAGE_IN_W 2688
#define IMAGE_IN_H 1520
#define IMAGE_OUT_W 1920
#define IMAGE_OUT_H 1080

#define BLOCK_W 4
#define BLOCK_H 3
#define BLOCK_EST_MAXNUM 1024
#define BLOCK_NOMAX_MAXNUM 128
#define BLOCK_KPS_MAXNUM 42
#define KPS_MAXNUM 512

#define MATCHES_MAXNUM 256
#define MATCHES_DISTANCE 256

// #define SOLVE_MATRIXF_RANSAC_NUM 256
// #define SOLVE_MATRIXF_RANSAC_POINT_NUM 8
// #define SOLVE_MATRIXF_RANSAC_PRECISE 1

#define solveH_8p 1
#define solveH_3p 0
#define solveH_4p 0
#define solveH_2p 0

#define SOLVE_MATRIXA_RANSAC_NUM 256
#define SOLVE_MATRIXA_RANSAC_POINT_NUM 6
#define SOLVE_MATRIXA_RANSAC_PRECISE 1.5
/*==========================================*/
#define BLOCK_NUM (BLOCK_W*BLOCK_H)
#define GRAY_W (IMAGE_IN_W/2)
#define GRAY_H (IMAGE_IN_H/2)
#define KPS_EST_THRESHOLD 30

#define MATCHES_MAXNUM_PRE ( MATCHES_MAXNUM * 2 )

#define CAM_PAM_F 2150
/*==========================================*/
#define EIS_COMMAND_MASK 0x0000FFFF
#define EIS_FLAG_MASK    0x000F0000
/*================= user care =========================*/

#define EIS_COMMAND_GET_BUFFER     0x00000000  // return value [negative: error, 0: got no buffer, 1: got one buffer]
#define EIS_COMMAND_RELEASE_BUFFER 0x00000001  // return value [negative: error, 0: buffer has been throwed away, 1: push one buffer]
#define EIS_COMMAND_CLEAR_ALL      0x00000002  // 
#define EIS_FLAG_NOLOCK            0x00010000  // 
/*==========================================*/
class quaternion_cls{
public:
	double q0;
	double q1;
	double q2;
	double q3;
	quaternion_cls(double q0, double q1, double q2, double q3):
		q0(q0), q1(q1), q2(q2), q3(q3){};
	quaternion_cls():
		q0(0.0), q1(0.0), q2(0.0), q3(0.0){};
	quaternion_cls(const quaternion_cls& a){
		q0 = a.q0;
		q1 = a.q1;
		q2 = a.q2;
		q3 = a.q3;
	}
	void operator=(const quaternion_cls& a){
		q0 = a.q0;
		q1 = a.q1;
		q2 = a.q2;
		q3 = a.q3;
	}
	double dot(const quaternion_cls& a){
		return q0*a.q0 + q1*a.q1 + q2*a.q2 + q3*a.q3;
	}
	quaternion_cls inv(){
		quaternion_cls b;
		double a = 1.0/(q0*q0 + q1*q1 + q2*q2 + q3*q3);
		b.q0 = +q0 * a;
		b.q1 = -q1 * a;
		b.q2 = -q2 * a;
		b.q3 = -q3 * a;
		return b;
	}
	double get_q0()const { return q0; }
	double get_q1()const { return q1; }
	double get_q2()const { return q2; }
	double get_q3()const { return q3; }
	template <typename T>  
	void set_q0(T a){ q0 = a; }
	template <typename T>  
	void set_q1(T a){ q1 = a; }
	template <typename T>  
	void set_q2(T a){ q2 = a; }
	template <typename T>  
	void set_q3(T a){ q3 = a; }
	quaternion_cls operator*(const quaternion_cls& a){
		quaternion_cls r;
		r.q0 = this->q0*a.q0 - this->q1*a.q1 - this->q2*a.q2 - this->q3*a.q3;
		r.q1 = this->q0*a.q1 + this->q1*a.q0 + this->q3*a.q2 - this->q2*a.q3;
		r.q2 = this->q0*a.q2 + this->q2*a.q0 + this->q1*a.q3 - this->q3*a.q1;
		r.q3 = this->q0*a.q3 + this->q3*a.q0 + this->q2*a.q1 - this->q1*a.q2;
		double b = 1.0 / (r.q0*r.q0 + r.q1*r.q1 + r.q2*r.q2 + r.q3*r.q3);
		r.q0 = r.q0 * b;
		r.q1 = r.q1 * b;
		r.q2 = r.q2 * b;
		r.q3 = r.q3 * b;
		return r;
	}
};

extern int eis_input_command(int command, void* ptr_image2d = NULL, quaternion_cls *pos = NULL, void** pptr_image2d = NULL);
extern int eis_output_command(int command, void* ptr_image2d = NULL, void** pptr_image2d = NULL);
extern int eis_init();
extern int eis_exit();
