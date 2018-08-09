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
/*==========================================*/
#define EIS_COMMAND_MASK 0x0000FFFF
#define EIS_FLAG_MASK    0x000F0000
/*================= user care =========================*/

#define EIS_COMMAND_GET_BUFFER     0x00000000  // return value [negative: error, 0: got no buffer, 1: got one buffer]
#define EIS_COMMAND_RELEASE_BUFFER 0x00000001  // return value [negative: error, 0: buffer has been throwed away, 1: push one buffer]
#define EIS_COMMAND_CLEAR_ALL      0x00000002  // 
#define EIS_FLAG_NOLOCK            0x00010000  // 
/*==========================================*/
extern int eis_input_command(int command, void* ptr_buffer = NULL, void* ptr_image2d = NULL, void** pptr_buffer = NULL, void** pptr_image2d = NULL);
extern int eis_output_command(int command, void* ptr_image2d = NULL, void** pptr_image2d = NULL);
extern int eis_init();
extern int eis_exit();
