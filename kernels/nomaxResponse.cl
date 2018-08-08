kernel void nomaxResponse(global const uchar * _img, global int* _kps, global int* _kps_o){
	int id0 = get_global_id(0);
	int id1 = get_global_id(1);
	global int* kps = _kps + (BLOCK_EST_MAXNUM + 1) * 2 * id1;

	if(id0 >= kps[0])
		return ;

	int2 point = vload2(id0+1, kps);
	/*====================================================================*/
	const ushort16 shuffle_table = (ushort16)(2,3,4,5,6,7,8,9,10,11,12,13,14,15,0,1);
	short s, s0, s1, s2, s3, s4, s5, s6, s7;
	#define CORNER_SCORE(S, D) { \
		short a0 = 0, b0 = 255; \
		for(int k=0;k<16;k+=2){ \
			short a = min(min(min(D.s1, D.s2), min(D.s3, D.s4)), min(min(D.s5, D.s6), min(D.s7, D.s8))); \
			short b = max(max(max(D.s1, D.s2), max(D.s3, D.s4)), max(max(D.s5, D.s6), max(D.s7, D.s8))); \
			a0 = max(a0, max(min(a, D.s0), min(a, D.s9))); \
			b0 = min(b0, min(max(b, D.s0), max(b, D.s9))); \
			D = shuffle(D, shuffle_table); \
		} \
		S = -min(convert_short(-a0), b0)-1; \
	}
	global const uchar* img = _img + point.y * GRAY_W + point.x - 4;
	/*====================================================================*/
	uchar8 p0 = vload8(0, img - 4 * GRAY_W);
	uchar8 p1 = vload8(0, img - 3 * GRAY_W);
	uchar8 p2 = vload8(0, img - 2 * GRAY_W);
	uchar8 p3 = vload8(0, img - 1 * GRAY_W);
	uchar8 p4 = vload8(0, img );
	uchar8 p5 = vload8(0, img + 1 * GRAY_W);
	uchar8 p6 = vload8(0, img + 2 * GRAY_W);
	uchar8 p7 = vload8(0, img + 3 * GRAY_W);

	short16 d  = convert_short16((uchar16)(p4.s7, p3.s7, p2.s6, p1.s5, p1.s4, p1.s3, p2.s2, p3.s1, p4.s1, p5.s1, p6.s2, p7.s3, p7.s4, p7.s5, p6.s6, p5.s7)) - convert_short(p4.s4);
	short16 d4 = convert_short16((uchar16)(p4.s6, p3.s6, p2.s5, p1.s4, p1.s3, p1.s2, p2.s1, p3.s0, p4.s0, p5.s0, p6.s1, p7.s2, p7.s3, p7.s4, p6.s5, p5.s6)) - convert_short(p4.s4);
	short16 d3 = convert_short16((uchar16)(p3.s6, p2.s6, p1.s5, p0.s4, p0.s3, p0.s2, p1.s1, p2.s0, p3.s0, p4.s0, p5.s1, p6.s2, p6.s3, p6.s4, p5.s5, p4.s6)) - convert_short(p4.s4);
	short16 d2 = convert_short16((uchar16)(p3.s7, p2.s7, p1.s6, p0.s5, p0.s4, p0.s3, p1.s2, p2.s1, p3.s1, p4.s1, p5.s2, p6.s3, p6.s4, p6.s5, p5.s6, p4.s7)) - convert_short(p4.s4);

	CORNER_SCORE(s,  d);
	CORNER_SCORE(s4, d4);
	CORNER_SCORE(s3, d3);
	CORNER_SCORE(s2, d2);
	if(s <= s2 || s <= s3 || s <= s4) return;
	/*====================================================================*/
	uchar8 p8 = vload8(0, img + 4 * GRAY_W);

	short16 d5 = convert_short16((uchar16)(p5.s6, p4.s6, p3.s5, p2.s4, p2.s3, p2.s2, p3.s1, p4.s0, p5.s0, p6.s0, p7.s1, p8.s2, p8.s3, p8.s4, p7.s5, p6.s6)) - convert_short(p4.s4);
	short16 d6 = convert_short16((uchar16)(p5.s7, p4.s7, p3.s6, p2.s5, p2.s4, p2.s3, p3.s2, p4.s1, p5.s1, p6.s1, p7.s2, p8.s3, p8.s4, p8.s5, p7.s6, p6.s7)) - convert_short(p4.s4);

	CORNER_SCORE(s5, d5);
	CORNER_SCORE(s6, d6);
	if(s <= s5 || s <= s6) return;
	/*====================================================================*/
	global const uchar* _imgi_1 = img + 8;
	uchar p0_s8 = _imgi_1[-4 * GRAY_W];
	uchar p1_s8 = _imgi_1[-3 * GRAY_W];
	uchar p2_s8 = _imgi_1[-2 * GRAY_W];
	uchar p3_s8 = _imgi_1[-1 * GRAY_W];
	uchar p4_s8 = _imgi_1[ 0 * GRAY_W];
	uchar p5_s8 = _imgi_1[ 1 * GRAY_W];
	uchar p6_s8 = _imgi_1[ 2 * GRAY_W];
	uchar p7_s8 = _imgi_1[ 3 * GRAY_W];
	uchar p8_s8 = _imgi_1[ 4 * GRAY_W];

	short16 d0 = convert_short16((uchar16)(p4_s8, p3_s8, p2.s7, p1.s6, p1.s5, p1.s4, p2.s3, p3.s2, p4.s2, p5.s2, p6.s3, p7.s4, p7.s5, p7.s6, p6.s7, p5_s8)) - convert_short(p4.s4);
	short16 d1 = convert_short16((uchar16)(p3_s8, p2_s8, p1.s7, p0.s6, p0.s5, p0.s4, p1.s3, p2.s2, p3.s2, p4.s2, p5.s3, p6.s4, p6.s5, p6.s6, p5.s7, p4_s8)) - convert_short(p4.s4);
	short16 d7 = convert_short16((uchar16)(p5_s8, p4_s8, p3.s7, p2.s6, p2.s5, p2.s4, p3.s3, p4.s2, p5.s2, p6.s2, p7.s3, p8.s4, p8.s5, p8.s6, p7.s7, p6_s8)) - convert_short(p4.s4);
	
	CORNER_SCORE(s0, d0);
	CORNER_SCORE(s1, d1);
	CORNER_SCORE(s7, d7);
	if(s <= s0 || s <= s1 || s <= s7) return;
	/*====================================================================*/
	{
		global int* kps_o = _kps_o + (BLOCK_EST_MAXNUM + 1) * 3 * id1;
		int num = atomic_inc(&kps_o[0]);
		vstore3((int3)(point, (int)s), num+1, kps_o);
	}
}
/*====================================================================*/
