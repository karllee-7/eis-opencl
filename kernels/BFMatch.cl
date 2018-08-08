#define VSUM16(X) (X.s0 + X.s1 + X.s2 + X.s3 + X.s4 + X.s5 + X.s6 + X.s7 + X.s8 + X.s9 + X.sa + X.sb + X.sc + X.sd + X.se + X.sf)
#define VSUM8(X) (X.s0 + X.s1 + X.s2 + X.s3 + X.s4 + X.s5 + X.s6 + X.s7)

kernel void ComputeAngle(global uchar * _img, global int* kps, int kps_num, global float * angle){
	int id = get_global_id(0);

	if(id >= kps_num)
		return ;

	int2 point = vload2(id, kps);
	global uchar* img = _img + point.y * GRAY_W + point.x;
	/*===================================*/
	uchar16 pl_n15 = vload16(0, img - 15 * GRAY_W - 15);
	uchar16 pl_n14 = vload16(0, img - 14 * GRAY_W - 15);
	uchar16 pl_n13 = vload16(0, img - 13 * GRAY_W - 15);
	uchar16 pl_n12 = vload16(0, img - 12 * GRAY_W - 15);
	uchar16 pl_n11 = vload16(0, img - 11 * GRAY_W - 15);
	uchar16 pl_n10 = vload16(0, img - 10 * GRAY_W - 15);
	uchar16 pl_n09 = vload16(0, img - 9  * GRAY_W - 15);
	uchar16 pl_n08 = vload16(0, img - 8  * GRAY_W - 15);
	uchar16 pl_n07 = vload16(0, img - 7  * GRAY_W - 15);
	uchar16 pl_n06 = vload16(0, img - 6  * GRAY_W - 15);
	uchar16 pl_n05 = vload16(0, img - 5  * GRAY_W - 15);
	uchar16 pl_n04 = vload16(0, img - 4  * GRAY_W - 15);
	uchar16 pl_n03 = vload16(0, img - 3  * GRAY_W - 15);
	uchar16 pl_n02 = vload16(0, img - 2  * GRAY_W - 15);
	uchar16 pl_n01 = vload16(0, img - 1  * GRAY_W - 15);
	uchar16 pl_p00 = vload16(0, img + 0  * GRAY_W - 15);
	uchar16 pl_p01 = vload16(0, img + 1  * GRAY_W - 15);
	uchar16 pl_p02 = vload16(0, img + 2  * GRAY_W - 15);
	uchar16 pl_p03 = vload16(0, img + 3  * GRAY_W - 15);
	uchar16 pl_p04 = vload16(0, img + 4  * GRAY_W - 15);
	uchar16 pl_p05 = vload16(0, img + 5  * GRAY_W - 15);
	uchar16 pl_p06 = vload16(0, img + 6  * GRAY_W - 15);
	uchar16 pl_p07 = vload16(0, img + 7  * GRAY_W - 15);
	uchar16 pl_p08 = vload16(0, img + 8  * GRAY_W - 15);
	uchar16 pl_p09 = vload16(0, img + 9  * GRAY_W - 15);
	uchar16 pl_p10 = vload16(0, img + 10 * GRAY_W - 15);
	uchar16 pl_p11 = vload16(0, img + 11 * GRAY_W - 15);
	uchar16 pl_p12 = vload16(0, img + 12 * GRAY_W - 15);
	uchar16 pl_p13 = vload16(0, img + 13 * GRAY_W - 15);
	uchar16 pl_p14 = vload16(0, img + 14 * GRAY_W - 15);
	uchar16 pl_p15 = vload16(0, img + 15 * GRAY_W - 15);
	
	uchar16 pr_n15 = vload16(0, img - 15 * GRAY_W);
	uchar16 pr_n14 = vload16(0, img - 14 * GRAY_W);
	uchar16 pr_n13 = vload16(0, img - 13 * GRAY_W);
	uchar16 pr_n12 = vload16(0, img - 12 * GRAY_W);
	uchar16 pr_n11 = vload16(0, img - 11 * GRAY_W);
	uchar16 pr_n10 = vload16(0, img - 10 * GRAY_W);
	uchar16 pr_n09 = vload16(0, img - 9  * GRAY_W);
	uchar16 pr_n08 = vload16(0, img - 8  * GRAY_W);
	uchar16 pr_n07 = vload16(0, img - 7  * GRAY_W);
	uchar16 pr_n06 = vload16(0, img - 6  * GRAY_W);
	uchar16 pr_n05 = vload16(0, img - 5  * GRAY_W);
	uchar16 pr_n04 = vload16(0, img - 4  * GRAY_W);
	uchar16 pr_n03 = vload16(0, img - 3  * GRAY_W);
	uchar16 pr_n02 = vload16(0, img - 2  * GRAY_W);
	uchar16 pr_n01 = vload16(0, img - 1  * GRAY_W);
	uchar16 pr_p00 = vload16(0, img + 0  * GRAY_W);
	uchar16 pr_p01 = vload16(0, img + 1  * GRAY_W);
	uchar16 pr_p02 = vload16(0, img + 2  * GRAY_W);
	uchar16 pr_p03 = vload16(0, img + 3  * GRAY_W);
	uchar16 pr_p04 = vload16(0, img + 4  * GRAY_W);
	uchar16 pr_p05 = vload16(0, img + 5  * GRAY_W);
	uchar16 pr_p06 = vload16(0, img + 6  * GRAY_W);
	uchar16 pr_p07 = vload16(0, img + 7  * GRAY_W);
	uchar16 pr_p08 = vload16(0, img + 8  * GRAY_W);
	uchar16 pr_p09 = vload16(0, img + 9  * GRAY_W);
	uchar16 pr_p10 = vload16(0, img + 10 * GRAY_W);
	uchar16 pr_p11 = vload16(0, img + 11 * GRAY_W);
	uchar16 pr_p12 = vload16(0, img + 12 * GRAY_W);
	uchar16 pr_p13 = vload16(0, img + 13 * GRAY_W);
	uchar16 pr_p14 = vload16(0, img + 14 * GRAY_W);
	uchar16 pr_p15 = vload16(0, img + 15 * GRAY_W);

	short8 mx_15_lo = convert_short8(pr_p15.lo) + convert_short8(pr_n15.lo) - convert_short8(pl_p15.sfedcba98) - convert_short8(pl_n15.sfedcba98);
	short8 mx_14_lo = convert_short8(pr_p14.lo) + convert_short8(pr_n14.lo) - convert_short8(pl_p14.sfedcba98) - convert_short8(pl_n14.sfedcba98);
	short8 mx_13_lo = convert_short8(pr_p13.lo) + convert_short8(pr_n13.lo) - convert_short8(pl_p13.sfedcba98) - convert_short8(pl_n13.sfedcba98);
	short8 mx_12_lo = convert_short8(pr_p12.lo) + convert_short8(pr_n12.lo) - convert_short8(pl_p12.sfedcba98) - convert_short8(pl_n12.sfedcba98);
	short8 mx_11_lo = convert_short8(pr_p11.lo) + convert_short8(pr_n11.lo) - convert_short8(pl_p11.sfedcba98) - convert_short8(pl_n11.sfedcba98);
	short8 mx_10_lo = convert_short8(pr_p10.lo) + convert_short8(pr_n10.lo) - convert_short8(pl_p10.sfedcba98) - convert_short8(pl_n10.sfedcba98);
	short8 mx_09_lo = convert_short8(pr_p09.lo) + convert_short8(pr_n09.lo) - convert_short8(pl_p09.sfedcba98) - convert_short8(pl_n09.sfedcba98);
	short8 mx_08_lo = convert_short8(pr_p08.lo) + convert_short8(pr_n08.lo) - convert_short8(pl_p08.sfedcba98) - convert_short8(pl_n08.sfedcba98);
	short8 mx_07_lo = convert_short8(pr_p07.lo) + convert_short8(pr_n07.lo) - convert_short8(pl_p07.sfedcba98) - convert_short8(pl_n07.sfedcba98);
	short8 mx_06_lo = convert_short8(pr_p06.lo) + convert_short8(pr_n06.lo) - convert_short8(pl_p06.sfedcba98) - convert_short8(pl_n06.sfedcba98);
	short8 mx_05_lo = convert_short8(pr_p05.lo) + convert_short8(pr_n05.lo) - convert_short8(pl_p05.sfedcba98) - convert_short8(pl_n05.sfedcba98);
	short8 mx_04_lo = convert_short8(pr_p04.lo) + convert_short8(pr_n04.lo) - convert_short8(pl_p04.sfedcba98) - convert_short8(pl_n04.sfedcba98);
	short8 mx_03_lo = convert_short8(pr_p03.lo) + convert_short8(pr_n03.lo) - convert_short8(pl_p03.sfedcba98) - convert_short8(pl_n03.sfedcba98);
	short8 mx_02_lo = convert_short8(pr_p02.lo) + convert_short8(pr_n02.lo) - convert_short8(pl_p02.sfedcba98) - convert_short8(pl_n02.sfedcba98);
	short8 mx_01_lo = convert_short8(pr_p01.lo) + convert_short8(pr_n01.lo) - convert_short8(pl_p01.sfedcba98) - convert_short8(pl_n01.sfedcba98);
	short8 mx_00_lo = convert_short8(pr_p00.lo)                             - convert_short8(pl_p00.sfedcba98);

	short8 mx_15_hi = convert_short8(pr_p15.hi) + convert_short8(pr_n15.hi) - convert_short8(pl_p15.s76543210) - convert_short8(pl_n15.s76543210);
	short8 mx_14_hi = convert_short8(pr_p14.hi) + convert_short8(pr_n14.hi) - convert_short8(pl_p14.s76543210) - convert_short8(pl_n14.s76543210);
	short8 mx_13_hi = convert_short8(pr_p13.hi) + convert_short8(pr_n13.hi) - convert_short8(pl_p13.s76543210) - convert_short8(pl_n13.s76543210);
	short8 mx_12_hi = convert_short8(pr_p12.hi) + convert_short8(pr_n12.hi) - convert_short8(pl_p12.s76543210) - convert_short8(pl_n12.s76543210);
	short8 mx_11_hi = convert_short8(pr_p11.hi) + convert_short8(pr_n11.hi) - convert_short8(pl_p11.s76543210) - convert_short8(pl_n11.s76543210);
	short8 mx_10_hi = convert_short8(pr_p10.hi) + convert_short8(pr_n10.hi) - convert_short8(pl_p10.s76543210) - convert_short8(pl_n10.s76543210);
	short8 mx_09_hi = convert_short8(pr_p09.hi) + convert_short8(pr_n09.hi) - convert_short8(pl_p09.s76543210) - convert_short8(pl_n09.s76543210);
	short8 mx_08_hi = convert_short8(pr_p08.hi) + convert_short8(pr_n08.hi) - convert_short8(pl_p08.s76543210) - convert_short8(pl_n08.s76543210);
	short8 mx_07_hi = convert_short8(pr_p07.hi) + convert_short8(pr_n07.hi) - convert_short8(pl_p07.s76543210) - convert_short8(pl_n07.s76543210);
	short8 mx_06_hi = convert_short8(pr_p06.hi) + convert_short8(pr_n06.hi) - convert_short8(pl_p06.s76543210) - convert_short8(pl_n06.s76543210);
	short8 mx_05_hi = convert_short8(pr_p05.hi) + convert_short8(pr_n05.hi) - convert_short8(pl_p05.s76543210) - convert_short8(pl_n05.s76543210);
	short8 mx_04_hi = convert_short8(pr_p04.hi) + convert_short8(pr_n04.hi) - convert_short8(pl_p04.s76543210) - convert_short8(pl_n04.s76543210);
	short8 mx_03_hi = convert_short8(pr_p03.hi) + convert_short8(pr_n03.hi) - convert_short8(pl_p03.s76543210) - convert_short8(pl_n03.s76543210);
	short8 mx_02_hi = convert_short8(pr_p02.hi) + convert_short8(pr_n02.hi) - convert_short8(pl_p02.s76543210) - convert_short8(pl_n02.s76543210);
	short8 mx_01_hi = convert_short8(pr_p01.hi) + convert_short8(pr_n01.hi) - convert_short8(pl_p01.s76543210) - convert_short8(pl_n01.s76543210);
	short8 mx_00_hi = convert_short8(pr_p00.hi)                             - convert_short8(pl_p00.s76543210);

	int sum_x = (convert_int(mx_00_lo.s1) + convert_int(mx_01_lo.s1) + convert_int(mx_02_lo.s1) + convert_int(mx_03_lo.s1) + convert_int(mx_04_lo.s1) + convert_int(mx_05_lo.s1) + convert_int(mx_06_lo.s1) + convert_int(mx_07_lo.s1) + convert_int(mx_08_lo.s1) + convert_int(mx_09_lo.s1) + convert_int(mx_10_lo.s1) + convert_int(mx_11_lo.s1) + convert_int(mx_12_lo.s1) + convert_int(mx_13_lo.s1) + convert_int(mx_14_lo.s1) + convert_int(mx_15_lo.s1))
	          + (convert_int(mx_00_lo.s2) + convert_int(mx_01_lo.s2) + convert_int(mx_02_lo.s2) + convert_int(mx_03_lo.s2) + convert_int(mx_04_lo.s2) + convert_int(mx_05_lo.s2) + convert_int(mx_06_lo.s2) + convert_int(mx_07_lo.s2) + convert_int(mx_08_lo.s2) + convert_int(mx_09_lo.s2) + convert_int(mx_10_lo.s2) + convert_int(mx_11_lo.s2) + convert_int(mx_12_lo.s2) + convert_int(mx_13_lo.s2) + convert_int(mx_14_lo.s2) + convert_int(mx_15_lo.s2)) * 2
	          + (convert_int(mx_00_lo.s3) + convert_int(mx_01_lo.s3) + convert_int(mx_02_lo.s3) + convert_int(mx_03_lo.s3) + convert_int(mx_04_lo.s3) + convert_int(mx_05_lo.s3) + convert_int(mx_06_lo.s3) + convert_int(mx_07_lo.s3) + convert_int(mx_08_lo.s3) + convert_int(mx_09_lo.s3) + convert_int(mx_10_lo.s3) + convert_int(mx_11_lo.s3) + convert_int(mx_12_lo.s3) + convert_int(mx_13_lo.s3) + convert_int(mx_14_lo.s3) + convert_int(mx_15_lo.s3)) * 3
	          + (convert_int(mx_00_lo.s4) + convert_int(mx_01_lo.s4) + convert_int(mx_02_lo.s4) + convert_int(mx_03_lo.s4) + convert_int(mx_04_lo.s4) + convert_int(mx_05_lo.s4) + convert_int(mx_06_lo.s4) + convert_int(mx_07_lo.s4) + convert_int(mx_08_lo.s4) + convert_int(mx_09_lo.s4) + convert_int(mx_10_lo.s4) + convert_int(mx_11_lo.s4) + convert_int(mx_12_lo.s4) + convert_int(mx_13_lo.s4) + convert_int(mx_14_lo.s4)) * 4
	          + (convert_int(mx_00_lo.s5) + convert_int(mx_01_lo.s5) + convert_int(mx_02_lo.s5) + convert_int(mx_03_lo.s5) + convert_int(mx_04_lo.s5) + convert_int(mx_05_lo.s5) + convert_int(mx_06_lo.s5) + convert_int(mx_07_lo.s5) + convert_int(mx_08_lo.s5) + convert_int(mx_09_lo.s5) + convert_int(mx_10_lo.s5) + convert_int(mx_11_lo.s5) + convert_int(mx_12_lo.s5) + convert_int(mx_13_lo.s5) + convert_int(mx_14_lo.s5)) * 5
	          + (convert_int(mx_00_lo.s6) + convert_int(mx_01_lo.s6) + convert_int(mx_02_lo.s6) + convert_int(mx_03_lo.s6) + convert_int(mx_04_lo.s6) + convert_int(mx_05_lo.s6) + convert_int(mx_06_lo.s6) + convert_int(mx_07_lo.s6) + convert_int(mx_08_lo.s6) + convert_int(mx_09_lo.s6) + convert_int(mx_10_lo.s6) + convert_int(mx_11_lo.s6) + convert_int(mx_12_lo.s6) + convert_int(mx_13_lo.s6) + convert_int(mx_14_lo.s6)) * 6
	          + (convert_int(mx_00_lo.s7) + convert_int(mx_01_lo.s7) + convert_int(mx_02_lo.s7) + convert_int(mx_03_lo.s7) + convert_int(mx_04_lo.s7) + convert_int(mx_05_lo.s7) + convert_int(mx_06_lo.s7) + convert_int(mx_07_lo.s7) + convert_int(mx_08_lo.s7) + convert_int(mx_09_lo.s7) + convert_int(mx_10_lo.s7) + convert_int(mx_11_lo.s7) + convert_int(mx_12_lo.s7) + convert_int(mx_13_lo.s7)) * 7
	          + (convert_int(mx_00_hi.s0) + convert_int(mx_01_hi.s0) + convert_int(mx_02_hi.s0) + convert_int(mx_03_hi.s0) + convert_int(mx_04_hi.s0) + convert_int(mx_05_hi.s0) + convert_int(mx_06_hi.s0) + convert_int(mx_07_hi.s0) + convert_int(mx_08_hi.s0) + convert_int(mx_09_hi.s0) + convert_int(mx_10_hi.s0) + convert_int(mx_11_hi.s0) + convert_int(mx_12_hi.s0) + convert_int(mx_13_hi.s0)) * 8
	          + (convert_int(mx_00_hi.s1) + convert_int(mx_01_hi.s1) + convert_int(mx_02_hi.s1) + convert_int(mx_03_hi.s1) + convert_int(mx_04_hi.s1) + convert_int(mx_05_hi.s1) + convert_int(mx_06_hi.s1) + convert_int(mx_07_hi.s1) + convert_int(mx_08_hi.s1) + convert_int(mx_09_hi.s1) + convert_int(mx_10_hi.s1) + convert_int(mx_11_hi.s1) + convert_int(mx_12_hi.s1)) * 9
	          + (convert_int(mx_00_hi.s2) + convert_int(mx_01_hi.s2) + convert_int(mx_02_hi.s2) + convert_int(mx_03_hi.s2) + convert_int(mx_04_hi.s2) + convert_int(mx_05_hi.s2) + convert_int(mx_06_hi.s2) + convert_int(mx_07_hi.s2) + convert_int(mx_08_hi.s2) + convert_int(mx_09_hi.s2) + convert_int(mx_10_hi.s2) + convert_int(mx_11_hi.s2)) * 10
	          + (convert_int(mx_00_hi.s3) + convert_int(mx_01_hi.s3) + convert_int(mx_02_hi.s3) + convert_int(mx_03_hi.s3) + convert_int(mx_04_hi.s3) + convert_int(mx_05_hi.s3) + convert_int(mx_06_hi.s3) + convert_int(mx_07_hi.s3) + convert_int(mx_08_hi.s3) + convert_int(mx_09_hi.s3) + convert_int(mx_10_hi.s3)) * 11
	          + (convert_int(mx_00_hi.s4) + convert_int(mx_01_hi.s4) + convert_int(mx_02_hi.s4) + convert_int(mx_03_hi.s4) + convert_int(mx_04_hi.s4) + convert_int(mx_05_hi.s4) + convert_int(mx_06_hi.s4) + convert_int(mx_07_hi.s4) + convert_int(mx_08_hi.s4) + convert_int(mx_09_hi.s4)) * 12
	          + (convert_int(mx_00_hi.s5) + convert_int(mx_01_hi.s5) + convert_int(mx_02_hi.s5) + convert_int(mx_03_hi.s5) + convert_int(mx_04_hi.s5) + convert_int(mx_05_hi.s5) + convert_int(mx_06_hi.s5) + convert_int(mx_07_hi.s5) + convert_int(mx_08_hi.s5)) * 13
	          + (convert_int(mx_00_hi.s6) + convert_int(mx_01_hi.s6) + convert_int(mx_02_hi.s6) + convert_int(mx_03_hi.s6) + convert_int(mx_04_hi.s6) + convert_int(mx_05_hi.s6) + convert_int(mx_06_hi.s6)) * 14
	          + (convert_int(mx_00_hi.s7) + convert_int(mx_01_hi.s7) + convert_int(mx_02_hi.s7) + convert_int(mx_03_hi.s7)) * 15;

	short8 my_15_lo = convert_short8(pr_p15.lo) - convert_short8(pr_n15.lo) + convert_short8(pl_p15.sfedcba98) - convert_short8(pl_n15.sfedcba98);
	short8 my_14_lo = convert_short8(pr_p14.lo) - convert_short8(pr_n14.lo) + convert_short8(pl_p14.sfedcba98) - convert_short8(pl_n14.sfedcba98);
	short8 my_13_lo = convert_short8(pr_p13.lo) - convert_short8(pr_n13.lo) + convert_short8(pl_p13.sfedcba98) - convert_short8(pl_n13.sfedcba98);
	short8 my_12_lo = convert_short8(pr_p12.lo) - convert_short8(pr_n12.lo) + convert_short8(pl_p12.sfedcba98) - convert_short8(pl_n12.sfedcba98);
	short8 my_11_lo = convert_short8(pr_p11.lo) - convert_short8(pr_n11.lo) + convert_short8(pl_p11.sfedcba98) - convert_short8(pl_n11.sfedcba98);
	short8 my_10_lo = convert_short8(pr_p10.lo) - convert_short8(pr_n10.lo) + convert_short8(pl_p10.sfedcba98) - convert_short8(pl_n10.sfedcba98);
	short8 my_09_lo = convert_short8(pr_p09.lo) - convert_short8(pr_n09.lo) + convert_short8(pl_p09.sfedcba98) - convert_short8(pl_n09.sfedcba98);
	short8 my_08_lo = convert_short8(pr_p08.lo) - convert_short8(pr_n08.lo) + convert_short8(pl_p08.sfedcba98) - convert_short8(pl_n08.sfedcba98);
	short8 my_07_lo = convert_short8(pr_p07.lo) - convert_short8(pr_n07.lo) + convert_short8(pl_p07.sfedcba98) - convert_short8(pl_n07.sfedcba98);
	short8 my_06_lo = convert_short8(pr_p06.lo) - convert_short8(pr_n06.lo) + convert_short8(pl_p06.sfedcba98) - convert_short8(pl_n06.sfedcba98);
	short8 my_05_lo = convert_short8(pr_p05.lo) - convert_short8(pr_n05.lo) + convert_short8(pl_p05.sfedcba98) - convert_short8(pl_n05.sfedcba98);
	short8 my_04_lo = convert_short8(pr_p04.lo) - convert_short8(pr_n04.lo) + convert_short8(pl_p04.sfedcba98) - convert_short8(pl_n04.sfedcba98);
	short8 my_03_lo = convert_short8(pr_p03.lo) - convert_short8(pr_n03.lo) + convert_short8(pl_p03.sfedcba98) - convert_short8(pl_n03.sfedcba98);
	short8 my_02_lo = convert_short8(pr_p02.lo) - convert_short8(pr_n02.lo) + convert_short8(pl_p02.sfedcba98) - convert_short8(pl_n02.sfedcba98);
	short8 my_01_lo = convert_short8(pr_p01.lo) - convert_short8(pr_n01.lo) + convert_short8(pl_p01.sfedcba98) - convert_short8(pl_n01.sfedcba98);
	short8 my_00_lo = convert_short8(pr_p00.lo)                             + convert_short8(pl_p00.sfedcba98);

	short8 my_15_hi = convert_short8(pr_p15.hi) - convert_short8(pr_n15.hi) + convert_short8(pl_p15.s76543210) - convert_short8(pl_n15.s76543210);
	short8 my_14_hi = convert_short8(pr_p14.hi) - convert_short8(pr_n14.hi) + convert_short8(pl_p14.s76543210) - convert_short8(pl_n14.s76543210);
	short8 my_13_hi = convert_short8(pr_p13.hi) - convert_short8(pr_n13.hi) + convert_short8(pl_p13.s76543210) - convert_short8(pl_n13.s76543210);
	short8 my_12_hi = convert_short8(pr_p12.hi) - convert_short8(pr_n12.hi) + convert_short8(pl_p12.s76543210) - convert_short8(pl_n12.s76543210);
	short8 my_11_hi = convert_short8(pr_p11.hi) - convert_short8(pr_n11.hi) + convert_short8(pl_p11.s76543210) - convert_short8(pl_n11.s76543210);
	short8 my_10_hi = convert_short8(pr_p10.hi) - convert_short8(pr_n10.hi) + convert_short8(pl_p10.s76543210) - convert_short8(pl_n10.s76543210);
	short8 my_09_hi = convert_short8(pr_p09.hi) - convert_short8(pr_n09.hi) + convert_short8(pl_p09.s76543210) - convert_short8(pl_n09.s76543210);
	short8 my_08_hi = convert_short8(pr_p08.hi) - convert_short8(pr_n08.hi) + convert_short8(pl_p08.s76543210) - convert_short8(pl_n08.s76543210);
	short8 my_07_hi = convert_short8(pr_p07.hi) - convert_short8(pr_n07.hi) + convert_short8(pl_p07.s76543210) - convert_short8(pl_n07.s76543210);
	short8 my_06_hi = convert_short8(pr_p06.hi) - convert_short8(pr_n06.hi) + convert_short8(pl_p06.s76543210) - convert_short8(pl_n06.s76543210);
	short8 my_05_hi = convert_short8(pr_p05.hi) - convert_short8(pr_n05.hi) + convert_short8(pl_p05.s76543210) - convert_short8(pl_n05.s76543210);
	short8 my_04_hi = convert_short8(pr_p04.hi) - convert_short8(pr_n04.hi) + convert_short8(pl_p04.s76543210) - convert_short8(pl_n04.s76543210);
	short8 my_03_hi = convert_short8(pr_p03.hi) - convert_short8(pr_n03.hi) + convert_short8(pl_p03.s76543210) - convert_short8(pl_n03.s76543210);
	short8 my_02_hi = convert_short8(pr_p02.hi) - convert_short8(pr_n02.hi) + convert_short8(pl_p02.s76543210) - convert_short8(pl_n02.s76543210);
	short8 my_01_hi = convert_short8(pr_p01.hi) - convert_short8(pr_n01.hi) + convert_short8(pl_p01.s76543210) - convert_short8(pl_n01.s76543210);
	short8 my_00_hi = convert_short8(pr_p00.hi)                             + convert_short8(pl_p00.s76543210);

	int sum_y = (convert_int(my_15_lo.s0/2) + convert_int(my_15_lo.s1) + convert_int(my_15_lo.s2) + convert_int(my_15_lo.s3)) * 15
	          + (convert_int(my_14_lo.s0/2) + convert_int(my_14_lo.s1) + convert_int(my_14_lo.s2) + convert_int(my_14_lo.s3) + convert_int(my_14_lo.s4) + convert_int(my_14_lo.s5) + convert_int(my_14_lo.s6)) * 14
	          + (convert_int(my_13_lo.s0/2) + convert_int(my_13_lo.s1) + convert_int(my_13_lo.s2) + convert_int(my_13_lo.s3) + convert_int(my_13_lo.s4) + convert_int(my_13_lo.s5) + convert_int(my_13_lo.s6) + convert_int(my_13_lo.s7) + convert_int(my_13_hi.s0)) * 13
	          + (convert_int(my_12_lo.s0/2) + convert_int(my_12_lo.s1) + convert_int(my_12_lo.s2) + convert_int(my_12_lo.s3) + convert_int(my_12_lo.s4) + convert_int(my_12_lo.s5) + convert_int(my_12_lo.s6) + convert_int(my_12_lo.s7) + convert_int(my_12_hi.s0) + convert_int(my_12_hi.s1)) * 12
	          + (convert_int(my_11_lo.s0/2) + convert_int(my_11_lo.s1) + convert_int(my_11_lo.s2) + convert_int(my_11_lo.s3) + convert_int(my_11_lo.s4) + convert_int(my_11_lo.s5) + convert_int(my_11_lo.s6) + convert_int(my_11_lo.s7) + convert_int(my_11_hi.s0) + convert_int(my_11_hi.s1) + convert_int(my_11_hi.s2)) * 11
	          + (convert_int(my_10_lo.s0/2) + convert_int(my_10_lo.s1) + convert_int(my_10_lo.s2) + convert_int(my_10_lo.s3) + convert_int(my_10_lo.s4) + convert_int(my_10_lo.s5) + convert_int(my_10_lo.s6) + convert_int(my_10_lo.s7) + convert_int(my_10_hi.s0) + convert_int(my_10_hi.s1) + convert_int(my_10_hi.s2) + convert_int(my_10_hi.s3)) * 10
	          + (convert_int(my_09_lo.s0/2) + convert_int(my_09_lo.s1) + convert_int(my_09_lo.s2) + convert_int(my_09_lo.s3) + convert_int(my_09_lo.s4) + convert_int(my_09_lo.s5) + convert_int(my_09_lo.s6) + convert_int(my_09_lo.s7) + convert_int(my_09_hi.s0) + convert_int(my_09_hi.s1) + convert_int(my_09_hi.s2) + convert_int(my_09_hi.s3) + convert_int(my_09_hi.s4)) * 9
	          + (convert_int(my_08_lo.s0/2) + convert_int(my_08_lo.s1) + convert_int(my_08_lo.s2) + convert_int(my_08_lo.s3) + convert_int(my_08_lo.s4) + convert_int(my_08_lo.s5) + convert_int(my_08_lo.s6) + convert_int(my_08_lo.s7) + convert_int(my_08_hi.s0) + convert_int(my_08_hi.s1) + convert_int(my_08_hi.s2) + convert_int(my_08_hi.s3) + convert_int(my_08_hi.s4) + convert_int(my_08_hi.s5)) * 8
	          + (convert_int(my_07_lo.s0/2) + convert_int(my_07_lo.s1) + convert_int(my_07_lo.s2) + convert_int(my_07_lo.s3) + convert_int(my_07_lo.s4) + convert_int(my_07_lo.s5) + convert_int(my_07_lo.s6) + convert_int(my_07_lo.s7) + convert_int(my_07_hi.s0) + convert_int(my_07_hi.s1) + convert_int(my_07_hi.s2) + convert_int(my_07_hi.s3) + convert_int(my_07_hi.s4) + convert_int(my_07_hi.s5)) * 7
	          + (convert_int(my_06_lo.s0/2) + convert_int(my_06_lo.s1) + convert_int(my_06_lo.s2) + convert_int(my_06_lo.s3) + convert_int(my_06_lo.s4) + convert_int(my_06_lo.s5) + convert_int(my_06_lo.s6) + convert_int(my_06_lo.s7) + convert_int(my_06_hi.s0) + convert_int(my_06_hi.s1) + convert_int(my_06_hi.s2) + convert_int(my_06_hi.s3) + convert_int(my_06_hi.s4) + convert_int(my_06_hi.s5) + convert_int(my_06_hi.s6)) * 6
	          + (convert_int(my_05_lo.s0/2) + convert_int(my_05_lo.s1) + convert_int(my_05_lo.s2) + convert_int(my_05_lo.s3) + convert_int(my_05_lo.s4) + convert_int(my_05_lo.s5) + convert_int(my_05_lo.s6) + convert_int(my_05_lo.s7) + convert_int(my_05_hi.s0) + convert_int(my_05_hi.s1) + convert_int(my_05_hi.s2) + convert_int(my_05_hi.s3) + convert_int(my_05_hi.s4) + convert_int(my_05_hi.s5) + convert_int(my_05_hi.s6)) * 5
	          + (convert_int(my_04_lo.s0/2) + convert_int(my_04_lo.s1) + convert_int(my_04_lo.s2) + convert_int(my_04_lo.s3) + convert_int(my_04_lo.s4) + convert_int(my_04_lo.s5) + convert_int(my_04_lo.s6) + convert_int(my_04_lo.s7) + convert_int(my_04_hi.s0) + convert_int(my_04_hi.s1) + convert_int(my_04_hi.s2) + convert_int(my_04_hi.s3) + convert_int(my_04_hi.s4) + convert_int(my_04_hi.s5) + convert_int(my_04_hi.s6)) * 4
	          + (convert_int(my_03_lo.s0/2) + convert_int(my_03_lo.s1) + convert_int(my_03_lo.s2) + convert_int(my_03_lo.s3) + convert_int(my_03_lo.s4) + convert_int(my_03_lo.s5) + convert_int(my_03_lo.s6) + convert_int(my_03_lo.s7) + convert_int(my_03_hi.s0) + convert_int(my_03_hi.s1) + convert_int(my_03_hi.s2) + convert_int(my_03_hi.s3) + convert_int(my_03_hi.s4) + convert_int(my_03_hi.s5) + convert_int(my_03_hi.s6) + convert_int(my_03_hi.s7)) * 3
	          + (convert_int(my_02_lo.s0/2) + convert_int(my_02_lo.s1) + convert_int(my_02_lo.s2) + convert_int(my_02_lo.s3) + convert_int(my_02_lo.s4) + convert_int(my_02_lo.s5) + convert_int(my_02_lo.s6) + convert_int(my_02_lo.s7) + convert_int(my_02_hi.s0) + convert_int(my_02_hi.s1) + convert_int(my_02_hi.s2) + convert_int(my_02_hi.s3) + convert_int(my_02_hi.s4) + convert_int(my_02_hi.s5) + convert_int(my_02_hi.s6) + convert_int(my_02_hi.s7)) * 2
	          + (convert_int(my_01_lo.s0/2) + convert_int(my_01_lo.s1) + convert_int(my_01_lo.s2) + convert_int(my_01_lo.s3) + convert_int(my_01_lo.s4) + convert_int(my_01_lo.s5) + convert_int(my_01_lo.s6) + convert_int(my_01_lo.s7) + convert_int(my_01_hi.s0) + convert_int(my_01_hi.s1) + convert_int(my_01_hi.s2) + convert_int(my_01_hi.s3) + convert_int(my_01_hi.s4) + convert_int(my_01_hi.s5) + convert_int(my_01_hi.s6) + convert_int(my_01_hi.s7)) * 1;

	/*============================================*/
	float2 normAngle = normalize((float2)(convert_float(sum_x), convert_float(sum_y)));
	vstore2(normAngle, id, angle);
}

kernel void ComputeDescriptor(read_only image2d_t _img, sampler_t sampler, global char * bit_pattern, global int* kps, int kps_num, global float * _angle, global uchar* descriptors){
	int id0 = get_global_id(0);
	int id1 = get_global_id(1);

	if(id1 >= kps_num)
		return ;

	int2 point = vload2(id1, kps);
	float2 angle = vload2(id1, _angle);
	/*======================================*/
	char16 pat0 = vload16(id0*8+0, bit_pattern);
	char16 pat1 = vload16(id0*8+1, bit_pattern);
	char16 pat2 = vload16(id0*8+2, bit_pattern);
	char16 pat3 = vload16(id0*8+3, bit_pattern);
	char16 pat4 = vload16(id0*8+4, bit_pattern);
	char16 pat5 = vload16(id0*8+5, bit_pattern);
	char16 pat6 = vload16(id0*8+6, bit_pattern);
	char16 pat7 = vload16(id0*8+7, bit_pattern);

	float4 pat00_rot_i = convert_float4(pat0.s0246) * angle.x - convert_float4(pat0.s1357) * angle.y + convert_float(point.x);
	float4 pat01_rot_i = convert_float4(pat0.s8ace) * angle.x - convert_float4(pat0.s9bdf) * angle.y + convert_float(point.x);
	float4 pat02_rot_i = convert_float4(pat1.s0246) * angle.x - convert_float4(pat1.s1357) * angle.y + convert_float(point.x);
	float4 pat03_rot_i = convert_float4(pat1.s8ace) * angle.x - convert_float4(pat1.s9bdf) * angle.y + convert_float(point.x);
	float4 pat04_rot_i = convert_float4(pat2.s0246) * angle.x - convert_float4(pat2.s1357) * angle.y + convert_float(point.x);
	float4 pat05_rot_i = convert_float4(pat2.s8ace) * angle.x - convert_float4(pat2.s9bdf) * angle.y + convert_float(point.x);
	float4 pat06_rot_i = convert_float4(pat3.s0246) * angle.x - convert_float4(pat3.s1357) * angle.y + convert_float(point.x);
	float4 pat07_rot_i = convert_float4(pat3.s8ace) * angle.x - convert_float4(pat3.s9bdf) * angle.y + convert_float(point.x);
	float4 pat08_rot_i = convert_float4(pat4.s0246) * angle.x - convert_float4(pat4.s1357) * angle.y + convert_float(point.x);
	float4 pat09_rot_i = convert_float4(pat4.s8ace) * angle.x - convert_float4(pat4.s9bdf) * angle.y + convert_float(point.x);
	float4 pat10_rot_i = convert_float4(pat5.s0246) * angle.x - convert_float4(pat5.s1357) * angle.y + convert_float(point.x);
	float4 pat11_rot_i = convert_float4(pat5.s8ace) * angle.x - convert_float4(pat5.s9bdf) * angle.y + convert_float(point.x);
	float4 pat12_rot_i = convert_float4(pat6.s0246) * angle.x - convert_float4(pat6.s1357) * angle.y + convert_float(point.x);
	float4 pat13_rot_i = convert_float4(pat6.s8ace) * angle.x - convert_float4(pat6.s9bdf) * angle.y + convert_float(point.x);
	float4 pat14_rot_i = convert_float4(pat7.s0246) * angle.x - convert_float4(pat7.s1357) * angle.y + convert_float(point.x);
	float4 pat15_rot_i = convert_float4(pat7.s8ace) * angle.x - convert_float4(pat7.s9bdf) * angle.y + convert_float(point.x);

	float4 pat00_rot_q = convert_float4(pat0.s0246) * angle.y + convert_float4(pat0.s1357) * angle.x + convert_float(point.y);
	float4 pat01_rot_q = convert_float4(pat0.s8ace) * angle.y + convert_float4(pat0.s9bdf) * angle.x + convert_float(point.y);
	float4 pat02_rot_q = convert_float4(pat1.s0246) * angle.y + convert_float4(pat1.s1357) * angle.x + convert_float(point.y);
	float4 pat03_rot_q = convert_float4(pat1.s8ace) * angle.y + convert_float4(pat1.s9bdf) * angle.x + convert_float(point.y);
	float4 pat04_rot_q = convert_float4(pat2.s0246) * angle.y + convert_float4(pat2.s1357) * angle.x + convert_float(point.y);
	float4 pat05_rot_q = convert_float4(pat2.s8ace) * angle.y + convert_float4(pat2.s9bdf) * angle.x + convert_float(point.y);
	float4 pat06_rot_q = convert_float4(pat3.s0246) * angle.y + convert_float4(pat3.s1357) * angle.x + convert_float(point.y);
	float4 pat07_rot_q = convert_float4(pat3.s8ace) * angle.y + convert_float4(pat3.s9bdf) * angle.x + convert_float(point.y);
	float4 pat08_rot_q = convert_float4(pat4.s0246) * angle.y + convert_float4(pat4.s1357) * angle.x + convert_float(point.y);
	float4 pat09_rot_q = convert_float4(pat4.s8ace) * angle.y + convert_float4(pat4.s9bdf) * angle.x + convert_float(point.y);
	float4 pat10_rot_q = convert_float4(pat5.s0246) * angle.y + convert_float4(pat5.s1357) * angle.x + convert_float(point.y);
	float4 pat11_rot_q = convert_float4(pat5.s8ace) * angle.y + convert_float4(pat5.s9bdf) * angle.x + convert_float(point.y);
	float4 pat12_rot_q = convert_float4(pat6.s0246) * angle.y + convert_float4(pat6.s1357) * angle.x + convert_float(point.y);
	float4 pat13_rot_q = convert_float4(pat6.s8ace) * angle.y + convert_float4(pat6.s9bdf) * angle.x + convert_float(point.y);
	float4 pat14_rot_q = convert_float4(pat7.s0246) * angle.y + convert_float4(pat7.s1357) * angle.x + convert_float(point.y);
	float4 pat15_rot_q = convert_float4(pat7.s8ace) * angle.y + convert_float4(pat7.s9bdf) * angle.x + convert_float(point.y);

	float16 p00 = (float16)(read_imagef(_img, sampler, (float2)(pat00_rot_i.s0, pat00_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat00_rot_i.s1, pat00_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat00_rot_i.s2, pat00_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat00_rot_i.s3, pat00_rot_q.s3)));
	float16 p01 = (float16)(read_imagef(_img, sampler, (float2)(pat01_rot_i.s0, pat01_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat01_rot_i.s1, pat01_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat01_rot_i.s2, pat01_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat01_rot_i.s3, pat01_rot_q.s3)));
	float16 p02 = (float16)(read_imagef(_img, sampler, (float2)(pat02_rot_i.s0, pat02_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat02_rot_i.s1, pat02_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat02_rot_i.s2, pat02_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat02_rot_i.s3, pat02_rot_q.s3)));
	float16 p03 = (float16)(read_imagef(_img, sampler, (float2)(pat03_rot_i.s0, pat03_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat03_rot_i.s1, pat03_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat03_rot_i.s2, pat03_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat03_rot_i.s3, pat03_rot_q.s3)));
	float16 p04 = (float16)(read_imagef(_img, sampler, (float2)(pat04_rot_i.s0, pat04_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat04_rot_i.s1, pat04_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat04_rot_i.s2, pat04_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat04_rot_i.s3, pat04_rot_q.s3)));
	float16 p05 = (float16)(read_imagef(_img, sampler, (float2)(pat05_rot_i.s0, pat05_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat05_rot_i.s1, pat05_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat05_rot_i.s2, pat05_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat05_rot_i.s3, pat05_rot_q.s3)));
	float16 p06 = (float16)(read_imagef(_img, sampler, (float2)(pat06_rot_i.s0, pat06_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat06_rot_i.s1, pat06_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat06_rot_i.s2, pat06_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat06_rot_i.s3, pat06_rot_q.s3)));
	float16 p07 = (float16)(read_imagef(_img, sampler, (float2)(pat07_rot_i.s0, pat07_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat07_rot_i.s1, pat07_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat07_rot_i.s2, pat07_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat07_rot_i.s3, pat07_rot_q.s3)));
	float16 p08 = (float16)(read_imagef(_img, sampler, (float2)(pat08_rot_i.s0, pat08_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat08_rot_i.s1, pat08_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat08_rot_i.s2, pat08_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat08_rot_i.s3, pat08_rot_q.s3)));
	float16 p09 = (float16)(read_imagef(_img, sampler, (float2)(pat09_rot_i.s0, pat09_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat09_rot_i.s1, pat09_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat09_rot_i.s2, pat09_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat09_rot_i.s3, pat09_rot_q.s3)));
	float16 p10 = (float16)(read_imagef(_img, sampler, (float2)(pat10_rot_i.s0, pat10_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat10_rot_i.s1, pat10_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat10_rot_i.s2, pat10_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat10_rot_i.s3, pat10_rot_q.s3)));
	float16 p11 = (float16)(read_imagef(_img, sampler, (float2)(pat11_rot_i.s0, pat11_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat11_rot_i.s1, pat11_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat11_rot_i.s2, pat11_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat11_rot_i.s3, pat11_rot_q.s3)));
	float16 p12 = (float16)(read_imagef(_img, sampler, (float2)(pat12_rot_i.s0, pat12_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat12_rot_i.s1, pat12_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat12_rot_i.s2, pat12_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat12_rot_i.s3, pat12_rot_q.s3)));
	float16 p13 = (float16)(read_imagef(_img, sampler, (float2)(pat13_rot_i.s0, pat13_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat13_rot_i.s1, pat13_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat13_rot_i.s2, pat13_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat13_rot_i.s3, pat13_rot_q.s3)));
	float16 p14 = (float16)(read_imagef(_img, sampler, (float2)(pat14_rot_i.s0, pat14_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat14_rot_i.s1, pat14_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat14_rot_i.s2, pat14_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat14_rot_i.s3, pat14_rot_q.s3)));
	float16 p15 = (float16)(read_imagef(_img, sampler, (float2)(pat15_rot_i.s0, pat15_rot_q.s0)), read_imagef(_img, sampler, (float2)(pat15_rot_i.s1, pat15_rot_q.s1)),
	                        read_imagef(_img, sampler, (float2)(pat15_rot_i.s2, pat15_rot_q.s2)), read_imagef(_img, sampler, (float2)(pat15_rot_i.s3, pat15_rot_q.s3)));

	uchar desc0 = ((p00.s0 > p00.s4)<<0) | ((p00.s8 > p00.sC)<<1) | ((p01.s0 > p01.s4)<<2) | ((p01.s8 > p01.sC)<<3) 
	            | ((p02.s0 > p02.s4)<<4) | ((p02.s8 > p02.sC)<<5) | ((p03.s0 > p03.s4)<<6) | ((p03.s8 > p03.sC)<<7);
	uchar desc1 = ((p04.s0 > p04.s4)<<0) | ((p04.s8 > p04.sC)<<1) | ((p05.s0 > p05.s4)<<2) | ((p05.s8 > p05.sC)<<3) 
	            | ((p06.s0 > p06.s4)<<4) | ((p06.s8 > p06.sC)<<5) | ((p07.s0 > p07.s4)<<6) | ((p07.s8 > p07.sC)<<7);
	uchar desc2 = ((p08.s0 > p08.s4)<<0) | ((p08.s8 > p08.sC)<<1) | ((p09.s0 > p09.s4)<<2) | ((p09.s8 > p09.sC)<<3) 
	            | ((p10.s0 > p10.s4)<<4) | ((p10.s8 > p10.sC)<<5) | ((p11.s0 > p11.s4)<<6) | ((p11.s8 > p11.sC)<<7);
	uchar desc3 = ((p12.s0 > p12.s4)<<0) | ((p12.s8 > p12.sC)<<1) | ((p13.s0 > p13.s4)<<2) | ((p13.s8 > p13.sC)<<3) 
	            | ((p14.s0 > p14.s4)<<4) | ((p14.s8 > p14.sC)<<5) | ((p15.s0 > p15.s4)<<6) | ((p15.s8 > p15.sC)<<7);

	vstore4((uchar4)(desc0, desc1, desc2, desc3), id1*8+id0, descriptors);
}

kernel void BFMatch(
global int * kps_a, global int * desc_a, int num_a, 
global int * kps_b, global int * desc_b, int num_b, 
global unsigned int * matches_a, global unsigned int* matches_b)
{
	int idx = get_global_id(0);
	int idy = get_global_id(1);

	if(idx >= num_a || idy >= num_b)
		return ;
	
	int2 point0 = vload2(idx, kps_a);
	int2 point1 = vload2(idy, kps_b);

	int8 desc0 = vload8(idx, desc_a);
	int8 desc1 = vload8(idy, desc_b);

	int8 diff = desc0 ^ desc1;
	/*==================================================*/
	short8 diff_L0 = convert_short8((diff>>0) & (short)1);
	short8 diff_L1 = convert_short8((diff>>1) & (short)1);
	short8 diff_L2 = convert_short8((diff>>2) & (short)1);
	short8 diff_L3 = convert_short8((diff>>3) & (short)1);
	short8 diff_L4 = convert_short8((diff>>4) & (short)1);
	short8 diff_L5 = convert_short8((diff>>5) & (short)1);
	short8 diff_L6 = convert_short8((diff>>6) & (short)1);
	short8 diff_L7 = convert_short8((diff>>7) & (short)1);
	short8 diff_L8 = convert_short8((diff>>8) & (short)1);
	short8 diff_L9 = convert_short8((diff>>9) & (short)1);
	short8 diff_LA = convert_short8((diff>>10) & (short)1);
	short8 diff_LB = convert_short8((diff>>11) & (short)1);
	short8 diff_LC = convert_short8((diff>>12) & (short)1);
	short8 diff_LD = convert_short8((diff>>13) & (short)1);
	short8 diff_LE = convert_short8((diff>>14) & (short)1);
	short8 diff_LF = convert_short8((diff>>15) & (short)1);

	short diff_sum0 = VSUM8(diff_L0);
	short diff_sum1 = VSUM8(diff_L1);
	short diff_sum2 = VSUM8(diff_L2);
	short diff_sum3 = VSUM8(diff_L3);
	short diff_sum4 = VSUM8(diff_L4);
	short diff_sum5 = VSUM8(diff_L5);
	short diff_sum6 = VSUM8(diff_L6);
	short diff_sum7 = VSUM8(diff_L7);
	short diff_sum8 = VSUM8(diff_L8);
	short diff_sum9 = VSUM8(diff_L9);
	short diff_sum10 = VSUM8(diff_LA);
	short diff_sum11 = VSUM8(diff_LB);
	short diff_sum12 = VSUM8(diff_LC);
	short diff_sum13 = VSUM8(diff_LD);
	short diff_sum14 = VSUM8(diff_LE);
	short diff_sum15 = VSUM8(diff_LF);

	short sum_all = diff_sum0 + diff_sum1 + diff_sum2 + diff_sum3 + diff_sum4 + diff_sum5 + diff_sum6 + diff_sum7
	              + diff_sum8 + diff_sum9 + diff_sum10 + diff_sum11 + diff_sum12 + diff_sum13 + diff_sum14 + diff_sum15;

	if(sum_all > 76)
		return ;

	/*==================================================*/
	short8 diff_H0 = convert_short8((diff>>16) & (short)1);
	short8 diff_H1 = convert_short8((diff>>17) & (short)1);
	short8 diff_H2 = convert_short8((diff>>18) & (short)1);
	short8 diff_H3 = convert_short8((diff>>19) & (short)1);
	short8 diff_H4 = convert_short8((diff>>20) & (short)1);
	short8 diff_H5 = convert_short8((diff>>21) & (short)1);
	short8 diff_H6 = convert_short8((diff>>22) & (short)1);
	short8 diff_H7 = convert_short8((diff>>23) & (short)1);
	short8 diff_H8 = convert_short8((diff>>24) & (short)1);
	short8 diff_H9 = convert_short8((diff>>25) & (short)1);
	short8 diff_HA = convert_short8((diff>>26) & (short)1);
	short8 diff_HB = convert_short8((diff>>27) & (short)1);
	short8 diff_HC = convert_short8((diff>>28) & (short)1);
	short8 diff_HD = convert_short8((diff>>29) & (short)1);
	short8 diff_HE = convert_short8((diff>>30) & (short)1);
	short8 diff_HF = convert_short8((diff>>31) & (short)1);

	short diff_sum16 = VSUM8(diff_H0);
	short diff_sum17 = VSUM8(diff_H1);
	short diff_sum18 = VSUM8(diff_H2);
	short diff_sum19 = VSUM8(diff_H3);
	short diff_sum20 = VSUM8(diff_H4);
	short diff_sum21 = VSUM8(diff_H5);
	short diff_sum22 = VSUM8(diff_H6);
	short diff_sum23 = VSUM8(diff_H7);
	short diff_sum24 = VSUM8(diff_H8);
	short diff_sum25 = VSUM8(diff_H9);
	short diff_sum26 = VSUM8(diff_HA);
	short diff_sum27 = VSUM8(diff_HB);
	short diff_sum28 = VSUM8(diff_HC);
	short diff_sum29 = VSUM8(diff_HD);
	short diff_sum30 = VSUM8(diff_HE);
	short diff_sum31 = VSUM8(diff_HF);

	sum_all += diff_sum16 + diff_sum17 + diff_sum18 + diff_sum19 + diff_sum20 + diff_sum21 + diff_sum22 + diff_sum23
	               + diff_sum24 + diff_sum25 + diff_sum26 + diff_sum27 + diff_sum28 + diff_sum29 + diff_sum30 + diff_sum31;

	if(sum_all > 76)
		return ;
	/*==================================================*/
	uint new_a = upsample(convert_ushort(256 - sum_all), convert_ushort(idy));
	uint new_b = upsample(convert_ushort(256 - sum_all), convert_ushort(idx));

	uint old_a = atomic_max(&matches_a[idx*2+0], new_a);
	atomic_max(&matches_a[idx*2+1], min(new_a, old_a));

	uint old_b = atomic_max(&matches_b[idy*2+0], new_b);
	atomic_max(&matches_b[idy*2+1], min(new_b, old_b));
}

