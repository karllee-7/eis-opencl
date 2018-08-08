kernel void computeResponses(global uchar * _img, global int* _kps, global float* _responses){
	int id0 = get_global_id(0);
	int id1 = get_global_id(1);

	global int* kps = _kps + (BLOCK_NOMAX_MAXNUM + 1) * 2 * id1;
	global float* responses = _responses + (BLOCK_NOMAX_MAXNUM + 1) * 1 * id1;

	if(id0 >= kps[0])
		return ;

	int2 point = vload2(id0+1, kps);

	global uchar* img = _img + point.y * GRAY_W + point.x - 4;

	uchar16 p0 = vload16(0, img - 4 * GRAY_W);
	uchar16 p1 = vload16(0, img - 3 * GRAY_W);
	uchar16 p2 = vload16(0, img - 2 * GRAY_W);
	uchar16 p3 = vload16(0, img - 1 * GRAY_W);
	uchar16 p4 = vload16(0, img + 0 * GRAY_W);
	uchar16 p5 = vload16(0, img + 1 * GRAY_W);
	uchar16 p6 = vload16(0, img + 2 * GRAY_W);
	uchar16 p7 = vload16(0, img + 3 * GRAY_W);
	uchar16 p8 = vload16(0, img + 4 * GRAY_W);


	short8 dx0 = convert_short8(p0.s23456788) - convert_short8(p0.s01234566);
	short8 dx1 = convert_short8(p1.s23456788) - convert_short8(p1.s01234566);
	short8 dx2 = convert_short8(p2.s23456788) - convert_short8(p2.s01234566);
	short8 dx3 = convert_short8(p3.s23456788) - convert_short8(p3.s01234566);
	short8 dx4 = convert_short8(p4.s23456788) - convert_short8(p4.s01234566);
	short8 dx5 = convert_short8(p5.s23456788) - convert_short8(p5.s01234566);
	short8 dx6 = convert_short8(p6.s23456788) - convert_short8(p6.s01234566);
	short8 dx7 = convert_short8(p7.s23456788) - convert_short8(p7.s01234566);
	short8 dx8 = convert_short8(p8.s23456788) - convert_short8(p8.s01234566);

	short8 dy0 = convert_short8((uchar8)(p2.s0,p3.s0,p4.s0,p5.s0,p6.s0,p7.s0,p8.s0,p8.s0)) - convert_short8((uchar8)(p0.s0,p1.s0,p2.s0,p3.s0,p4.s0,p5.s0,p6.s0,p6.s0));
	short8 dy1 = convert_short8((uchar8)(p2.s1,p3.s1,p4.s1,p5.s1,p6.s1,p7.s1,p8.s1,p8.s1)) - convert_short8((uchar8)(p0.s1,p1.s1,p2.s1,p3.s1,p4.s1,p5.s1,p6.s1,p6.s1));
	short8 dy2 = convert_short8((uchar8)(p2.s2,p3.s2,p4.s2,p5.s2,p6.s2,p7.s2,p8.s2,p8.s2)) - convert_short8((uchar8)(p0.s2,p1.s2,p2.s2,p3.s2,p4.s2,p5.s2,p6.s2,p6.s2));
	short8 dy3 = convert_short8((uchar8)(p2.s3,p3.s3,p4.s3,p5.s3,p6.s3,p7.s3,p8.s3,p8.s3)) - convert_short8((uchar8)(p0.s3,p1.s3,p2.s3,p3.s3,p4.s3,p5.s3,p6.s3,p6.s3));
	short8 dy4 = convert_short8((uchar8)(p2.s4,p3.s4,p4.s4,p5.s4,p6.s4,p7.s4,p8.s4,p8.s4)) - convert_short8((uchar8)(p0.s4,p1.s4,p2.s4,p3.s4,p4.s4,p5.s4,p6.s4,p6.s4));
	short8 dy5 = convert_short8((uchar8)(p2.s5,p3.s5,p4.s5,p5.s5,p6.s5,p7.s5,p8.s5,p8.s5)) - convert_short8((uchar8)(p0.s5,p1.s5,p2.s5,p3.s5,p4.s5,p5.s5,p6.s5,p6.s5));
	short8 dy6 = convert_short8((uchar8)(p2.s6,p3.s6,p4.s6,p5.s6,p6.s6,p7.s6,p8.s6,p8.s6)) - convert_short8((uchar8)(p0.s6,p1.s6,p2.s6,p3.s6,p4.s6,p5.s6,p6.s6,p6.s6));
	short8 dy7 = convert_short8((uchar8)(p2.s7,p3.s7,p4.s7,p5.s7,p6.s7,p7.s7,p8.s7,p8.s7)) - convert_short8((uchar8)(p0.s7,p1.s7,p2.s7,p3.s7,p4.s7,p5.s7,p6.s7,p6.s7));
	short8 dy8 = convert_short8((uchar8)(p2.s8,p3.s8,p4.s8,p5.s8,p6.s8,p7.s8,p8.s8,p8.s8)) - convert_short8((uchar8)(p0.s8,p1.s8,p2.s8,p3.s8,p4.s8,p5.s8,p6.s8,p6.s8));

	short8 Ix0 = dx1 * (short)2 + dx0 + dx2;
	short8 Ix1 = dx2 * (short)2 + dx1 + dx3;
	short8 Ix2 = dx3 * (short)2 + dx2 + dx4;
	short8 Ix3 = dx4 * (short)2 + dx3 + dx5;
	short8 Ix4 = dx5 * (short)2 + dx4 + dx6;
	short8 Ix5 = dx6 * (short)2 + dx5 + dx7;
	short8 Ix6 = dx7 * (short)2 + dx6 + dx8;

	short8 Iy0 = dy1 * (short)2 + dy0 + dy2;
	short8 Iy1 = dy2 * (short)2 + dy1 + dy3;
	short8 Iy2 = dy3 * (short)2 + dy2 + dy4;
	short8 Iy3 = dy4 * (short)2 + dy3 + dy5;
	short8 Iy4 = dy5 * (short)2 + dy4 + dy6;
	short8 Iy5 = dy6 * (short)2 + dy5 + dy7;
	short8 Iy6 = dy7 * (short)2 + dy6 + dy8;

	int A = (int)Ix0.s0*(int)Ix0.s0 + (int)Ix0.s1*(int)Ix0.s1 + (int)Ix0.s2*(int)Ix0.s2 + (int)Ix0.s3*(int)Ix0.s3 + (int)Ix0.s4*(int)Ix0.s4 + (int)Ix0.s5*(int)Ix0.s5 + (int)Ix0.s6*(int)Ix0.s6
	      + (int)Ix1.s0*(int)Ix1.s0 + (int)Ix1.s1*(int)Ix1.s1 + (int)Ix1.s2*(int)Ix1.s2 + (int)Ix1.s3*(int)Ix1.s3 + (int)Ix1.s4*(int)Ix1.s4 + (int)Ix1.s5*(int)Ix1.s5 + (int)Ix1.s6*(int)Ix1.s6
	      + (int)Ix2.s0*(int)Ix2.s0 + (int)Ix2.s1*(int)Ix2.s1 + (int)Ix2.s2*(int)Ix2.s2 + (int)Ix2.s3*(int)Ix2.s3 + (int)Ix2.s4*(int)Ix2.s4 + (int)Ix2.s5*(int)Ix2.s5 + (int)Ix2.s6*(int)Ix2.s6
	      + (int)Ix3.s0*(int)Ix3.s0 + (int)Ix3.s1*(int)Ix3.s1 + (int)Ix3.s2*(int)Ix3.s2 + (int)Ix3.s3*(int)Ix3.s3 + (int)Ix3.s4*(int)Ix3.s4 + (int)Ix3.s5*(int)Ix3.s5 + (int)Ix3.s6*(int)Ix3.s6
	      + (int)Ix4.s0*(int)Ix4.s0 + (int)Ix4.s1*(int)Ix4.s1 + (int)Ix4.s2*(int)Ix4.s2 + (int)Ix4.s3*(int)Ix4.s3 + (int)Ix4.s4*(int)Ix4.s4 + (int)Ix4.s5*(int)Ix4.s5 + (int)Ix4.s6*(int)Ix4.s6
	      + (int)Ix5.s0*(int)Ix5.s0 + (int)Ix5.s1*(int)Ix5.s1 + (int)Ix5.s2*(int)Ix5.s2 + (int)Ix5.s3*(int)Ix5.s3 + (int)Ix5.s4*(int)Ix5.s4 + (int)Ix5.s5*(int)Ix5.s5 + (int)Ix5.s6*(int)Ix5.s6
	      + (int)Ix6.s0*(int)Ix6.s0 + (int)Ix6.s1*(int)Ix6.s1 + (int)Ix6.s2*(int)Ix6.s2 + (int)Ix6.s3*(int)Ix6.s3 + (int)Ix6.s4*(int)Ix6.s4 + (int)Ix6.s5*(int)Ix6.s5 + (int)Ix6.s6*(int)Ix6.s6;

	int B = (int)Iy0.s0*(int)Iy0.s0 + (int)Iy0.s1*(int)Iy0.s1 + (int)Iy0.s2*(int)Iy0.s2 + (int)Iy0.s3*(int)Iy0.s3 + (int)Iy0.s4*(int)Iy0.s4 + (int)Iy0.s5*(int)Iy0.s5 + (int)Iy0.s6*(int)Iy0.s6
	      + (int)Iy1.s0*(int)Iy1.s0 + (int)Iy1.s1*(int)Iy1.s1 + (int)Iy1.s2*(int)Iy1.s2 + (int)Iy1.s3*(int)Iy1.s3 + (int)Iy1.s4*(int)Iy1.s4 + (int)Iy1.s5*(int)Iy1.s5 + (int)Iy1.s6*(int)Iy1.s6
	      + (int)Iy2.s0*(int)Iy2.s0 + (int)Iy2.s1*(int)Iy2.s1 + (int)Iy2.s2*(int)Iy2.s2 + (int)Iy2.s3*(int)Iy2.s3 + (int)Iy2.s4*(int)Iy2.s4 + (int)Iy2.s5*(int)Iy2.s5 + (int)Iy2.s6*(int)Iy2.s6
	      + (int)Iy3.s0*(int)Iy3.s0 + (int)Iy3.s1*(int)Iy3.s1 + (int)Iy3.s2*(int)Iy3.s2 + (int)Iy3.s3*(int)Iy3.s3 + (int)Iy3.s4*(int)Iy3.s4 + (int)Iy3.s5*(int)Iy3.s5 + (int)Iy3.s6*(int)Iy3.s6
	      + (int)Iy4.s0*(int)Iy4.s0 + (int)Iy4.s1*(int)Iy4.s1 + (int)Iy4.s2*(int)Iy4.s2 + (int)Iy4.s3*(int)Iy4.s3 + (int)Iy4.s4*(int)Iy4.s4 + (int)Iy4.s5*(int)Iy4.s5 + (int)Iy4.s6*(int)Iy4.s6
	      + (int)Iy5.s0*(int)Iy5.s0 + (int)Iy5.s1*(int)Iy5.s1 + (int)Iy5.s2*(int)Iy5.s2 + (int)Iy5.s3*(int)Iy5.s3 + (int)Iy5.s4*(int)Iy5.s4 + (int)Iy5.s5*(int)Iy5.s5 + (int)Iy5.s6*(int)Iy5.s6
	      + (int)Iy6.s0*(int)Iy6.s0 + (int)Iy6.s1*(int)Iy6.s1 + (int)Iy6.s2*(int)Iy6.s2 + (int)Iy6.s3*(int)Iy6.s3 + (int)Iy6.s4*(int)Iy6.s4 + (int)Iy6.s5*(int)Iy6.s5 + (int)Iy6.s6*(int)Iy6.s6;

	int C = (int)Ix0.s0*(int)Iy0.s0 + (int)Ix0.s1*(int)Iy0.s1 + (int)Ix0.s2*(int)Iy0.s2 + (int)Ix0.s3*(int)Iy0.s3 + (int)Ix0.s4*(int)Iy0.s4 + (int)Ix0.s5*(int)Iy0.s5 + (int)Ix0.s6*(int)Iy0.s6
	      + (int)Ix1.s0*(int)Iy1.s0 + (int)Ix1.s1*(int)Iy1.s1 + (int)Ix1.s2*(int)Iy1.s2 + (int)Ix1.s3*(int)Iy1.s3 + (int)Ix1.s4*(int)Iy1.s4 + (int)Ix1.s5*(int)Iy1.s5 + (int)Ix1.s6*(int)Iy1.s6
	      + (int)Ix2.s0*(int)Iy2.s0 + (int)Ix2.s1*(int)Iy2.s1 + (int)Ix2.s2*(int)Iy2.s2 + (int)Ix2.s3*(int)Iy2.s3 + (int)Ix2.s4*(int)Iy2.s4 + (int)Ix2.s5*(int)Iy2.s5 + (int)Ix2.s6*(int)Iy2.s6
	      + (int)Ix3.s0*(int)Iy3.s0 + (int)Ix3.s1*(int)Iy3.s1 + (int)Ix3.s2*(int)Iy3.s2 + (int)Ix3.s3*(int)Iy3.s3 + (int)Ix3.s4*(int)Iy3.s4 + (int)Ix3.s5*(int)Iy3.s5 + (int)Ix3.s6*(int)Iy3.s6
	      + (int)Ix4.s0*(int)Iy4.s0 + (int)Ix4.s1*(int)Iy4.s1 + (int)Ix4.s2*(int)Iy4.s2 + (int)Ix4.s3*(int)Iy4.s3 + (int)Ix4.s4*(int)Iy4.s4 + (int)Ix4.s5*(int)Iy4.s5 + (int)Ix4.s6*(int)Iy4.s6
	      + (int)Ix5.s0*(int)Iy5.s0 + (int)Ix5.s1*(int)Iy5.s1 + (int)Ix5.s2*(int)Iy5.s2 + (int)Ix5.s3*(int)Iy5.s3 + (int)Ix5.s4*(int)Iy5.s4 + (int)Ix5.s5*(int)Iy5.s5 + (int)Ix5.s6*(int)Iy5.s6
	      + (int)Ix6.s0*(int)Iy6.s0 + (int)Ix6.s1*(int)Iy6.s1 + (int)Ix6.s2*(int)Iy6.s2 + (int)Ix6.s3*(int)Iy6.s3 + (int)Ix6.s4*(int)Iy6.s4 + (int)Ix6.s5*(int)Iy6.s5 + (int)Ix6.s6*(int)Iy6.s6;

	responses[id0+1] = ((float)A * (float)B - (float)C * (float)C - (float)(A+B) * (float)(A+B) * 0.04f) * 3.8477527114806922778496159673236e-16f;
}

