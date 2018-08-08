kernel void cvtColor_gray(global ushort* img, global uchar* gray)
{
	int id0 = get_global_id(0);
	int id1 = get_global_id(1);

	ushort8 p0 = vload8(id0, img + (id1 * 2) * IMAGE_IN_W);

	ushort4 color_r = (p0.even & (ushort)0xF800);
	ushort4 color_g = (p0.even & (ushort)0x07E0);
	ushort4 color_b = (p0.even & (ushort)0x001F);

	float4 color = convert_float4(color_r) * (float)(0.299f * 0.00390625f) + convert_float4(color_g) * (float)(0.587f * 0.125f) + convert_float4(color_b) * (float)(0.114f * 8.0f);

	vstore4(convert_uchar4(color), id0, gray + id1 * GRAY_W);
}

kernel void cvtColor_gauss(global uchar* img, write_only image2d_t dstImage, sampler_t sampler){
	int x = get_global_id(0);
	int y = get_global_id(1);

	if( (x<1) || (x>=(GRAY_W/3-1)) || (y<2) || (y>=(GRAY_H-2)))
		return; 

	global uchar* _img = img + y*GRAY_W + x*3 - 2;
	uchar8 t0 = vload8(0, _img - 2 * GRAY_W);
	uchar8 t1 = vload8(0, _img - 1 * GRAY_W);
	uchar8 t2 = vload8(0, _img + 0 * GRAY_W);
	uchar8 t3 = vload8(0, _img + 1 * GRAY_W);
	uchar8 t4 = vload8(0, _img + 2 * GRAY_W);

	ushort8 tt0 = convert_ushort8(t0);
	ushort8 tt1 = convert_ushort8(t1);
	ushort8 tt2 = convert_ushort8(t2);
	ushort8 tt3 = convert_ushort8(t3);
	ushort8 tt4 = convert_ushort8(t4);

	uint p0 = (tt0.s0 + tt0.s4 + tt4.s0 + tt4.s4)*1 + (tt0.s1 + tt0.s3 + tt1.s0 + tt1.s4     +  tt3.s0 + tt3.s4 + tt4.s1 + tt4.s3)*4
	          + (tt0.s2 + tt2.s0 + tt2.s4 + tt4.s2)*7 + (tt1.s1 + tt1.s3 + tt3.s1 + tt3.s3)*16 + (tt1.s2 + tt2.s1 + tt2.s3 + tt3.s2)*26 + tt2.s2*41;

	uint p1 = (tt0.s1 + tt0.s5 + tt4.s1 + tt4.s5)*1 + (tt0.s2 + tt0.s4 + tt1.s1 + tt1.s5     +  tt3.s1 + tt3.s5 + tt4.s2 + tt4.s4)*4
	          + (tt0.s3 + tt2.s1 + tt2.s5 + tt4.s3)*7 + (tt1.s2 + tt1.s4 + tt3.s2 + tt3.s4)*16 + (tt1.s3 + tt2.s2 + tt2.s4 + tt3.s3)*26 + tt2.s3*41;
	
	uint p2 = (tt0.s2 + tt0.s6 + tt4.s2 + tt4.s6)*1 + (tt0.s3 + tt0.s5 + tt1.s2 + tt1.s6     +  tt3.s2 + tt3.s6 + tt4.s3 + tt4.s5)*4
	          + (tt0.s4 + tt2.s2 + tt2.s6 + tt4.s4)*7 + (tt1.s3 + tt1.s5 + tt3.s3 + tt3.s5)*16 + (tt1.s4 + tt2.s3 + tt2.s5 + tt3.s4)*26 + tt2.s4*41;
	
	float3 p = convert_float3((uint3)(p0, p1, p2)) * 1.4308608058608058608058608058608e-5f;

	write_imagef(dstImage, (int2)(x*3+0, y), (float4)(p.s0, 0.0f, 0.0f, 1.0f));
	write_imagef(dstImage, (int2)(x*3+1, y), (float4)(p.s1, 0.0f, 0.0f, 1.0f));
	write_imagef(dstImage, (int2)(x*3+2, y), (float4)(p.s2, 0.0f, 0.0f, 1.0f));
}
