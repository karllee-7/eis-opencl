kernel void wrapImage(read_only image2d_t srcImage, write_only image2d_t dstImage, sampler_t sampler, global float* matrixA){
	constant float16 STP = (float16)(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f);

        int id0 = get_global_id(0) * 15;
        int id1 = get_global_id(1);
	float id0_shift = (float)id0 + (float)((IMAGE_IN_W - IMAGE_OUT_W) * 0.5f);
	float id1_shift = (float)id1 + (float)((IMAGE_IN_H - IMAGE_OUT_H) * 0.5f);

	float16 A = vload16(0, matrixA);

	float x0 = dot(A.s012, (float3)((float)id0_shift, (float)id1_shift, 1.0f));
	float y0 = dot(A.s345, (float3)((float)id0_shift, (float)id1_shift, 1.0f));
	float z0 = dot(A.s678, (float3)((float)id0_shift, (float)id1_shift, 1.0f));

	float16 xx = x0 + A.s0 * STP;
	float16 yy = y0 + A.s3 * STP;
	float16 zz = z0 + A.s6 * STP;

	float16 px = (float16)(x0, xx.s01234567, xx.s89AB, xx.sCDE) / (float16)(z0, zz.s01234567, zz.s89AB, zz.sCDE);
	float16 py = (float16)(y0, yy.s01234567, yy.s89AB, yy.sCDE) / (float16)(z0, zz.s01234567, zz.s89AB, zz.sCDE);

	float4 color0 = read_imagef(srcImage, sampler, (float2)(px.s0, py.s0));
	float4 color1 = read_imagef(srcImage, sampler, (float2)(px.s1, py.s1));
	float4 color2 = read_imagef(srcImage, sampler, (float2)(px.s2, py.s2));
	float4 color3 = read_imagef(srcImage, sampler, (float2)(px.s3, py.s3));
	float4 color4 = read_imagef(srcImage, sampler, (float2)(px.s4, py.s4));
	float4 color5 = read_imagef(srcImage, sampler, (float2)(px.s5, py.s5));
	float4 color6 = read_imagef(srcImage, sampler, (float2)(px.s6, py.s6));
	float4 color7 = read_imagef(srcImage, sampler, (float2)(px.s7, py.s7));
	float4 color8 = read_imagef(srcImage, sampler, (float2)(px.s8, py.s8));
	float4 color9 = read_imagef(srcImage, sampler, (float2)(px.s9, py.s9));
	float4 colorA = read_imagef(srcImage, sampler, (float2)(px.sA, py.sA));
	float4 colorB = read_imagef(srcImage, sampler, (float2)(px.sB, py.sB));
	float4 colorC = read_imagef(srcImage, sampler, (float2)(px.sC, py.sC));
	float4 colorD = read_imagef(srcImage, sampler, (float2)(px.sD, py.sD));
	float4 colorE = read_imagef(srcImage, sampler, (float2)(px.sE, py.sE));

	write_imagef(dstImage, (int2)(id0+0, id1), color0);
	write_imagef(dstImage, (int2)(id0+1, id1), color1);
	write_imagef(dstImage, (int2)(id0+2, id1), color2);
	write_imagef(dstImage, (int2)(id0+3, id1), color3);
	write_imagef(dstImage, (int2)(id0+4, id1), color4);
	write_imagef(dstImage, (int2)(id0+5, id1), color5);
	write_imagef(dstImage, (int2)(id0+6, id1), color6);
	write_imagef(dstImage, (int2)(id0+7, id1), color7);
	write_imagef(dstImage, (int2)(id0+8, id1), color8);
	write_imagef(dstImage, (int2)(id0+9, id1), color9);
	write_imagef(dstImage, (int2)(id0+10, id1), colorA);
	write_imagef(dstImage, (int2)(id0+11, id1), colorB);
	write_imagef(dstImage, (int2)(id0+12, id1), colorC);
	write_imagef(dstImage, (int2)(id0+13, id1), colorD);
	write_imagef(dstImage, (int2)(id0+14, id1), colorE);
}

