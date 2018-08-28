kernel void prepareImage(read_only image2d_t srcImage, global uchar* dstGray, sampler_t sampler, global float* matrixA, write_only image2d_t debugImg){
	int x = get_global_id(0);
	int y = get_global_id(1);
	int stride = get_global_size(0);
	float16 A = vload16(0, matrixA);
	float2 P_o;
	{
		float3 a = (float3)((float)(x*2-IMAGE_IN_W/2), (float)(y*2-IMAGE_IN_H/2), (float)CAM_PAM_F);
		float3 b = (float3)(dot(A.s012, a), dot(A.s345, a), dot(A.s678, a));
		float scale = (float)CAM_PAM_F/b.z;
		P_o = b.xy * scale + (float2)(IMAGE_IN_W/2, IMAGE_IN_H/2);
	}

	float4 color_rgb = read_imagef(srcImage, sampler, P_o);
	float  color_gray = color_rgb.s0 * 0.299f + color_rgb.s1 * 0.587f + color_rgb.s2 * 0.114f;
        *(dstGray + stride * y + x) = (uchar)(color_gray * 255.0f);

	if(x < IMAGE_OUT_W && y < IMAGE_OUT_H){
		write_imagef(debugImg, (int2)(x, y), (float4)(color_gray, color_gray, color_gray, 0.0f));
	}
}
/*===============================================================================================================================*/
kernel void gaussFiler(global uchar* srcImg, global uchar* dstImg, write_only image2d_t debugImg){
	int x0 = get_global_id(0);
	int y0 = get_global_id(1);
	int stride = get_global_size(0);
	int x1 = x0 * 2 + (x0 == 0);
	int y1 = y0 * 2 + (y0 == 0);

	uchar3 L0 = vload3(0, srcImg + stride * 2 * (y1-1) + (x1-1));
	uchar3 L1 = vload3(0, srcImg + stride * 2 * (y1+0) + (x1-1));
	uchar3 L2 = vload3(0, srcImg + stride * 2 * (y1+1) + (x1-1));

	ushort3 XL0 = convert_ushort3(L0) * (ushort3)((ushort)1, (ushort)2, (ushort)1);
	ushort3 XL1 = convert_ushort3(L1) * (ushort3)((ushort)2, (ushort)4, (ushort)2);
	ushort3 XL2 = convert_ushort3(L2) * (ushort3)((ushort)1, (ushort)2, (ushort)1);

	ushort s = XL0.s0 + XL0.s1 + XL0.s2 + XL1.s0 + XL1.s1 + XL1.s2 + XL2.s0 + XL2.s1 + XL2.s2;
        *(dstImg + stride * y0 + x0) = (uchar)(s/16);
	if(x0 < IMAGE_OUT_W && y0 < IMAGE_OUT_H){
		float color = (float)(s/16) / 256.0f;
		write_imagef(debugImg, (int2)(x0, y0), (float4)(color, color, color, 0.0f));
	}
}
/*===============================================================================================================================*/
kernel void wrapImage(read_only image2d_t srcImage, write_only image2d_t dstImage, sampler_t sampler, global float* matrixA){
	int x0 = get_global_id(0);
	int y0 = get_global_id(1);
	float16 A = vload16(0, matrixA);
	float3 P0 = (float3)((float)(x0-IMAGE_OUT_W/2), (float)(y0-IMAGE_OUT_H/2), (float)CAM_PAM_F);
	float3 P1 = (float3)(dot(A.s012, P0), dot(A.s345, P0), dot(A.s678, P0));
	float scale = (float)CAM_PAM_F/P1.z;
	float2 P2 = P1.xy * scale + (float2)(IMAGE_IN_W/2, IMAGE_IN_H/2);

	float4 color0 = read_imagef(srcImage, sampler, P2);
	write_imagef(dstImage, (int2)(x0, y0), color0);
}
/*===============================================================================================================================*/
/* kernel void cvtColor_gray(read_only image2d_t srcImage, sampler_t sampler, global uchar* gray)
{
        int x = get_global_id(0);
        int y = get_global_id(1);
        float4 color_rgba = read_imagef(srcImage, sampler, (float2)((float)x, (float)y));
        float  color_gray = color_rgba.s0 * 0.299f + color_rgba.s1 * 0.587f + color_rgba.s2 * 0.114f;
        *(gray + get_global_size(0) * y + x) = (uchar)color_gray;
}*/
/*===============================================================================================================================*/
kernel void cvtColor_gray(read_only image2d_t srcImage, global uchar* dstGray, sampler_t sampler, write_only image2d_t debugImg){
	int x = get_global_id(0);
	int y = get_global_id(1);
	int stride = get_global_size(0);
	float4 color_rgb = read_imagef(srcImage, sampler, (float2)((float)(x*2), (float)(y*2)));
	float  color_gray = color_rgb.s0 * 0.299f + color_rgb.s1 * 0.587f + color_rgb.s2 * 0.114f;
        *(dstGray + stride * y + x) = (uchar)(color_gray * 255.0f);

	if(x < IMAGE_OUT_W && y < IMAGE_OUT_H){
		write_imagef(debugImg, (int2)(x, y), (float4)(color_gray, color_gray, color_gray, 0.0f));
	}
}
/*===============================================================================================================================*/

/*===============================================================================================================================*/

