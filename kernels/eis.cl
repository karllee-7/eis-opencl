kernel void prepareImage(read_only image2d_t srcImage, global uchar* dstGray, sampler_t sampler, global float* matrixA, write_only image2d_t debugImg){
	int x = get_global_id(0);
	int y = get_global_id(1);
	int stride = get_global_size(0);
	float16 A = vload16(0, matrixA);
	float2 P_o;
	{
		float3 a = (float3)((float)(x-IMAGE_IN_W/2), (float)(y-IMAGE_IN_H/2), (float)CAM_PAM_F);
		float3 b = (float3)(dot(A.s012, a), dot(A.s345, a), dot(A.s678, a));
		float scale = (float)CAM_PAM_F/b.z;
		P_o = b.xy * scale + (float2)(IMAGE_IN_W/2, IMAGE_IN_H/2);
	}

	float4 color_rgb = read_imagef(srcImage, sampler, P_o);
	float  color_gray = color_rgb.s0 * 0.299f + color_rgb.s1 * 0.587f + color_rgb.s2 * 0.114f;
        *(dstGray + stride * y + x) = (uchar)color_gray;

	if(x < IMAGE_OUT_W && y < IMAGE_OUT_H){
		write_imagef(debugImg, (int2)(x, y), (float4)(color_gray, color_gray, color_gray, color_rgb.s3));
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
kernel void cvtColor_gray(read_only image2d_t srcImage, sampler_t sampler, global uchar* gray)
{
        int x = get_global_id(0);
        int y = get_global_id(1);
        float4 color_rgba = read_imagef(srcImage, sampler, (float2)((float)x, (float)y));
        float  color_gray = color_rgba.s0 * 0.299f + color_rgba.s1 * 0.587f + color_rgba.s2 * 0.114f;
        *(gray + get_global_size(0) * y + x) = (uchar)color_gray;
}
/*===============================================================================================================================*/

