kernel void cvtColor_gray(read_only image2d_t srcImage, sampler_t sampler, global uchar* gray)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	float4 color_rgba = read_imagef(srcImage, sampler, (float2)((float)x, (float)y));
	float  color_gray = color_rgba.s0 * 0.299f + color_rgba.s1 * 0.587f + color_rgba.s2 * 0.114f;
	*(gray + get_global_size(0) * y + x) = (uchar)color_gray;
}

