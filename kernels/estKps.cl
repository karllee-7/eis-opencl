kernel void estKps(global uchar * _img, global int* _offset, global int* _kps)
{
	/*=======================================================================*/
	int id0 = get_global_id(0);
	int id1 = get_global_id(1);
	int id2 = get_global_id(2);

	int2 offset = vload2(id2, _offset);
	global uchar* img = _img + (offset.y + id1) * GRAY_W + (offset.x + id0) - 3;

	short VL, VH;
	ushort m0, m1;

	#define MASK_BIT(X) (1<<X)
	/*=======================================================================*/
	{
		uchar8 t0 = vload8(0, img);

		VL = convert_short(t0.s3) - KPS_EST_THRESHOLD;
		VH = convert_short(t0.s3) + KPS_EST_THRESHOLD;
	
		short8 L_0 = convert_short8(t0) < VL;
		short8 H_0 = convert_short8(t0) > VH;

		m0 = (L_0.s0 & MASK_BIT(0)) | (L_0.s6 & MASK_BIT(8));
		m1 = (H_0.s0 & MASK_BIT(0)) | (H_0.s6 & MASK_BIT(8));
	}

	if((m0 | m1) == 0)
		return;
	/*=======================================================================*/
	{
		uchar8 t0 = vload8(0, img - 2 * GRAY_W);
		uchar8 t1 = vload8(0, img + 2 * GRAY_W);
		uchar8 t2 = vload8(0, img - 3 * GRAY_W);
		uchar8 t3 = vload8(0, img + 3 * GRAY_W);

		short8 L_0 = convert_short8((uchar8)(t0.s1, t2.s2, t2.s3, t2.s4, t0.s5, 0, 0, 0)) < VL;
		short8 L_1 = convert_short8((uchar8)(t1.s5, t3.s4, t3.s3, t3.s2, t1.s1, 0, 0, 0)) < VL;

		short8 H_0 = convert_short8((uchar8)(t0.s1, t2.s2, t2.s3, t2.s4, t0.s5, 0, 0, 0)) > VH;
		short8 H_1 = convert_short8((uchar8)(t1.s5, t3.s4, t3.s3, t3.s2, t1.s1, 0, 0, 0)) > VH;
		
		m0 |= (L_0.s0 & MASK_BIT(2))  | (L_0.s1 & MASK_BIT(3))  | (L_0.s2 & MASK_BIT(4))  | (L_0.s3 & MASK_BIT(5))  | (L_0.s4 & MASK_BIT(6))
		    | (L_1.s0 & MASK_BIT(10)) | (L_1.s1 & MASK_BIT(11)) | (L_1.s2 & MASK_BIT(12)) | (L_1.s3 & MASK_BIT(13)) | (L_1.s4 & MASK_BIT(14));

		m1 |= (H_0.s0 & MASK_BIT(2))  | (H_0.s1 & MASK_BIT(3))  | (H_0.s2 & MASK_BIT(4))  | (H_0.s3 & MASK_BIT(5))  | (H_0.s4 & MASK_BIT(6))
		    | (H_1.s0 & MASK_BIT(10)) | (H_1.s1 & MASK_BIT(11)) | (H_1.s2 & MASK_BIT(12)) | (H_1.s3 & MASK_BIT(13)) | (H_1.s4 & MASK_BIT(14));
	}

	if( ((m0|(m0>>8))&0x7D) != 0x7D && ((m1|(m1>>8))&0x7D) != 0x7D )
		return ;
	/*====================================================================*/
	{
		uchar8 t0 = vload8(0, img - GRAY_W);
		uchar8 t1 = vload8(0, img + GRAY_W);

		short4 L_0 = convert_short4((uchar4)(t0.s0, t0.s6, t1.s6, t1.s0)) < VL;
		short4 H_0 = convert_short4((uchar4)(t0.s0, t0.s6, t1.s6, t1.s0)) > VH;

		m0 |= (L_0.s0 & MASK_BIT(1)) | (L_0.s1 & MASK_BIT(7)) | (L_0.s2 & MASK_BIT(9)) | (L_0.s3 & MASK_BIT(15));
		m1 |= (H_0.s0 & MASK_BIT(1)) | (H_0.s1 & MASK_BIT(7)) | (H_0.s2 & MASK_BIT(9)) | (H_0.s3 & MASK_BIT(15));
	}
	if( ((m0|(m0>>8))&0xFF) != 0xFF && ((m1|(m1>>8))&0xFF) != 0xFF )
		return ;
	/*====================================================================*/
	{
		uint xm0 = (((uint)m0)<<16) | (uint)m0;
		uint xm1 = (((uint)m1)<<16) | (uint)m1;

		#define CHECK0(i) ((xm0 & (511 << i)) == (511 << i))
		#define CHECK1(i) ((xm1 & (511 << i)) == (511 << i))	

		if( CHECK0(0) + CHECK0(1) + CHECK0(2) + CHECK0(3) +
		    CHECK0(4) + CHECK0(5) + CHECK0(6) + CHECK0(7) +
		    CHECK0(8) + CHECK0(9) + CHECK0(10) + CHECK0(11) +
		    CHECK0(12) + CHECK0(13) + CHECK0(14) + CHECK0(15) +

		    CHECK1(0) + CHECK1(1) + CHECK1(2) + CHECK1(3) +
		    CHECK1(4) + CHECK1(5) + CHECK1(6) + CHECK1(7) +
		    CHECK1(8) + CHECK1(9) + CHECK1(10) + CHECK1(11) +
		    CHECK1(12) + CHECK1(13) + CHECK1(14) + CHECK1(15) == 0 )
			return ;
	}
	/*====================================================================*/
	{
		global int* kps = _kps + (BLOCK_EST_MAXNUM + 1) * 2 * id2;
		int num = atomic_inc(&kps[0]);
		if(num >= BLOCK_EST_MAXNUM)
			return;

		vstore2((int2)(offset.x + id0, offset.y + id1), num+1, kps);
	}
}
/*====================================================================*/
