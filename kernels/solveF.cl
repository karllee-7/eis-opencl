kernel void solveF(global float * matches, int matches_num, global float * matrixF, global int * evaluate){
        int id0 = get_global_id(0);
        int id1 = get_global_id(1);

        if(id0 >= matches_num)
                return ;

        float4  P = vload4(id0, matches);
        float16 F = vload16(id1, matrixF);

	float line_0 = dot(F.s012, (float3)(P.s01, 1.0f));
	float line_1 = dot(F.s345, (float3)(P.s01, 1.0f));
	float line_2 = dot(F.s678, (float3)(P.s01, 1.0f));
	float3 line = normalize((float3)(line_0, line_1, line_2));

	float e = dot((float3)(P.s23, 1.0f), line);
	float e_abs = fabs(e);

	if(e_abs > SOLVE_MATRIXF_RANSAC_PRECISE)
                return ;

        atomic_inc(&evaluate[id1*2+0]);
	atomic_add(&evaluate[id1*2+1], convert_int(e_abs*65536.0f));
}
