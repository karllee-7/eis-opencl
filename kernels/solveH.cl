kernel void solveH(global float * matches, int matches_num, global float * matrixA, global int * evaluate){
        int id0 = get_global_id(0);
        int id1 = get_global_id(1);

        if(id0 >= matches_num)
                return ;

        float4 P0 = vload4(id0, matches);
        float16 H = vload16(id1, matrixA);

	float3 P1_t = H.s036 * P0.s0 + H.s147 * P0.s1 + H.s258;
	float2 P1 = P1_t.s01 / P1_t.s2;

        float deviation = distance(P1, P0.s23);
	if(deviation > SOLVE_MATRIXA_RANSAC_PRECISE)
                return ;

        atomic_inc(&evaluate[id1]);
}
