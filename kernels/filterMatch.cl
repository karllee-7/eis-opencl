kernel void filterMatch(global int * matches, int matches_num, global float * matrixA, global int * confidence){
        int id0 = get_global_id(0);
        int id1 = get_global_id(1);

	if(id0 >= matches_num)
		return ;

        float4 match = convert_float4(vload4(id0+8, matches));
        float8 A = vload8(id1, matrixA);

	float deviation = distance(A.s03 * match.s0 + A.s14 * match.s1 + A.s25, match.s23);

	if(deviation > 3.0f)
		return ;

        atomic_inc(&confidence[id1]);
}
