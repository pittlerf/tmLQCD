#ifndef _BGQ_SU3_H
#define _BGQ_SU3_H

#define _bgq_declare_su3regs() \
  vector4double ALIGN a00, a01, a02, a10, a11, a12, a20, a21, a22; \
  vector4double ALIGN b00, b01, b02, b10, b11, b12, b20, b21, b22; \
  vector4double ALIGN out00, out01, out02, out10, out11, out12, out20, out21, out22; \
  vector4double ALIGN t1, t2, t3; 

#define _bgq_ld_su3(r00, r01, r02, r10, r11, r12, r20, r21, r22, u) \
  r00 = vec_ld2(0L, (double*) &(u).c00 ); \
  r01 = vec_ld2(0L, (double*) &(u).c01 ); \
  r02 = vec_ld2(0L, (double*) &(u).c02 ); \
  r10 = vec_ld2(0L, (double*) &(u).c10 ); \
  r11 = vec_ld2(0L, (double*) &(u).c11 ); \
  r12 = vec_ld2(0L, (double*) &(u).c12 ); \
  r20 = vec_ld2(0L, (double*) &(u).c20 ); \
  r21 = vec_ld2(0L, (double*) &(u).c21 ); \
  r22 = vec_ld2(0L, (double*) &(u).c22 );
  
#define _bgq_store_su3(u,u00,u01,u02,u10,u11,u12,u20,u21,u22) \
  vec_st2(u00, 0L, (double*) &(u).c00); \
  vec_st2(u01, 16L, (double*) &(u).c00); \
  vec_st2(u02, 32L, (double*) &(u).c00); \
  vec_st2(u10, 48L, (double*) &(u).c00); \
  vec_st2(u11, 64L, (double*) &(u).c00); \
  vec_st2(u12, 80L, (double*) &(u).c00); \
  vec_st2(u20, 96L, (double*) &(u).c00); \
  vec_st2(u21, 112L, (double*) &(u).c00); \
  vec_st2(u22, 128L, (double*) &(u).c00);
  
#define _bgq_complex_times_complex(out,in1,in2,temp) \
  temp = vec_xmul(in1, in2); \
  out = vec_xxnpmadd(in2, in1, temp);
  
#define _bgq_complex_times_complex_conj(out,in1,in2,temp) \
  temp = vec_xmul(in2, in1); \
  out = vec_xxcpnmadd(in1, in2, temp);
  
#define _bgq_su3_vec_times_vec_sum(out, a1, a2, a3, b1, b2, b3, temp1, temp2, temp3) \
  _bgq_complex_times_complex(temp1, a1, b1, temp2) \
  out = temp1; \
  _bgq_complex_times_complex(temp1, a2, b2, temp2) \
  temp3 = vec_add(out, temp1); \
  _bgq_complex_times_complex(temp1, a3, b3, temp2) \
  out = vec_add(temp3, temp1);
  
#define _bgq_su3_vec_times_vec_conj_sum(out, a1, a2, a3, b1, b2, b3, temp1, temp2, temp3) \
  _bgq_complex_times_complex_conj(temp1, a1, b1, temp2) \
  out = temp1; \
  _bgq_complex_times_complex_conj(temp1, a2, b2, temp2) \
  temp3 = vec_add(out, temp1); \
  _bgq_complex_times_complex_conj(temp1, a3, b3, temp2) \
  out = vec_add(temp3, temp1);
  
#define _bgq_su3_times_su3(u,v,w) \
  _bgq_ld_su3(a00, a01, a02, a10, a11, a12, a20, a21, a22, v) \
  _bgq_ld_su3(b00, b01, b02, b10, b11, b12, b20, b21, b22, w) \
  \
  _bgq_su3_vec_times_vec_sum(out00,a00,a01,a02,b00,b10,b20,t1,t2,t3) \
  _bgq_su3_vec_times_vec_sum(out01,a00,a01,a02,b01,b11,b21,t1,t2,t3) \
  _bgq_su3_vec_times_vec_sum(out02,a00,a01,a02,b02,b12,b22,t1,t2,t3) \
  _bgq_su3_vec_times_vec_sum(out10,a10,a11,a12,b00,b10,b20,t1,t2,t3) \
  _bgq_su3_vec_times_vec_sum(out11,a10,a11,a12,b01,b11,b21,t1,t2,t3) \
  _bgq_su3_vec_times_vec_sum(out12,a10,a11,a12,b02,b12,b22,t1,t2,t3) \
  _bgq_su3_vec_times_vec_sum(out20,a20,a21,a22,b00,b10,b20,t1,t2,t3) \
  _bgq_su3_vec_times_vec_sum(out21,a20,a21,a22,b01,b11,b21,t1,t2,t3) \
  _bgq_su3_vec_times_vec_sum(out22,a20,a21,a22,b02,b12,b22,t1,t2,t3) \
  \
  _bgq_store_su3(u,out00,out01,out02,out10,out11,out12,out20,out21,out22)
  
#define _bgq_su3_times_su3d(u,v,w) \
  _bgq_ld_su3(a00, a01, a02, a10, a11, a12, a20, a21, a22, v) \
  _bgq_ld_su3(b00, b01, b02, b10, b11, b12, b20, b21, b22, w) \
  \
  _bgq_su3_vec_times_vec_conj_sum(out00,a00,a01,a02,b00,b01,b02,t1,t2,t3) \
  _bgq_su3_vec_times_vec_conj_sum(out01,a00,a01,a02,b10,b11,b12,t1,t2,t3) \
  _bgq_su3_vec_times_vec_conj_sum(out02,a00,a01,a02,b20,b21,b22,t1,t2,t3) \
  _bgq_su3_vec_times_vec_conj_sum(out10,a10,a11,a12,b00,b01,b02,t1,t2,t3) \
  _bgq_su3_vec_times_vec_conj_sum(out11,a10,a11,a12,b10,b11,b12,t1,t2,t3) \
  _bgq_su3_vec_times_vec_conj_sum(out12,a10,a11,a12,b20,b21,b22,t1,t2,t3) \
  _bgq_su3_vec_times_vec_conj_sum(out20,a20,a21,a22,b00,b01,b02,t1,t2,t3) \
  _bgq_su3_vec_times_vec_conj_sum(out21,a20,a21,a22,b10,b11,b12,t1,t2,t3) \
  _bgq_su3_vec_times_vec_conj_sum(out22,a20,a21,a22,b20,b21,b22,t1,t2,t3) \
  \
  _bgq_store_su3(u,out00,out01,out02,out10,out11,out12,out20,out21,out22)
  
#define _su3_times_su3 _bgq_su3_times_su3  
#define _su3_times_su3d _bgq_su3_times_su3d 

#endif /* _BGQ_SU3_H */
