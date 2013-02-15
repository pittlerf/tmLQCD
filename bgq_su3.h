/**********************************************************************
 *
 * Copyright (C) 2013 Bartosz Kostrzewa
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 *
 * These are routines and macros for using half of the available
 * four floating point units of the BG/Q processor
 *
 **********************************************************************/

#ifndef _BGQ_SU3_H
#define _BGQ_SU3_H

// declare 30 registers for su3 multiplication
// aRC hold left input
// bRC hold right input
// outRC hold output 

#define _bgq_declare_su3regs() \
  vector4double ALIGN a00, a01, a02, a10, a11, a12, a20, a21, a22; \
  vector4double ALIGN b00, b01, b02, b10, b11, b12, b20, b21, b22; \
  vector4double ALIGN out00, out01, out02, out10, out11, out12, out20, out21, out22; \
  vector4double ALIGN t1, t2, t3; 

// load su3 matrix into 9 vector4double
// u is of type su3

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
  
// store su3 matrix from 9 vector4double  
// u is of type su3
  
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

// multiply one complex number with another or two complex numbers with two others

#define _bgq_complex_times_complex(out,in1,in2,temp) \
  temp = vec_xmul(in1, in2); \
  out = vec_xxnpmadd(in2, in1, temp);
  
// multiply one or two complex numbers with the conjugate(s) of the second input
  
#define _bgq_complex_times_complex_conj(out,in1,in2,temp) \
  temp = vec_xmul(in2, in1); \
  out = vec_xxcpnmadd(in1, in2, temp);
  
// multiply a vector of complex numbers stored in a1, a2, a3 with the vector in b1, b2, b3
// element by element and sum the result into out  
  
#define _bgq_su3_vec_times_vec_sum(out, a1, a2, a3, b1, b2, b3, temp1, temp2, temp3) \
  _bgq_complex_times_complex(temp1, a1, b1, temp2) \
  out = temp1; \
  _bgq_complex_times_complex(temp1, a2, b2, temp2) \
  temp3 = vec_add(out, temp1); \
  _bgq_complex_times_complex(temp1, a3, b3, temp2) \
  out = vec_add(temp3, temp1);

// multiply a vector of complex numbers stored in a1, a2, a3 with the vector in b1, b2, b3
// element by element and sum the result into out, the elements of the second input are
// conjugated

#define _bgq_su3_vec_times_vec_conj_sum(out, a1, a2, a3, b1, b2, b3, temp1, temp2, temp3) \
  _bgq_complex_times_complex_conj(temp1, a1, b1, temp2) \
  out = temp1; \
  _bgq_complex_times_complex_conj(temp1, a2, b2, temp2) \
  temp3 = vec_add(out, temp1); \
  _bgq_complex_times_complex_conj(temp1, a3, b3, temp2) \
  out = vec_add(temp3, temp1);

// multiply a vector of complex numbers stored in a1, a2, a3 with the vector in b1, b2, b3
// element by element and sum the result into out, the elements of the first input are
// conjugated

#define _bgq_su3_vec_conj_times_vec_sum(out, a1, a2, a3, b1, b2, b3, temp1, temp2, temp3) \
  _bgq_complex_times_complex_conj(temp1, b1, a1, temp2) \
  out = temp1; \
  _bgq_complex_times_complex_conj(temp1, b2, a2, temp2) \
  temp3 = vec_add(out, temp1); \
  _bgq_complex_times_complex_conj(temp1, b3, a3, temp2) \
  out = vec_add(temp3, temp1);

#define _bgq_su3_plus_su3() \
  out00 = vec_add(a00,b00); \
  out01 = vec_add(a01,b01); \
  out02 = vec_add(a02,b02); \
  out10 = vec_add(a10,b10); \
  out11 = vec_add(a11,b11); \
  out12 = vec_add(a12,b12); \
  out20 = vec_add(a20,b20); \
  out21 = vec_add(a21,b21); \
  out22 = vec_add(a22,b22);
      
#define _bgq_su3_acc(u,v) \
  _bgq_ld_su3(a00, a01, a02, a10, a11, a12, a20, a21, a22, u) \
  _bgq_ld_su3(b00, b01, b02, b10, b11, b12, b20, b21, b22, v) \
  _bgq_su3_plus_su3() \
  _bgq_store_su3(u,out00,out01,out02,out10,out11,out12,out20,out21,out22)

#define _bgq_su3_acc_regs(r00,r01,r02,r10,r11,r12,r20,r21,r22,s00,s01,s02,s10,s11,s12,s20,s21,s22) \
  r00 = vec_add(r00,s00); \
  r01 = vec_add(r01,s01); \
  r02 = vec_add(r02,s02); \
  r10 = vec_add(r10,s10); \
  r11 = vec_add(r11,s11); \
  r12 = vec_add(r12,s12); \
  r20 = vec_add(r20,s20); \
  r21 = vec_add(r21,s21); \
  r22 = vec_add(r22,s22);
  
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
  
#define _bgq_su3_times_su3_acc(u,v,w) \
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
  _bgq_ld_su3(a00, a01, a02, a10, a11, a12, a20, a21, a22, u) \
  _bgq_su3_acc_regs(out00,out01,out02,out10,out11,out12,out20,out21,out22,a00,a01,a02,a10,a11,a12,a20,a21,a22) \
  _bgq_store_su3(u,out00,out01,out02,out10,out11,out12,out20,out21,out22)

#define _su3_times_su3 _bgq_su3_times_su3
#define _su3_times_su3d _bgq_su3_times_su3d
#define _su3_times_su3_acc _bgq_su3_times_su3_acc

#endif /* _BGQ_SU3_H */
