/**********************************************************************
 *
 * Copyright (C) 2012 Carsten Urbach
 *
 * BG and halfspinor versions (C) 2007, 2008 Carsten Urbach
 *
 * This file is based on an implementation of the Dirac operator 
 * written by Martin Luescher, modified by Martin Hasenbusch in 2002 
 * and modified and extended by Carsten Urbach from 2003-2008
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
 **********************************************************************/

#if (defined BGQ && defined XLC)

#define _declare_regs()							\
  vector4double ALIGN r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11; \
  vector4double ALIGN rs0, rs1, rs2, rs3, rs4, rs5, rs6, rs7, rs8, rs9, rs10, rs11; \
  vector4double ALIGN U0, U1, U2, U3, U4, U6, U7;			\
  vector4double ALIGN rtmp;							\
  __alignx(16,P);							\
  __alignx(16,Q);

#define _1_imu() \
  rtmp = vec_ld2(0, (double*) &rho); \
  _vec_load_spinor(r6, r7, r8, r9, r10, r11, s->s0); \
  _vec_cmplx_mul_double2c(r0, r1, r2, r6, r7, r8, rtmp); \
  _vec_cmplxcg_mul_double2c(r6, r7, r8, r9, r10, r11, rtmp); \
  _vec_unfuse(r0, r1, r2, r3, r4, r5); \
  _vec_unfuse(r6, r7, r8, r9, r10, r11); \
  _vec_add_double2(rs0, rs1, rs2, rs3, rs4, rs5, r0, r1, r2, r3, r4, r5); \
  _vec_add_double2(rs6, rs7, rs8, rs9, rs10, rs11, r6, r7, r8, r9, r10, r11);  
  
#define _p0add()							\
  _vec_load_spinor(r4, r5, r6, r7, r8, r9, sp->s0);			\
  _vec_add_ul_spinor(r0, r1, r2, r4, r5, r6, r7, r8, r9);		\
  _vec_su3_multiply_double2ct(up);					\
  rtmp = vec_ld2(0, (double*) &phase_0);					\
  _vec_cmplx_mul_double2c(rs0, rs1, rs2, r4, r5, r6, rtmp);		\
  _vec_unfuse(rs0, rs1, rs2, rs3, rs4, rs5);				\
  rs6 = rs0; rs7 = rs1; rs8 = rs2;					\
  rs9 = rs3; rs10= rs4; rs11= rs5;

#define _m0add()							\
  _vec_load_spinor(r4, r5, r6, r7, r8, r9, sm->s0);			\
  _vec_sub_ul_spinor(r0, r1, r2, r4, r5, r6, r7, r8, r9);		\
  _vec_su3_inverse_multiply_double2ct(um);				\
  _vec_cmplxcg_mul_double2c(r0, r1, r2, r4, r5, r6, rtmp);		\
  _vec_unfuse(r0, r1, r2, r3, r4, r5);					\
  _vec_add_double2(rs0, rs1, rs2, rs3, rs4, rs5, r0, r1, r2, r3, r4, r5); \
  _vec_sub_double2(rs6, rs7, rs8, rs9, rs10, rs11, r0, r1, r2, r3, r4, r5);

#define _p1add()							\
  _vec_load(r4, r5, sp->s0);						\
  _vec_load16(r6, r7, sp->s1, U0);					\
  _vec_load(r10, r11, sp->s2);						\
  _vec_load16(r8, r9, sp->s3, U0);					\
  _vec_i_mul_add(r0, r1, r4, r5, r8, r9, U0);				\
  _vec_i_mul_add(r2, r3, r6, r7, r10, r11, U0);				\
  _vec_su3_multiply_double2c(up);					\
  rtmp = vec_ld2(0, (double*) &phase_1);					\
  _vec_cmplx_mul_double2c(r0, r1, r2, r4, r5, r6, rtmp);		\
  _vec_unfuse(r0, r1, r2, r3, r4, r5);					\
  _vec_add_double2(rs0, rs1, rs2, rs3, rs4, rs5, r0, r1, r2, r3, r4, r5); \
  _vec_i_mul_sub2(rs6, rs7, rs8, r3, r4, r5, U0);			\
  _vec_i_mul_sub2(rs9, rs10, rs11, r0, r1, r2, U1);

#define _m1add()							\
  _vec_load(r4, r5, sm->s0);						\
  _vec_load16(r6, r7, sm->s1, U0);					\
  _vec_load(r10, r11, sm->s2);						\
  _vec_load16(r8, r9, sm->s3, U0);					\
  _vec_i_mul_sub(r0, r1, r4, r5, r8, r9, U0);				\
  _vec_i_mul_sub(r2, r3, r6, r7, r10, r11, U0);				\
  _vec_su3_inverse_multiply_double2c(um);				\
  _vec_cmplxcg_mul_double2c(r0, r1, r2, r4, r5, r6, rtmp);		\
  _vec_unfuse(r0, r1, r2, r3, r4, r5);					\
  _vec_add_double2(rs0, rs1, rs2, rs3, rs4, rs5, r0, r1, r2, r3, r4, r5); \
  _vec_i_mul_add2(rs6, rs7, rs8, r3, r4, r5, U0);			\
  _vec_i_mul_add2(rs9, rs10, rs11, r0, r1, r2, U1);

#define _p2add()							\
  _vec_load(r4, r5, sp->s0);						\
  _vec_load16(r6, r7, sp->s1, U0);					\
  _vec_load(r10, r11, sp->s2);						\
  _vec_load16(r8, r9, sp->s3, U0);					\
  _vec_add(r0, r1, r4, r5, r8, r9);					\
  _vec_sub(r2, r3, r6, r7, r10, r11);					\
  _vec_su3_multiply_double2c(up);					\
  rtmp = vec_ld2(0, (double*) &phase_2);					\
  _vec_cmplx_mul_double2c(r0, r1, r2, r4, r5, r6, rtmp);		\
  _vec_unfuse(r0, r1, r2, r3, r4, r5);					\
  _vec_add_double2(rs0, rs1, rs2, rs3, rs4, rs5, r0, r1, r2, r3, r4, r5);	\
  _vec_sub2(rs6, rs7, rs8, r3, r4, r5);					\
  _vec_add2(rs9, rs10, rs11, r0, r1, r2);

#define _m2add()							\
  _vec_load(r4, r5, sm->s0);						\
  _vec_load16(r6, r7, sm->s1, U0);					\
  _vec_load(r10, r11, sm->s2);						\
  _vec_load16(r8, r9, sm->s3, U0);					\
  _vec_sub(r0, r1, r4, r5, r8, r9);					\
  _vec_add(r2, r3, r6, r7, r10, r11);					\
  _vec_su3_inverse_multiply_double2c(um);				\
  _vec_cmplxcg_mul_double2c(r0, r1, r2, r4, r5, r6, rtmp);		\
  _vec_unfuse(r0, r1, r2, r3, r4, r5);					\
  _vec_add_double2(rs0, rs1, rs2, rs3, rs4, rs5, r0, r1, r2, r3, r4, r5); \
  _vec_add2(rs6, rs7, rs8, r3, r4, r5);					\
  _vec_sub2(rs9, rs10, rs11, r0, r1, r2);

#define _p3add()							\
  _vec_load(r4, r5, sp->s0);						\
  _vec_load16(r6, r7, sp->s1, U0);					\
  _vec_load(r8, r9, sp->s2);						\
  _vec_load16(r10, r11, sp->s3, U0);					\
  _vec_i_mul_add(r0, r1, r4, r5, r8, r9, U0);				\
  _vec_i_mul_sub(r2, r3, r6, r7, r10, r11, U1);				\
  _vec_su3_multiply_double2c(up);					\
  rtmp = vec_ld2(0, (double*) &phase_3);					\
  _vec_cmplx_mul_double2c(r0, r1, r2, r4, r5, r6, rtmp);		\
  _vec_unfuse(r0, r1, r2, r3, r4, r5);					\
  _vec_add_double2(rs0, rs1, rs2, rs3, rs4, rs5, r0, r1, r2, r3, r4, r5); \
  _vec_i_mul_sub2(rs6, rs7, rs8, r0, r1, r2, U0);			\
  _vec_i_mul_add2(rs9, rs10, rs11, r3, r4, r5, U1);

#define _m3add()							\
  _vec_load(r4, r5, sm->s0);						\
  _vec_load16(r6, r7, sm->s1, U0);					\
  _vec_load(r8, r9, sm->s2);						\
  _vec_load16(r10, r11, sm->s3, U0);					\
  _vec_i_mul_sub(r0, r1, r4, r5, r8, r9, U0);				\
  _vec_i_mul_add(r2, r3, r6, r7, r10, r11, U1);				\
  _vec_su3_inverse_multiply_double2c(um);				\
  _vec_cmplxcg_mul_double2c(r0, r1, r2, r4, r5, r6, rtmp);		\
  _vec_unfuse(r0, r1, r2, r3, r4, r5);					\
  _vec_add_double2(rs0, rs1, rs2, rs3, rs4, rs5, r0, r1, r2, r3, r4, r5); \
  _vec_i_mul_add2(rs6, rs7, rs8, r0, r1, r2, U0);			\
  _vec_i_mul_sub2(rs9, rs10, rs11, r3, r4, r5, U1);

#define _store_res()				\
  _vec_store2(rn->s0, rs0, rs1, rs2);		\
  _vec_store2(rn->s1, rs3, rs4, rs5);		\
  _vec_store2(rn->s2, rs6, rs7, rs8);		\
  _vec_store2(rn->s3, rs9, rs10, rs11);

#endif
