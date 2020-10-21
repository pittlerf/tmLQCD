/***********************************************************************
 *
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach,
 * 2014 Mario Schroeck
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more deta_BSMils.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
 *
 *******************************************************************************/

/*******************************************************************************
 
 * Implementation of symmetric derivative version of Frezzotti-Rossi Dirac operator
 * with a scalar field coupling.
 *
 *******************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "global.h"
#include "su3.h"
#include "su3spinor.h"
#include "sse.h"
#include "boundary.h"
#ifdef MPI
# include "xchange/xchange.h"
#endif
#include "update_backward_gauge.h"
#include "operator/D_psi_BSM2b.h"
#include "operator/D_psi_BSM2f.h"
#include "solver/dirac_operator_eigenvectors.h"
#include "buffers/utils.h"
#if defined MPI
#include "buffers/utils_nonblocking.h"
#endif
#include "linalg_eo.h"
#include "fatal_error.h"


/* operation out(x) += F(y)*in(x)
 * F(y) := [ \phi_0(y) + i \gamma_5 \tau^j \phi_j(y) ] * c
 * this operator acts locally on a site x, pass pointers accordingly.
 * out: the resulting bispinor, out += F*in
 * in:  the input bispinor at site x
 * phi: pointer to the four scalars phi0,...,phi3 at site y, y = x or x+-\mu
 * c:    constant double
 *
 * sign = +1 -> Fadd
 * sign = -1 -> Fbaradd
 */

static bispinor *vm0;
static bispinor *vm1;
static bispinor *vm2;
static bispinor *vm3;

static bispinor *vp0;
static bispinor *vp1;
static bispinor *vp2;
static bispinor *vp3;

static bispinor *tempor;

void init_D_psi_BSMf(){

     vm0 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vm1 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vm2 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vm3 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vp0 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vp1 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vp2 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vp3 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));

     tempor=(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));

}
void free_D_psi_BSM3f(){
     free(vm0);
     free(vm1);
     free(vm2);
     free(vm3);
     free(vp0);
     free(vp1);
     free(vp2);
     free(vp3);

     free(tempor);

}

static inline void tm3_add(bispinor * const out, const bispinor * const in, const double sign)
{  
  /*out+=s*i\gamma_5 \tau_3 mu3 *in
   * sign>0 for D+i\gamma_5\tau_3
   * sign<0 for D_dag-i\gamma_5\tau_3
   */
  double s = +1.;
  if(sign < 0) s = -1.;
  
  // out_up += s * i \gamma_5 \mu3 * in_up
  _vector_add_i_mul(out->sp_up.s0,  s*mu03_BSM, in->sp_up.s0);
  _vector_add_i_mul(out->sp_up.s1,  s*mu03_BSM, in->sp_up.s1);
  _vector_add_i_mul(out->sp_up.s2, -s*mu03_BSM, in->sp_up.s2);
  _vector_add_i_mul(out->sp_up.s3, -s*mu03_BSM, in->sp_up.s3);
  
  
  // out_dn +=- s * i \gamma_5 \mu3 * in_dn
  _vector_add_i_mul(out->sp_dn.s0, -s*mu03_BSM, in->sp_dn.s0);
  _vector_add_i_mul(out->sp_dn.s1, -s*mu03_BSM, in->sp_dn.s1);
  _vector_add_i_mul(out->sp_dn.s2,  s*mu03_BSM, in->sp_dn.s2);
  _vector_add_i_mul(out->sp_dn.s3,  s*mu03_BSM, in->sp_dn.s3);
  
}
static inline void tm1_add(bispinor * const out, const bispinor * const in, const double sign)
{  
  /*out+=s*i\gamma_5 \tau_1 mu1 *in
   * sign>0 for D+i\gamma_5\tau_1
   * sign<0 for D_dag-i\gamma_5\tau_1
   */
  double s = +1.;
  if(sign < 0) s = -1.;
  
  // out_up += s * i \gamma_5 \mu1 * in_dn
  _vector_add_i_mul(out->sp_up.s0,  s*mu01_BSM, in->sp_dn.s0);
  _vector_add_i_mul(out->sp_up.s1,  s*mu01_BSM, in->sp_dn.s1);
  _vector_add_i_mul(out->sp_up.s2, -s*mu01_BSM, in->sp_dn.s2);
  _vector_add_i_mul(out->sp_up.s3, -s*mu01_BSM, in->sp_dn.s3);
  
  
  // out_dn += s * i \gamma_5 \mu1 * in_up
  _vector_add_i_mul(out->sp_dn.s0,  s*mu01_BSM, in->sp_up.s0);
  _vector_add_i_mul(out->sp_dn.s1,  s*mu01_BSM, in->sp_up.s1);
  _vector_add_i_mul(out->sp_dn.s2, -s*mu01_BSM, in->sp_up.s2);
  _vector_add_i_mul(out->sp_dn.s3, -s*mu01_BSM, in->sp_up.s3);
  
}

static inline void Fadd(bispinor * const out, const bispinor * const in, const scalar * const phi, const double c, const double sign) {
  static spinor tmp;
  double s = +1.;
  if(sign < 0) s = -1.;

  // flavour 1:
  // tmp_up = \phi_0 * in_up
  _vector_mul(tmp.s0, phi[0], in->sp_up.s0);
  _vector_mul(tmp.s1, phi[0], in->sp_up.s1);
  _vector_mul(tmp.s2, phi[0], in->sp_up.s2);
  _vector_mul(tmp.s3, phi[0], in->sp_up.s3);
  
  // tmp_up += s * i \gamma_5 \phi_1 * in_dn
  _vector_add_i_mul(tmp.s0,  s*phi[1], in->sp_dn.s0);
  _vector_add_i_mul(tmp.s1,  s*phi[1], in->sp_dn.s1);
  _vector_add_i_mul(tmp.s2, -s*phi[1], in->sp_dn.s2);
  _vector_add_i_mul(tmp.s3, -s*phi[1], in->sp_dn.s3);
  
  // tmp_up += s * \gamma_5 \phi_2 * in_dn
  _vector_add_mul(tmp.s0,  s*phi[2], in->sp_dn.s0);
  _vector_add_mul(tmp.s1,  s*phi[2], in->sp_dn.s1);
  _vector_add_mul(tmp.s2, -s*phi[2], in->sp_dn.s2);
  _vector_add_mul(tmp.s3, -s*phi[2], in->sp_dn.s3);
  
  // tmp_up += s * i \gamma_5 \phi_3 * in_up
  _vector_add_i_mul(tmp.s0,  s*phi[3], in->sp_up.s0);
  _vector_add_i_mul(tmp.s1,  s*phi[3], in->sp_up.s1);
  _vector_add_i_mul(tmp.s2, -s*phi[3], in->sp_up.s2);
  _vector_add_i_mul(tmp.s3, -s*phi[3], in->sp_up.s3);
  
  // out_up += c * tmp;
  _vector_add_mul(out->sp_up.s0,c,tmp.s0);
  _vector_add_mul(out->sp_up.s1,c,tmp.s1);
  _vector_add_mul(out->sp_up.s2,c,tmp.s2);
  _vector_add_mul(out->sp_up.s3,c,tmp.s3);
  
  
  // flavour 2:
  // tmp_dn = \phi_0 * in_dn
  _vector_mul(tmp.s0, phi[0], in->sp_dn.s0);
  _vector_mul(tmp.s1, phi[0], in->sp_dn.s1);
  _vector_mul(tmp.s2, phi[0], in->sp_dn.s2);
  _vector_mul(tmp.s3, phi[0], in->sp_dn.s3);
  
  // tmp_dn += s * i \gamma_5 \phi_1 * in_up
  _vector_add_i_mul(tmp.s0,  s*phi[1], in->sp_up.s0);
  _vector_add_i_mul(tmp.s1,  s*phi[1], in->sp_up.s1);
  _vector_add_i_mul(tmp.s2, -s*phi[1], in->sp_up.s2);
  _vector_add_i_mul(tmp.s3, -s*phi[1], in->sp_up.s3);
  
  // tmp_dn -= s * \gamma_5 \phi_2 * in_up
  _vector_add_mul(tmp.s0, -s*phi[2], in->sp_up.s0);
  _vector_add_mul(tmp.s1, -s*phi[2], in->sp_up.s1);
  _vector_add_mul(tmp.s2,  s*phi[2], in->sp_up.s2);
  _vector_add_mul(tmp.s3,  s*phi[2], in->sp_up.s3);
  
  // tmp_dn -= s * i \gamma_5 \phi_3 * in_dn
  _vector_add_i_mul(tmp.s0, -s*phi[3], in->sp_dn.s0);
  _vector_add_i_mul(tmp.s1, -s*phi[3], in->sp_dn.s1);
  _vector_add_i_mul(tmp.s2,  s*phi[3], in->sp_dn.s2);
  _vector_add_i_mul(tmp.s3,  s*phi[3], in->sp_dn.s3);
  
  // out_dn += c * tmp;
  _vector_add_mul(out->sp_dn.s0,c,tmp.s0);
  _vector_add_mul(out->sp_dn.s1,c,tmp.s1);
  _vector_add_mul(out->sp_dn.s2,c,tmp.s2);
  _vector_add_mul(out->sp_dn.s3,c,tmp.s3);
}

static inline void bispinor_times_phase_times_u(bispinor * const us, const _Complex double phase,
            su3 const * restrict const u, bispinor const * const s)
{
  static su3_vector chi;
  _su3_multiply(chi, (*u), s->sp_up.s0);
  _complex_times_vector(us->sp_up.s0, phase, chi);

  _su3_multiply(chi, (*u), s->sp_up.s1);
  _complex_times_vector(us->sp_up.s1, phase, chi);

  _su3_multiply(chi, (*u), s->sp_up.s2);
  _complex_times_vector(us->sp_up.s2, phase, chi);

  _su3_multiply(chi, (*u), s->sp_up.s3);
  _complex_times_vector(us->sp_up.s3, phase, chi);

  _su3_multiply(chi, (*u), s->sp_dn.s0);
  _complex_times_vector(us->sp_dn.s0, phase, chi);

  _su3_multiply(chi, (*u), s->sp_dn.s1);
  _complex_times_vector(us->sp_dn.s1, phase, chi);

  _su3_multiply(chi, (*u), s->sp_dn.s2);
  _complex_times_vector(us->sp_dn.s2, phase, chi);

  _su3_multiply(chi, (*u), s->sp_dn.s3);
  _complex_times_vector(us->sp_dn.s3, phase, chi);

  return;
}

static inline void bispinor_times_phase_times_inverse_u(bispinor * const us, const _Complex double phase,
              su3 const * restrict const u, bispinor const * const s)
{
  static su3_vector chi;
  _su3_inverse_multiply(chi, (*u), s->sp_up.s0);
  _complexcjg_times_vector(us->sp_up.s0, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_up.s1);
  _complexcjg_times_vector(us->sp_up.s1, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_up.s2);
  _complexcjg_times_vector(us->sp_up.s2, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_up.s3);
  _complexcjg_times_vector(us->sp_up.s3, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s0);
  _complexcjg_times_vector(us->sp_dn.s0, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s1);
  _complexcjg_times_vector(us->sp_dn.s1, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s2);
  _complexcjg_times_vector(us->sp_dn.s2, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s3);
  _complexcjg_times_vector(us->sp_dn.s3, phase, chi);

  return;
}

static inline void p0add(bispinor * restrict const dest, bispinor * const restrict const source,
                         su3 const * restrict const u, const hopdirection dir, const _Complex double phase,
                         const double phaseF, const scalar * const phi, const scalar * const phip, int sign) 
{
  static bispinor chitmp;
  // chitmp = phase*u*s
  if( dir == HOP_DN )
    bispinor_times_phase_times_inverse_u(&chitmp, phase, u, source);
  else
    bispinor_times_phase_times_u        (&chitmp, phase, u, source);

  // dest += \gamma_0*chitmp
  _vector_add_assign(dest->sp_up.s0, chitmp.sp_up.s2);
  _vector_add_assign(dest->sp_up.s1, chitmp.sp_up.s3);
  _vector_add_assign(dest->sp_up.s2, chitmp.sp_up.s0);
  _vector_add_assign(dest->sp_up.s3, chitmp.sp_up.s1);

  _vector_add_assign(dest->sp_dn.s0, chitmp.sp_dn.s2);
  _vector_add_assign(dest->sp_dn.s1, chitmp.sp_dn.s3);
  _vector_add_assign(dest->sp_dn.s2, chitmp.sp_dn.s0);
  _vector_add_assign(dest->sp_dn.s3, chitmp.sp_dn.s1);

  // dest -= F(phi )*chitmp
  Fadd(dest, &chitmp, phi,  phaseF, sign);
  // dest -= F(phip)*chitmp
  Fadd(dest, &chitmp, phip, phaseF, sign);

}

static inline void p1add(bispinor * restrict const dest, bispinor * const restrict const source,
                         su3 const * restrict const u, const hopdirection dir, const _Complex double phase,
                         const double phaseF, const scalar * const phi, const scalar * const phip, int sign)
{
  static bispinor chitmp;
  // chitmp = phase*u*s
  if( dir == HOP_DN )
    bispinor_times_phase_times_inverse_u(&chitmp, phase, u, source);
  else
    bispinor_times_phase_times_u        (&chitmp, phase, u, source);


  // dest += \gamma_1*chitmp
  _vector_i_add_assign(dest->sp_up.s0, chitmp.sp_up.s3);
  _vector_i_add_assign(dest->sp_up.s1, chitmp.sp_up.s2);
  _vector_i_sub_assign(dest->sp_up.s2, chitmp.sp_up.s1);
  _vector_i_sub_assign(dest->sp_up.s3, chitmp.sp_up.s0);

  _vector_i_add_assign(dest->sp_dn.s0, chitmp.sp_dn.s3);
  _vector_i_add_assign(dest->sp_dn.s1, chitmp.sp_dn.s2);
  _vector_i_sub_assign(dest->sp_dn.s2, chitmp.sp_dn.s1);
  _vector_i_sub_assign(dest->sp_dn.s3, chitmp.sp_dn.s0);

  // dest -= F(phi )*chitmp
  Fadd(dest, &chitmp, phi,  phaseF, sign);
  // dest -= F(phip)*chitmp
  Fadd(dest, &chitmp, phip, phaseF, sign);

  return;

}
static inline void p2add(bispinor * restrict const dest, bispinor * const restrict const source,
                         su3 const * restrict const u, const hopdirection dir, const _Complex double phase,
                         const double phaseF, const scalar * const phi, const scalar * const phip, int sign)
{
  static bispinor chitmp;
  // chitmp = phase*u*s
  if( dir == HOP_DN )
    bispinor_times_phase_times_inverse_u(&chitmp, phase, u, source);
  else
    bispinor_times_phase_times_u        (&chitmp, phase, u, source);

  // dest += \gamma_2*chitmp
  _vector_add_assign(dest->sp_up.s0, chitmp.sp_up.s3);
  _vector_sub_assign(dest->sp_up.s1, chitmp.sp_up.s2);
  _vector_sub_assign(dest->sp_up.s2, chitmp.sp_up.s1);
  _vector_add_assign(dest->sp_up.s3, chitmp.sp_up.s0);

  _vector_add_assign(dest->sp_dn.s0, chitmp.sp_dn.s3);
  _vector_sub_assign(dest->sp_dn.s1, chitmp.sp_dn.s2);
  _vector_sub_assign(dest->sp_dn.s2, chitmp.sp_dn.s1);
  _vector_add_assign(dest->sp_dn.s3, chitmp.sp_dn.s0);

  // dest -= F(phi )*chitmp
  Fadd(dest, &chitmp, phi,  phaseF, sign);
  // dest -= F(phip)*chitmp
  Fadd(dest, &chitmp, phip, phaseF, sign);

  return;
}

static inline void p3add(bispinor * restrict const dest, bispinor * const restrict const source,
                         su3 const * restrict const u, const hopdirection dir, const _Complex double phase,
                         const double phaseF, const scalar * const phi, const scalar * const phip, int sign)
{
  static bispinor chitmp;
  // chitmp = phase*u*s
  if( dir == HOP_DN )
    bispinor_times_phase_times_inverse_u(&chitmp, phase, u, source);
  else
    bispinor_times_phase_times_u        (&chitmp, phase, u, source);

  // dest += \gamma_3*chitmp
  _vector_i_add_assign(dest->sp_up.s0, chitmp.sp_up.s2);
  _vector_i_sub_assign(dest->sp_up.s1, chitmp.sp_up.s3);
  _vector_i_sub_assign(dest->sp_up.s2, chitmp.sp_up.s0);
  _vector_i_add_assign(dest->sp_up.s3, chitmp.sp_up.s1);

  _vector_i_add_assign(dest->sp_dn.s0, chitmp.sp_dn.s2);
  _vector_i_sub_assign(dest->sp_dn.s1, chitmp.sp_dn.s3);
  _vector_i_sub_assign(dest->sp_dn.s2, chitmp.sp_dn.s0);
  _vector_i_add_assign(dest->sp_dn.s3, chitmp.sp_dn.s1);

  // dest -= F(phi )*chitmp
  Fadd(dest, &chitmp, phi,  phaseF, sign);
  // dest -= F(phip)*chitmp
  Fadd(dest, &chitmp, phip, phaseF, sign);
    
  return;

}


/* D_psi_BSM3f acts on bispinor fields */
void D_psi_BSM3f(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM3f (D_psi_BSM3f.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
    update_backward_gauge(g_gauge_field);
  }
#endif
  int ix;
  su3 * restrict upm;
  bispinor * restrict rr;
  bispinor * restrict rr0;
  bispinor * restrict rr1;
  bispinor * restrict rr2;
  bispinor * restrict rr3;
  bispinor * restrict rr4;
  bispinor * restrict rr5;
  bispinor * restrict rr6;
  bispinor * restrict rr7;

  bispinor const * restrict s;         // Q(x)
  bispinor const * restrict spm;
  scalar phi[4];                       // phi_i(x)
  scalar phip[4][4];                   // phi_i(x+mu) = phip[mu][i]
  scalar phim[4][4];                   // phi_i(x-mu) = phim[mu][i]

#if defined MPI
  MPI_Status  statuses[8];
  MPI_Request *request;
  request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

//start gathering forward
  int count=0;
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), TUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), XUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), YUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), ZUP, request, &count);
#endif
//  computing backward
  for (ix=0;ix<VOLUME;ix++)
  {

//  prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

    for ( int mu=0; mu<4; mu++ )
    {
      phim[mu][0] = g_scalar_field[0][g_idn[ix][mu]];
      phim[mu][1] = g_scalar_field[1][g_idn[ix][mu]];
      phim[mu][2] = g_scalar_field[2][g_idn[ix][mu]];
      phim[mu][3] = g_scalar_field[3][g_idn[ix][mu]];
    }
// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr0 = vm0 + ix;
    rr1 = vm1 + ix;
    rr2 = vm2 + ix;
    rr3 = vm3 + ix;

    _bispinor_null(*rr0);
    _bispinor_null(*rr1);
    _bispinor_null(*rr2);
    _bispinor_null(*rr3);

//source buffer
    s  = (bispinor *) Q + ix;

//  Direction 0 -
    upm = &g_gauge_field[ix][TUP];
    p0add(rr0, s, upm, HOP_DN, 0.5*phase_0, 0.5*rho_BSM, phi, phim[0], +1);

//  Direction 1 -
    upm = &g_gauge_field[ix][XUP];
    p1add(rr1, s, upm, HOP_DN, 0.5*phase_1, 0.5*rho_BSM, phi, phim[1], +1);

//  Direction 2 -
    upm = &g_gauge_field[ix][YUP];
    p2add(rr2, s, upm, HOP_DN, 0.5*phase_2, 0.5*rho_BSM, phi, phim[2], +1);

//  Direction 3 -
    upm = &g_gauge_field[ix][ZUP];
    p3add(rr3, s, upm, HOP_DN, 0.5*phase_3, 0.5*rho_BSM, phi, phim[3], +1 );
  }
 
#if defined MPI 
  MPI_Waitall( count, request, statuses);

//gathering backward
  count=0;
  generic_exchange_direction_nonblocking(vm0, sizeof(bispinor), TDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm1, sizeof(bispinor), XDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm2, sizeof(bispinor), YDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm3, sizeof(bispinor), ZDOWN, request, &count);
#endif
//computing forward

  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing forward connections : U_mu(x) psi(x+mu)
    rr0 = vp0 + ix;
    rr1 = vp1 + ix;
    rr2 = vp2 + ix;
    rr3 = vp3 + ix;

// intermedieate buffer for multiplication with gamma_mu 
    _bispinor_null(*rr0);
    _bispinor_null(*rr1);
    _bispinor_null(*rr2);
    _bispinor_null(*rr3);

//  prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

    for ( int mu=0; mu<4; mu++ )
    {
      phip[mu][0] = g_scalar_field[0][g_iup[ix][mu]];
      phip[mu][1] = g_scalar_field[1][g_iup[ix][mu]];
      phip[mu][2] = g_scalar_field[2][g_iup[ix][mu]];
      phip[mu][3] = g_scalar_field[3][g_iup[ix][mu]];
    }

//  Direction 0 +
    upm = &g_gauge_field[ix][TUP];
    spm = (bispinor *) Q + g_iup[ix][TUP];
    p0add(rr0, spm, upm, HOP_UP, 0.5*phase_0, -0.5*rho_BSM, phi, phip[0],+1);

//  Direction 1 +
    upm = &g_gauge_field[ix][XUP];
    spm = (bispinor *) Q + g_iup[ix][XUP];
    p1add(rr1, spm, upm, HOP_UP, 0.5*phase_1, -0.5*rho_BSM, phi, phip[1],+1);

//  Direction 2 +
    upm = &g_gauge_field[ix][YUP];
    spm = (bispinor *) Q + g_iup[ix][YUP];
    p2add(rr2, spm, upm, HOP_UP, 0.5*phase_2, -0.5*rho_BSM, phi, phip[2],+1);

//  Direction 3 +
    upm = &g_gauge_field[ix][ZUP];
    spm = (bispinor *) Q + g_iup[ix][ZUP];
    p3add(rr3, spm, upm, HOP_UP, 0.5*phase_3, -0.5*rho_BSM, phi, phip[3],+1);

  }
#if defined MPI
  MPI_Waitall( count, request, statuses);
#endif
// join
  for (ix=0; ix<VOLUME; ++ix){

// destination buffer
    rr = (bispinor *) P + ix;
// source buffer
     s = (bispinor *) Q + ix;
    _bispinor_null(*rr);

// intermediate buffers for storing backward connections : U_mu( x )^dagg psi(x+mu)
    rr0 = vp0 + ix;
    rr1 = vp1 + ix;
    rr2 = vp2 + ix;
    rr3 = vp3 + ix;

// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = vm0 + g_idn[ix][TUP];
    rr5 = vm1 + g_idn[ix][XUP];
    rr6 = vm2 + g_idn[ix][YUP];
    rr7 = vm3 + g_idn[ix][ZUP];

    _bispinor_add_mul( *rr, +1.0, *rr0 );
    _bispinor_add_mul( *rr, -1.0, *rr4 );

    _bispinor_add_mul( *rr, +1.0, *rr1 );
    _bispinor_add_mul( *rr, -1.0, *rr5 );

    _bispinor_add_mul( *rr, +1.0, *rr2 );
    _bispinor_add_mul( *rr, -1.0, *rr6 );

    _bispinor_add_mul( *rr, +1.0, *rr3 );
    _bispinor_add_mul( *rr, -1.0, *rr7 );
  
    // prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

    for( int mu=0; mu<4; mu++ )
    {
      phip[mu][0] = g_scalar_field[0][g_iup[ix][mu]];
      phip[mu][1] = g_scalar_field[1][g_iup[ix][mu]];
      phip[mu][2] = g_scalar_field[2][g_iup[ix][mu]];
      phip[mu][3] = g_scalar_field[3][g_iup[ix][mu]];

      phim[mu][0] = g_scalar_field[0][g_idn[ix][mu]];
      phim[mu][1] = g_scalar_field[1][g_idn[ix][mu]];
      phim[mu][2] = g_scalar_field[2][g_idn[ix][mu]];
      phim[mu][3] = g_scalar_field[3][g_idn[ix][mu]];

    }

    // the local part (not local in phi)


    // tmpr += (\eta_BSM+2*\rho_BSM) * F(x)*Q(x)
    Fadd(rr, s, phi, eta_BSM+2.0*rho_BSM, +1.);

    // tmpr += \sum_\mu (\rho_BSM/4) * F(x+-\mu)*Q
    for( int mu=0; mu<4; mu++ ) {
      Fadd(rr, s, phip[mu], 0.25*rho_BSM, +1.);
      Fadd(rr, s, phim[mu], 0.25*rho_BSM, +1.);
    }

  }

#if defined MPI
  free(request);
#endif

}

/* D_psi_BSM3f acts on bispinor fields */
void D_psi_dagger_BSM3f(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM3f (D_psi_BSM3f.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
    update_backward_gauge(g_gauge_field);
  }
#endif
  int ix;
  su3 * restrict upm;
  bispinor * restrict rr;
  bispinor * restrict rr0;
  bispinor * restrict rr1;
  bispinor * restrict rr2;
  bispinor * restrict rr3;
  bispinor * restrict rr4;
  bispinor * restrict rr5;
  bispinor * restrict rr6;
  bispinor * restrict rr7;

  bispinor const * restrict s;         // Q(x)
  bispinor const * restrict spm;
  scalar phi[4];                       // phi_i(x)
  scalar phip[4][4];                   // phi_i(x+mu) = phip[mu][i]
  scalar phim[4][4];                   // phi_i(x-mu) = phim[mu][i]

#if defined MPI
  MPI_Status  statuses[8];
  MPI_Request *request;
  request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

//start gathering forward
  int count=0;
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), TUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), XUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), YUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), ZUP, request, &count);
#endif
//  computing backward
  for (ix=0;ix<VOLUME;ix++)
  {

//  prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

    for ( int mu=0; mu<4; mu++ )
    {
      phim[mu][0] = g_scalar_field[0][g_idn[ix][mu]];
      phim[mu][1] = g_scalar_field[1][g_idn[ix][mu]];
      phim[mu][2] = g_scalar_field[2][g_idn[ix][mu]];
      phim[mu][3] = g_scalar_field[3][g_idn[ix][mu]];
    }
// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr0 = vm0 + ix;
    rr1 = vm1 + ix;
    rr2 = vm2 + ix;
    rr3 = vm3 + ix;

    _bispinor_null(*rr0);
    _bispinor_null(*rr1);
    _bispinor_null(*rr2);
    _bispinor_null(*rr3);

//source buffer
    s  = (bispinor *) Q + ix;

//  Direction 0 -
    upm = &g_gauge_field[ix][TUP];
    p0add(rr0, s, upm, HOP_DN, 0.5*phase_0, 0.5*rho_BSM, phi, phim[0], -1);

//  Direction 1 -
    upm = &g_gauge_field[ix][XUP];
    p1add(rr1, s, upm, HOP_DN, 0.5*phase_1, 0.5*rho_BSM, phi, phim[1], -1);

//  Direction 2 -
    upm = &g_gauge_field[ix][YUP];
    p2add(rr2, s, upm, HOP_DN, 0.5*phase_2, 0.5*rho_BSM, phi, phim[2], -1);

//  Direction 3 -
    upm = &g_gauge_field[ix][ZUP];
    p3add(rr3, s, upm, HOP_DN, 0.5*phase_3, 0.5*rho_BSM, phi, phim[3], -1);
  }

#if defined MPI 
  MPI_Waitall( count, request, statuses);

//gathering backward
  count=0;
  generic_exchange_direction_nonblocking(vm0, sizeof(bispinor), TDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm1, sizeof(bispinor), XDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm2, sizeof(bispinor), YDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm3, sizeof(bispinor), ZDOWN, request, &count);
#endif
//computing forward

  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing forward connections : U_mu(x) psi(x+mu)
    rr0 = vp0 + ix;
    rr1 = vp1 + ix;
    rr2 = vp2 + ix;
    rr3 = vp3 + ix;

// intermedieate buffer for multiplication with gamma_mu 
    _bispinor_null(*rr0);
    _bispinor_null(*rr1);
    _bispinor_null(*rr2);
    _bispinor_null(*rr3);

//  prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

    for ( int mu=0; mu<4; mu++ )
    {
      phip[mu][0] = g_scalar_field[0][g_iup[ix][mu]];
      phip[mu][1] = g_scalar_field[1][g_iup[ix][mu]];
      phip[mu][2] = g_scalar_field[2][g_iup[ix][mu]];
      phip[mu][3] = g_scalar_field[3][g_iup[ix][mu]];
    }

//  Direction 0 +
    upm = &g_gauge_field[ix][TUP];
    spm = (bispinor *) Q + g_iup[ix][TUP];
    p0add(rr0, spm, upm, HOP_UP, 0.5*phase_0, -0.5*rho_BSM, phi, phip[0], -1);

//  Direction 1 +
    upm = &g_gauge_field[ix][XUP];
    spm = (bispinor *) Q + g_iup[ix][XUP];
    p1add(rr1, spm, upm, HOP_UP, 0.5*phase_1, -0.5*rho_BSM, phi, phip[1], -1);

//  Direction 2 +
    upm = &g_gauge_field[ix][YUP];
    spm = (bispinor *) Q + g_iup[ix][YUP];
    p2add(rr2, spm, upm, HOP_UP, 0.5*phase_2, -0.5*rho_BSM, phi, phip[2], -1);

//  Direction 3 +
    upm = &g_gauge_field[ix][ZUP];
    spm = (bispinor *) Q + g_iup[ix][ZUP];
    p3add(rr3, spm, upm, HOP_UP, 0.5*phase_3, -0.5*rho_BSM, phi, phip[3], -1);

  }
#if defined MPI
  MPI_Waitall( count, request, statuses);
#endif
// join
  for (ix=0; ix<VOLUME; ++ix){

// destination buffer
    rr = (bispinor *) P + ix;
// source buffer
     s = (bispinor *) Q + ix;
    _bispinor_null(*rr);

// intermediate buffers for storing backward connections : U_mu( x )^dagg psi(x+mu)
    rr0 = vp0 + ix;
    rr1 = vp1 + ix;
    rr2 = vp2 + ix;
    rr3 = vp3 + ix;

// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = vm0 + g_idn[ix][TUP];
    rr5 = vm1 + g_idn[ix][XUP];
    rr6 = vm2 + g_idn[ix][YUP];
    rr7 = vm3 + g_idn[ix][ZUP];

    _bispinor_add_mul( *rr, +1.0, *rr0 );
    _bispinor_add_mul( *rr, -1.0, *rr4 );

    _bispinor_add_mul( *rr, +1.0, *rr1 );
    _bispinor_add_mul( *rr, -1.0, *rr5 );

    _bispinor_add_mul( *rr, +1.0, *rr2 );
    _bispinor_add_mul( *rr, -1.0, *rr6 );

    _bispinor_add_mul( *rr, +1.0, *rr3 );
    _bispinor_add_mul( *rr, -1.0, *rr7 );

    // prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

    for( int mu=0; mu<4; mu++ )
    {
      phip[mu][0] = g_scalar_field[0][g_iup[ix][mu]];
      phip[mu][1] = g_scalar_field[1][g_iup[ix][mu]];
      phip[mu][2] = g_scalar_field[2][g_iup[ix][mu]];
      phip[mu][3] = g_scalar_field[3][g_iup[ix][mu]];

      phim[mu][0] = g_scalar_field[0][g_idn[ix][mu]];
      phim[mu][1] = g_scalar_field[1][g_idn[ix][mu]];
      phim[mu][2] = g_scalar_field[2][g_idn[ix][mu]];
      phim[mu][3] = g_scalar_field[3][g_idn[ix][mu]];

    }

    // the local part (not local in phi)


    // tmpr += (\eta_BSM+2*\rho_BSM) * F(x)*Q(x)
    Fadd(rr, s, phi, eta_BSM+2.0*rho_BSM, -1.);

    // tmpr += \sum_\mu (\rho_BSM/4) * F(x+-\mu)*Q
    for( int mu=0; mu<4; mu++ ) {
      Fadd(rr, s, phip[mu], 0.25*rho_BSM, -1.);
      Fadd(rr, s, phim[mu], 0.25*rho_BSM, -1.);
    }

  }

#if defined MPI
  free(request);
#endif

}


/* Q2_psi_BSM2f acts on bispinor fields */
void Q2_psi_BSM3f(bispinor * const P, bispinor * const Q){

  /* TODO: the use of [3] has to be changed to avoid future conflicts */
  D_psi_dagger_BSM3f(tempor , Q);
  D_psi_BSM3f(P, tempor);
  // only use these cycles if the m0_BSM parameter is really nonzero...
  if( fabs(m0_BSM) > 1.e-10 ){
    /* Q and P are spinor, not bispinor ==> made a cast */
    assign_add_mul_r((spinor*)P, (spinor*)Q, m0_BSM, 2*VOLUME);
  }

}
