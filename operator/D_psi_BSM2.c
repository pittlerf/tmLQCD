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
 *
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
#include "sse.h"
#include "boundary.h"
#ifdef MPI
# include "xchange/xchange.h"
#endif
#include "update_backward_gauge.h"
#include "block.h"
#include "operator/D_psi_BSM2.h"
#include "solver/dirac_operator_eigenvectors.h"
#include "buffers/utils.h"
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

static inline void Fadd(bispinor * const out, const bispinor * const in, const scalar * const phi, const double c, const double sign) {
#ifdef OMP
#define static
#endif
  static spinor tmp;
#ifdef OMP
#undef static
#endif
  
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
#ifdef OMP
#define static
#endif
  static su3_vector chi;
#ifdef OMP
#undef static
#endif

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
#ifdef OMP
#define static
#endif
  static su3_vector chi;
#ifdef OMP
#undef static
#endif

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

static inline void Fadd2hop( bispinor * const out, const bispinor * const in, const scalar * const phipm, 
                             const double c, const double sign, const su3 * restrict const u1, const su3 * restrict const u2,
                             const _Complex double phase, const hopdirection dir )
{
#ifdef OMP
#define static
#endif
  static bispinor uus;
#ifdef OMP
#undef static
#endif

  switch( dir ){
    case HOP_UP:
      bispinor_times_phase_times_u(&uus, phase, u2, in);
      bispinor_times_phase_times_u(&uus, phase, u1, &uus); 
      break;
    case HOP_DN:
      bispinor_times_phase_times_inverse_u(&uus, phase, u2, in);
      bispinor_times_phase_times_inverse_u(&uus, phase, u1, &uus); 
      break;
    default:
      fatal_error("Invalid value for 'hopdirection'","Fadd2hop");
      break;
  }

  Fadd(out, &uus, phipm, c, sign); 
}

static inline void p0add(bispinor * restrict const tmpr , bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase) {

#ifdef OMP
#define static
#endif
  static bispinor us;
#ifdef OMP
#undef static
#endif

  // us = phase*u*s
  if( inv )
    bispinor_times_phase_times_inverse_u(&us, phase, u, s);
  else
    bispinor_times_phase_times_u(&us, phase, u, s);

  // tmpr += \gamma_0*us
  _vector_add_assign(tmpr->sp_up.s0, us.sp_up.s2);
  _vector_add_assign(tmpr->sp_up.s1, us.sp_up.s3);
  _vector_add_assign(tmpr->sp_up.s2, us.sp_up.s0);
  _vector_add_assign(tmpr->sp_up.s3, us.sp_up.s1);

  _vector_add_assign(tmpr->sp_dn.s0, us.sp_dn.s2);
  _vector_add_assign(tmpr->sp_dn.s1, us.sp_dn.s3);
  _vector_add_assign(tmpr->sp_dn.s2, us.sp_dn.s0);
  _vector_add_assign(tmpr->sp_dn.s3, us.sp_dn.s1);

  return;
}

static inline void p1add(bispinor * restrict const tmpr, bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static bispinor us;
#ifdef OMP
#undef static
#endif

  // us = phase*u*s
  if( inv )
    bispinor_times_phase_times_inverse_u(&us, phase, u, s);
  else
    bispinor_times_phase_times_u(&us, phase, u, s);

  // tmpr += \gamma_1*us
  _vector_i_add_assign(tmpr->sp_up.s0, us.sp_up.s3);
  _vector_i_add_assign(tmpr->sp_up.s1, us.sp_up.s2);
  _vector_i_sub_assign(tmpr->sp_up.s2, us.sp_up.s1);
  _vector_i_sub_assign(tmpr->sp_up.s3, us.sp_up.s0);

  _vector_i_add_assign(tmpr->sp_dn.s0, us.sp_dn.s3);
  _vector_i_add_assign(tmpr->sp_dn.s1, us.sp_dn.s2);
  _vector_i_sub_assign(tmpr->sp_dn.s2, us.sp_dn.s1);
  _vector_i_sub_assign(tmpr->sp_dn.s3, us.sp_dn.s0);

  return;
}

static inline void p2add(bispinor * restrict const tmpr, bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static bispinor us;
#ifdef OMP
#undef static
#endif

  // us = phase*u*s
  if( inv )
    bispinor_times_phase_times_inverse_u(&us, phase, u, s);
  else
    bispinor_times_phase_times_u(&us, phase, u, s);

  // tmpr += \gamma_2*us
  _vector_add_assign(tmpr->sp_up.s0, us.sp_up.s3);
  _vector_sub_assign(tmpr->sp_up.s1, us.sp_up.s2);
  _vector_sub_assign(tmpr->sp_up.s2, us.sp_up.s1);
  _vector_add_assign(tmpr->sp_up.s3, us.sp_up.s0);

  _vector_add_assign(tmpr->sp_dn.s0, us.sp_dn.s3);
  _vector_sub_assign(tmpr->sp_dn.s1, us.sp_dn.s2);
  _vector_sub_assign(tmpr->sp_dn.s2, us.sp_dn.s1);
  _vector_add_assign(tmpr->sp_dn.s3, us.sp_dn.s0);

  return;
}

static inline void p3add(bispinor * restrict const tmpr, bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static bispinor us;
#ifdef OMP
#undef static
#endif

  // us = phase*u*s
  if( inv )
    bispinor_times_phase_times_inverse_u(&us, phase, u, s);
  else
    bispinor_times_phase_times_u(&us, phase, u, s);

  // tmpr += \gamma_3*us
  _vector_i_add_assign(tmpr->sp_up.s0, us.sp_up.s2);
  _vector_i_sub_assign(tmpr->sp_up.s1, us.sp_up.s3);
  _vector_i_sub_assign(tmpr->sp_up.s2, us.sp_up.s0);
  _vector_i_add_assign(tmpr->sp_up.s3, us.sp_up.s1);

  _vector_i_add_assign(tmpr->sp_dn.s0, us.sp_dn.s2);
  _vector_i_sub_assign(tmpr->sp_dn.s1, us.sp_dn.s3);
  _vector_i_sub_assign(tmpr->sp_dn.s2, us.sp_dn.s0);
  _vector_i_add_assign(tmpr->sp_dn.s3, us.sp_dn.s1);

  return;
}


/* D_psi_BSM acts on bispinor fields */
void D_psi_BSM2(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM2 (D_psi_BSM2.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
    update_backward_gauge(g_gauge_field);
  }
#endif
#ifdef MPI
  generic_exchange(Q, sizeof(bispinor));
#endif
        
#ifdef OMP
#pragma omp parallel
  {
#endif

  int ix;
  int ixpm, ixpm2;
  su3 * restrict upm;
  su3 * restrict upm2;                 // U(x)_{+-\mu}, U(x)_{+-2\mu}
  bispinor * restrict rr;              // P(x)
  bispinor const * restrict s;         // Q(x)
  bispinor const * restrict spm;
  bispinor const * restrict spm2;      // Q(x+-\mu), Q(x+-2\mu)
  scalar phi[4];                       // phi_i(x)
  scalar phip[4][4];                   // phi_i(x+mu) = phip[mu][i]
  scalar phim[4][4];                   // phi_i(x-mu) = phim[mu][i]

    /************************ loop over all lattice sites *************************/

#ifdef OMP
#pragma omp for
#endif
  for (ix=0;ix<VOLUME;ix++)
    {
    rr = (bispinor *) P + ix;
    s  = (bispinor *) Q + ix;

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

  // tmpr = m0_BSM*Q(x)
  //_vector_mul(rr->sp_up.s0, m0_BSM, s->sp_up.s0);
  //_vector_mul(rr->sp_up.s1, m0_BSM, s->sp_up.s1);
  //_vector_mul(rr->sp_up.s2, m0_BSM, s->sp_up.s2);
  //_vector_mul(rr->sp_up.s3, m0_BSM, s->sp_up.s3);
  //_vector_mul(rr->sp_dn.s0, m0_BSM, s->sp_dn.s0);
  //_vector_mul(rr->sp_dn.s1, m0_BSM, s->sp_dn.s1);
  //_vector_mul(rr->sp_dn.s2, m0_BSM, s->sp_dn.s2);
  //_vector_mul(rr->sp_dn.s3, m0_BSM, s->sp_dn.s3);
  // no longer needed now...
    _spinor_null(rr->sp_up);
    _spinor_null(rr->sp_dn);


    // tmpr += (\eta_BSM) * F(x)*Q(x)
    Fadd(rr, s, phi, eta_BSM, +1.);

    // tmpr += \sum_\mu (\rho_BSM/8) * F(x+-\mu)*Q
    for( int mu=0; mu<4; mu++ ) {
      Fadd(rr, s, phip[mu], 0.125*rho_BSM, +1.);
      Fadd(rr, s, phim[mu], 0.125*rho_BSM, +1.);
    }


    // the hopping part:
    // tmpr += +-1/2 \sum_\mu (\gamma_\mu - \rho_BSM/4*F(x+-\mu)*U_{+-\mu}(x)U_{x+-2\mu)*Q(x+-2\mu)
    /******************************* direction +0 *********************************/
    ixpm=g_iup[ix][0]; ixpm2=g_iup[ixpm][0];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ix][0]; upm2=&g_gauge_field[ixpm][0];
    Fadd2hop( rr, spm2, phip[0], -0.125*rho_BSM, +1.0, upm, upm2, phase_0, HOP_UP );
    p0add(rr, spm, upm, 0, 0.5*phase_0);

    /******************************* direction -0 *********************************/
    ixpm=g_idn[ix][0]; ixpm2=g_idn[ixpm][0];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ixpm][0]; upm2=&g_gauge_field[ixpm2][0];
    Fadd2hop(rr, spm2, phim[0], -0.125*rho_BSM, +1.0, upm, upm2, phase_0, HOP_DN );
    p0add(rr, spm, upm, 0, -0.5*phase_0);
    
    /******************************* direction +1 *********************************/
    ixpm=g_iup[ix][1]; ixpm2=g_iup[ixpm][1];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ix][1]; upm2=&g_gauge_field[ixpm][1];
    Fadd2hop( rr, spm2, phip[1], -0.125*rho_BSM, +1.0, upm, upm2, phase_1, HOP_UP );
    p1add(rr, spm, upm, 0, 0.5*phase_1);

    /******************************* direction -1 *********************************/
    ixpm=g_idn[ix][1]; ixpm2=g_idn[ixpm][1];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ixpm][1]; upm2=&g_gauge_field[ixpm2][1];
    Fadd2hop(rr, spm2, phim[1], -0.125*rho_BSM, +1.0, upm, upm2, phase_1, HOP_DN );
    p1add(rr, spm, upm, 0, -0.5*phase_1);
    
    /******************************* direction +2 *********************************/
    ixpm=g_iup[ix][2]; ixpm2=g_iup[ixpm][2];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ix][2]; upm2=&g_gauge_field[ixpm][2];
    Fadd2hop( rr, spm2, phip[2], -0.125*rho_BSM, +1.0, upm, upm2, phase_2, HOP_UP );
    p2add(rr, spm, upm, 0, 0.5*phase_2);

    /******************************* direction -2 *********************************/
    ixpm=g_idn[ix][2]; ixpm2=g_idn[ixpm][2];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ixpm][2]; upm2=&g_gauge_field[ixpm2][2];
    Fadd2hop(rr, spm2, phim[2], -0.125*rho_BSM, +1.0, upm, upm2, phase_2, HOP_DN );
    p2add(rr, spm, upm, 0, -0.5*phase_2);
    
    /******************************* direction +3 *********************************/
    ixpm=g_iup[ix][3]; ixpm2=g_iup[ixpm][3];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ix][3]; upm2=&g_gauge_field[ixpm][3];
    Fadd2hop( rr, spm2, phip[3], -0.125*rho_BSM, +1.0, upm, upm2, phase_3, HOP_UP );
    p3add(rr, spm, upm, 0, 0.5*phase_3);

    /******************************* direction -3 *********************************/
    ixpm=g_idn[ix][3]; ixpm2=g_idn[ixpm][3];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ixpm][3]; upm2=&g_gauge_field[ixpm2][3];
    Fadd2hop(rr, spm2, phim[3], -0.125*rho_BSM, +1.0, upm, upm2, phase_3, HOP_DN );
    p3add(rr, spm, upm, 0, -0.5*phase_3);

  } 
#ifdef OMP
  } /* OpenMP closing brace */
#endif
}

void D_psi_dagger_BSM2(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_dagger_BSM2 (D_psi_BSM2.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
    update_backward_gauge(g_gauge_field);
  }
#endif
#ifdef MPI
  generic_exchange(Q, sizeof(bispinor));
#endif
        
#ifdef OMP
#pragma omp parallel
  {
#endif

  int ix;
  int ixpm, ixpm2;
  su3 * restrict upm;
  su3 * restrict upm2;                 // U(x)_{+-\mu}, U(x)_{+-2\mu}
  bispinor * restrict rr;              // P(x)
  bispinor const * restrict s;         // Q(x)
  bispinor const * restrict spm;
  bispinor  const * restrict spm2;     // Q(x+-\mu), Q(x+-2\mu)
  scalar phi[4];                       // phi_i(x)
  scalar phip[4][4];                   // phi_i(x+mu) = phip[mu][i]
  scalar phim[4][4];                   // phi_i(x-mu) = phim[mu][i]

    /************************ loop over all lattice sites *************************/

#ifdef OMP
#pragma omp for
#endif
  for (ix=0;ix<VOLUME;ix++)
  {
    rr = (bispinor *) P + ix;
    s  = (bispinor *) Q + ix;

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

    // tmpr = m0_BSM*Q(x)
    //_vector_mul(rr->sp_up.s0, m0_BSM, s->sp_up.s0);
    //_vector_mul(rr->sp_up.s1, m0_BSM, s->sp_up.s1);
    //_vector_mul(rr->sp_up.s2, m0_BSM, s->sp_up.s2);
    //_vector_mul(rr->sp_up.s3, m0_BSM, s->sp_up.s3);
    //_vector_mul(rr->sp_dn.s0, m0_BSM, s->sp_dn.s0);
    //_vector_mul(rr->sp_dn.s1, m0_BSM, s->sp_dn.s1);
    //_vector_mul(rr->sp_dn.s2, m0_BSM, s->sp_dn.s2);
    //_vector_mul(rr->sp_dn.s3, m0_BSM, s->sp_dn.s3);
    // no longer needed now...
    _spinor_null(rr->sp_up);
    _spinor_null(rr->sp_dn);


    // tmpr += (\eta_BSM) * Fbar(x)*Q(x)
    Fadd(rr, s, phi, eta_BSM, -1.);

    // tmpr += \sum_\mu (\rho_BSM/8) * Fbar(x+-\mu)*Q
    for( int mu=0; mu<4; mu++ ) {
      Fadd(rr, s, phip[mu], 0.125*rho_BSM, -1.);
      Fadd(rr, s, phim[mu], 0.125*rho_BSM, -1.);
    }


    // the hopping part:
    // tmpr += +-1/2 \sum_\mu (\gamma_\mu - \rho_BSM/4*F(x+-\mu)*U_{+-\mu}(x)U_{x+-2\mu)*Q(x+-2\mu)
    /******************************* direction +0 *********************************/
    ixpm=g_iup[ix][0]; ixpm2=g_iup[ixpm][0];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ix][0]; upm2=&g_gauge_field[ixpm][0];
    Fadd2hop( rr, spm2, phip[0], -0.125*rho_BSM, -1.0, upm, upm2, phase_0, HOP_UP );
    p0add(rr, spm, upm, 0, -0.5*phase_0);

    /******************************* direction -0 *********************************/
    ixpm=g_idn[ix][0]; ixpm2=g_idn[ixpm][0];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ixpm][0]; upm2=&g_gauge_field[ixpm2][0];
    Fadd2hop(rr, spm2, phim[0], -0.125*rho_BSM, -1.0, upm, upm2, phase_0, HOP_DN );
    p0add(rr, spm, upm, 0, +0.5*phase_0);
    
    /******************************* direction +1 *********************************/
    ixpm=g_iup[ix][1]; ixpm2=g_iup[ixpm][1];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ix][1]; upm2=&g_gauge_field[ixpm][1];
    Fadd2hop( rr, spm2, phip[1], -0.125*rho_BSM, -1.0, upm, upm2, phase_1, HOP_UP );
    p1add(rr, spm, upm, 0, -0.5*phase_1);

    /******************************* direction -1 *********************************/
    ixpm=g_idn[ix][1]; ixpm2=g_idn[ixpm][1];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ixpm][1]; upm2=&g_gauge_field[ixpm2][1];
    Fadd2hop(rr, spm2, phim[1], -0.125*rho_BSM, -1.0, upm, upm2, phase_1, HOP_DN );
    p1add(rr, spm, upm, 0, +0.5*phase_1);
    
    /******************************* direction +2 *********************************/
    ixpm=g_iup[ix][2]; ixpm2=g_iup[ixpm][2];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ix][2]; upm2=&g_gauge_field[ixpm][2];
    Fadd2hop( rr, spm2, phip[2], -0.125*rho_BSM, -1.0, upm, upm2, phase_2, HOP_UP );
    p2add(rr, spm, upm, 0, -0.5*phase_2);

    /******************************* direction -2 *********************************/
    ixpm=g_idn[ix][2]; ixpm2=g_idn[ixpm][2];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ixpm][2]; upm2=&g_gauge_field[ixpm2][2];
    Fadd2hop(rr, spm2, phim[2], -0.125*rho_BSM, +1.0, upm, upm2, phase_2, HOP_DN );
    p2add(rr, spm, upm, 0, 0.5*phase_2);
    
    /******************************* direction +3 *********************************/
    ixpm=g_iup[ix][3]; ixpm2=g_iup[ixpm][3];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ix][3]; upm2=&g_gauge_field[ixpm][3];
    Fadd2hop( rr, spm2, phip[3], -0.125*rho_BSM, -1.0, upm, upm2, phase_3, HOP_UP );
    p3add(rr, spm, upm, 0, -0.5*phase_3);

    /******************************* direction -3 *********************************/
    ixpm=g_idn[ix][3]; ixpm2=g_idn[ixpm][3];
    spm = (bispinor *) Q + ixpm; spm2 = (bispinor *) Q + ixpm2;
    upm=&g_gauge_field[ixpm][3]; upm2=&g_gauge_field[ixpm2][3];
    Fadd2hop(rr, spm2, phim[3], -0.125*rho_BSM, -1.0, upm, upm2, phase_3, HOP_DN );
    p3add(rr, spm, upm, 0, 0.5*phase_3);

  } 
#ifdef OMP
  } /* OpenMP closing brace */
#endif
}

/* Q2_psi_BSM acts on bispinor fields */
void Q2_psi_BSM2(bispinor * const P, bispinor * const Q){

  D_psi_dagger_BSM2(g_bispinor_field[3] , Q);
  D_psi_BSM2(P, g_bispinor_field[3]);
  assign_add_mul_r((spinor*)P, (spinor*)Q, m0_BSM, 2*VOLUME); 
  /* Q and P are spinor, not bispinor ==> made a cast */
  /* the use of [3] has to be changed to avoid future conflicts */

}
