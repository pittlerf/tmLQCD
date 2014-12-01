/***********************************************************************
 *
 * Copyright (C) 2001 Martin Luescher
 * original code 
 * changed and extended for twisted mass 2002 Andrea Shindler
 *               2007,2008 Carsten Urbach
 *
 * Blue Gene version Copyright (C) 2007 Carsten Urbach 
 * Block Dirac operator Copyright (C) 2008 Carsten Urbach
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
 * Action of a Dirac operator D (BSM toy model) on a given bispinor field
 *
 * various versions including a block version.
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
#include "operator/D_psi.h"
#include "solver/dirac_operator_eigenvectors.h"
#include "buffers/utils.h"




static inline void p0add(spinor * restrict const tmpr , spinor const * restrict const s, 
			 su3 const * restrict const u, const _Complex double phase) {

#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
#ifdef OMP
#undef static
#endif

  _vector_add(psi,s->s0, s->s2);
  _su3_multiply(chi, (*u), psi);

  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);
  _vector_add_assign(tmpr->s2, psi);

  _vector_add(psi, s->s1, s->s3);
  _su3_multiply(chi, (*u), psi);

  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);
  _vector_add_assign(tmpr->s3, psi);

  return;
}


static inline void m0add(spinor * restrict const tmpr, spinor const * restrict const s, 
			 su3 const * restrict const u, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
#ifdef OMP
#undef static
#endif

  _vector_sub(psi, s->s0, s->s2);
  _su3_inverse_multiply(chi, (*u), psi);

  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);
  _vector_sub_assign(tmpr->s2, psi);

  _vector_sub(psi, s->s1, s->s3);
  _su3_inverse_multiply(chi, (*u), psi);

  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);
  _vector_sub_assign(tmpr->s3, psi);

  return;
}

static inline void p1add(spinor * restrict const tmpr, spinor const * restrict const s, 
			 su3 const * restrict const u, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
#ifdef OMP
#undef static
#endif

  _vector_i_add(psi,s->s0,s->s3);
  _su3_multiply(chi,(*u),psi);

  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);
  _vector_i_sub_assign(tmpr->s3, psi);
 
  _vector_i_add(psi, s->s1, s->s2);
  _su3_multiply(chi, (*u), psi);

  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);
  _vector_i_sub_assign(tmpr->s2, psi);

  return;
}

static inline void m1add(spinor * restrict const tmpr, spinor const * restrict const s, 
			 su3 const * restrict const u, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
#ifdef OMP
#undef static
#endif

  _vector_i_sub(psi,s->s0, s->s3);
  _su3_inverse_multiply(chi,(*u), psi);

  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);
  _vector_i_add_assign(tmpr->s3, psi);

  _vector_i_sub(psi, s->s1, s->s2);
  _su3_inverse_multiply(chi, (*u), psi);

  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);
  _vector_i_add_assign(tmpr->s2, psi);

  return;
}

static inline void p2add(spinor * restrict const tmpr, spinor const * restrict const s, 
			 su3 const * restrict const u, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
#ifdef OMP
#undef static
#endif

  _vector_add(psi,s->s0,s->s3);
  _su3_multiply(chi, (*u), psi);

  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);
  _vector_add_assign(tmpr->s3, psi);

  _vector_sub(psi,s->s1,s->s2);
  _su3_multiply(chi, (*u), psi);

  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);
  _vector_sub_assign(tmpr->s2, psi);


  return;
}

static inline void m2add(spinor * restrict const tmpr, spinor const * restrict const s, 
			 su3 const * restrict const u, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
#ifdef OMP
#undef static
#endif

  _vector_sub(psi, s->s0, s->s3);
  _su3_inverse_multiply(chi, (*u), psi);

  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);
  _vector_sub_assign(tmpr->s3, psi);

  _vector_add(psi, s->s1, s->s2);
  _su3_inverse_multiply(chi, (*u),psi);

  _complexcjg_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);
  _vector_add_assign(tmpr->s2, psi);

  return;
}

static inline void p3add(spinor * restrict const tmpr, spinor const * restrict const s, 
			 su3 const * restrict const u, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
#ifdef OMP
#undef static
#endif

  _vector_i_add(psi, s->s0, s->s2);
  _su3_multiply(chi, (*u), psi);

  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s0, psi);
  _vector_i_sub_assign(tmpr->s2, psi);

  _vector_i_sub(psi,s->s1, s->s3);
  _su3_multiply(chi, (*u), psi);

  _complex_times_vector(psi, phase, chi);
  _vector_add_assign(tmpr->s1, psi);
  _vector_i_add_assign(tmpr->s3, psi);

  return;
}

static inline void m3addandstore(spinor * restrict const r, spinor const * restrict const s, 
				 su3 const * restrict const u, const _Complex double phase,
         spinor const * restrict const tmpr) {
#ifdef OMP
#define static
#endif
  static su3_vector chi, psi;
#ifdef OMP
#undef static
#endif

  _vector_i_sub(psi,s->s0, s->s2);
  _su3_inverse_multiply(chi, (*u), psi);

  _complexcjg_times_vector(psi, phase, chi);
  _vector_add(r->s0, tmpr->s0, psi);
  _vector_i_add(r->s2, tmpr->s2, psi);

  _vector_i_add(psi, s->s1, s->s3);
  _su3_inverse_multiply(chi, (*u), psi);

  _complexcjg_times_vector(psi, phase, chi);
  _vector_add(r->s1, tmpr->s1, psi);
  _vector_i_sub(r->s3, tmpr->s3, psi);

  return;
}


/* operator F := \phi_0 + i \gamma_5 \tau^j \phi_j acting on bispinor field */
void F_psi(bispinor * const P, bispinor * const Q)
{
#ifdef OMP
#pragma omp parallel
  {
#endif

  int ix;
  scalar phi0;
  scalar phi1;
  scalar phi2;
  scalar phi3;

  bispinor * restrict out;
  bispinor const * restrict in;

  /************************ loop over all lattice sites *************************/

#ifdef OMP
#pragma omp for
#endif
  for (ix=0;ix<VOLUME;ix++)
  {
	  // get local bispinor fields
	  out = P + ix;
	  in  = Q + ix;

	  // get local scalar fields
	  phi0 = *(g_scalar_field[0] + ix);
	  phi1 = *(g_scalar_field[1] + ix);
	  phi2 = *(g_scalar_field[2] + ix);
	  phi3 = *(g_scalar_field[3] + ix);

	  // flavour 1:
	  // out_up = \phi_0 * in_up
	  _vector_mul(out->sp_up.s0, phi0, in->sp_up.s0);
	  _vector_mul(out->sp_up.s1, phi0, in->sp_up.s1);
	  _vector_mul(out->sp_up.s2, phi0, in->sp_up.s2);
	  _vector_mul(out->sp_up.s3, phi0, in->sp_up.s3);

	  // out_up += i \gamma_5 \phi_1 * in_dn
	  _vector_add_i_mul(out->sp_up.s0,  phi1, in->sp_dn.s0);
	  _vector_add_i_mul(out->sp_up.s1,  phi1, in->sp_dn.s1);
	  _vector_add_i_mul(out->sp_up.s2, -phi1, in->sp_dn.s2);
	  _vector_add_i_mul(out->sp_up.s3, -phi1, in->sp_dn.s3);

	  // out_up += \gamma_5 \phi_2 * in_dn
	  _vector_add_mul(out->sp_up.s0,  phi2, in->sp_dn.s0);
	  _vector_add_mul(out->sp_up.s1,  phi2, in->sp_dn.s1);
	  _vector_add_mul(out->sp_up.s2, -phi2, in->sp_dn.s2);
	  _vector_add_mul(out->sp_up.s3, -phi2, in->sp_dn.s3);

	  // out_up += i \gamma_5 \phi_3 * in_up
	  _vector_add_i_mul(out->sp_up.s0,  phi3, in->sp_up.s0);
	  _vector_add_i_mul(out->sp_up.s1,  phi3, in->sp_up.s1);
	  _vector_add_i_mul(out->sp_up.s2, -phi3, in->sp_up.s2);
	  _vector_add_i_mul(out->sp_up.s3, -phi3, in->sp_up.s3);


	  // flavour 2:
	  // out_dn = \phi_0 * in_dn
	  _vector_mul(out->sp_dn.s0, phi0, in->sp_dn.s0);
	  _vector_mul(out->sp_dn.s1, phi0, in->sp_dn.s1);
	  _vector_mul(out->sp_dn.s2, phi0, in->sp_dn.s2);
	  _vector_mul(out->sp_dn.s3, phi0, in->sp_dn.s3);

	  // out_dn += i \gamma_5 \phi_1 * in_up
	  _vector_add_i_mul(out->sp_dn.s0,  phi1, in->sp_up.s0);
	  _vector_add_i_mul(out->sp_dn.s1,  phi1, in->sp_up.s1);
	  _vector_add_i_mul(out->sp_dn.s2, -phi1, in->sp_up.s2);
	  _vector_add_i_mul(out->sp_dn.s3, -phi1, in->sp_up.s3);

	  // out_dn += \gamma_5 \phi_2 * in_up
	  _vector_add_mul(out->sp_dn.s0, -phi2, in->sp_up.s0);
	  _vector_add_mul(out->sp_dn.s1, -phi2, in->sp_up.s1);
	  _vector_add_mul(out->sp_dn.s2,  phi2, in->sp_up.s2);
	  _vector_add_mul(out->sp_dn.s3,  phi2, in->sp_up.s3);

	  // out_dn += i \gamma_5 \phi_3 * in_dn
	  _vector_add_i_mul(out->sp_dn.s0, -phi3, in->sp_dn.s0);
	  _vector_add_i_mul(out->sp_dn.s1, -phi3, in->sp_dn.s1);
	  _vector_add_i_mul(out->sp_dn.s2,  phi3, in->sp_dn.s2);
	  _vector_add_i_mul(out->sp_dn.s3,  phi3, in->sp_dn.s3);
  }
#ifdef OMP
  } /* OpenMP closing brace */
#endif
}


/* D_psi_BSM acts on bispinor fields */
void D_psi_BSM(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM (D_psi_BSM.c):\n");
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

  // call F_psi here

#ifdef OMP
#pragma omp parallel
  {
#endif

  int ix,iy;
  su3 * restrict up,* restrict um;
  spinor * restrict rr; 
  spinor const * restrict s;
  spinor const * restrict sp;
  spinor const * restrict sm;
  _Complex double rho1, rho2;
  spinor tmpr;

  rho1 = 1. + g_mu * I;
  rho2 = conj(rho1);

  /************************ loop over all lattice sites *************************/

#ifdef OMP
#pragma omp for
#endif
  for (ix=0;ix<VOLUME;ix++)
  {
    rr  = (spinor *) &P->sp_up +ix;
    s   = (spinor *) &Q->sp_up +ix;

    _complex_times_vector(tmpr.s0, rho1, s->s0);
    _complex_times_vector(tmpr.s1, rho1, s->s1);
    _complex_times_vector(tmpr.s2, rho2, s->s2);
    _complex_times_vector(tmpr.s3, rho2, s->s3);

    /******************************* direction +0 *********************************/
    iy=g_iup[ix][0];
    sp = (spinor *) &Q->sp_up +iy;
    up=&g_gauge_field[ix][0];
    p0add(&tmpr, sp, up, phase_0);

    /******************************* direction -0 *********************************/
    iy=g_idn[ix][0];
    sm  = (spinor *) &Q->sp_up +iy;
    um=&g_gauge_field[iy][0];
    m0add(&tmpr, sm, um, phase_0);

    /******************************* direction +1 *********************************/
    iy=g_iup[ix][1];
    sp = (spinor *) &Q->sp_up +iy;
    up=&g_gauge_field[ix][1];
    p1add(&tmpr, sp, up, phase_1);

    /******************************* direction -1 *********************************/
    iy=g_idn[ix][1];
    sm = (spinor *) &Q->sp_up +iy;
    um=&g_gauge_field[iy][1];
    m1add(&tmpr, sm, um, phase_1);

    /******************************* direction +2 *********************************/
    iy=g_iup[ix][2];
    sp = (spinor *) &Q->sp_up +iy;
    up=&g_gauge_field[ix][2];
    p2add(&tmpr, sp, up, phase_2);

    /******************************* direction -2 *********************************/
    iy=g_idn[ix][2];
    sm = (spinor *) &Q->sp_up +iy;
    um=&g_gauge_field[iy][2];
    m2add(&tmpr, sm, um, phase_2);

    /******************************* direction +3 *********************************/
    iy=g_iup[ix][3];
    sp = (spinor *) &Q->sp_up +iy;
    up=&g_gauge_field[ix][3];
    p3add(&tmpr, sp, up, phase_3);

    /******************************* direction -3 *********************************/
    iy=g_idn[ix][3];
    sm = (spinor *) &Q->sp_up +iy;
    um=&g_gauge_field[iy][3];
    m3addandstore(rr, sm, um, phase_3, &tmpr);
  }
#ifdef OMP
  } /* OpenMP closing brace */
#endif
}

