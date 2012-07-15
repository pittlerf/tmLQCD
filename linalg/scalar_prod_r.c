/***********************************************************************
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
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
 * File scalar_prod_r.c
 *
 *   double scalar_prod_r(spinor * const S,spinor * const R, const int N)
 *     Returns the real part of the scalar product (*R,*S)
 *
 *******************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#ifdef MPI
# include <mpi.h>
#endif
#include "su3.h"
#include "scalar_prod_r.h"

/*  R input, S input */

#include <complex.h>

#if (defined BGQ && defined XLC)

double scalar_prod_r(spinor * const S, spinor * const R, const int N, const int parallel) {
  vector4double ks, kc, ds, tr, ts, tt;
  vector4double x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5;
  vector4double z0, z1, z2, z3, z4, z4;
  double *s, *r;
#ifdef MPI
  vector4double buffer;
#endif
  ks = vec_splats(0.0);
  kc = vec_splats(0.0);

#pragma unroll(6)
  for (int ix = 0; ix < N; ++ix) {
    s=(double*)((spinor *) S + ix);
    r=(double*)((spinor *) R + ix);
    x0 = vec_ld(0, s);
    x1 = vec_ld(0, s+4);
    x2 = vec_ld(0, s+8);
    x3 = vec_ld(0, s+12);
    x4 = vec_ld(0, s+16);
    x5 = vec_ld(0, s+20);
    y0 = vec_ld(0, r);
    y1 = vec_ld(0, r+4);
    y2 = vec_ld(0, r+8);
    y3 = vec_ld(0, r+12);
    y4 = vec_ld(0, r+16);
    y5 = vec_ld(0, r+20);
    z0 = vec_mul(x0, y0);
    z1 = vec_mul(x1, y1);
    z2 = vec_mul(x2, y2);
    z3 = vec_mul(x3, y3);
    z4 = vec_mul(x4, y4);
    z5 = vec_mul(x5, y5);
    x0 = vec_add(z0, z1);
    x1 = vec_add(z2, z3);
    x2 = vec_add(z4, z5);
    x3 = vec_add(x0, x1);
    ds = vec_add(x2, x3);

    tr = vec_add(ds, kc);
    ts = vec_add(tr, ks);
    tt = vec_sub(ts, ks);
    ks = ts;
    kc = vec_sub(tr, tt);
  }
#if defined MPI
  if(parallel) {
    buffer = kc;
    MPI_Allreduce(&buffer, &kc, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return(buffer1);
  }
#endif
  return (kc[0] + kc[1] + kc[2] + kc[3]);
}

#else

double scalar_prod_r(spinor * const S, spinor * const R, const int N, const int parallel)
{
  
  static double ks,kc,ds,tr,ts,tt;
  spinor *s,*r;
  
  ks=0.0;
  kc=0.0;
  
#if (defined BGL && defined XLC)
  __alignx(16, S);
  __alignx(16, R);
#endif

  for (int ix = 0; ix < N; ++ix) {
    s=(spinor *) S + ix;
    r=(spinor *) R + ix;
    
    ds = creal(r->s0.c0 * conj(s->s0.c0)) + creal(r->s0.c1 * conj(s->s0.c1)) + creal(r->s0.c2 * conj(s->s0.c2)) +
      creal(r->s1.c0 * conj(s->s1.c0)) + creal(r->s1.c1 * conj(s->s1.c1)) + creal(r->s1.c2 * conj(s->s1.c2)) +
      creal(r->s2.c0 * conj(s->s2.c0)) + creal(r->s2.c1 * conj(s->s2.c1)) + creal(r->s2.c2 * conj(s->s2.c2)) +
      creal(r->s3.c0 * conj(s->s3.c0)) + creal(r->s3.c1 * conj(s->s3.c1)) + creal(r->s3.c2 * conj(s->s3.c2));    
    
    tr=ds+kc;
    ts=tr+ks;
    tt=ts-ks;
    ks=ts;
    kc=tr-tt;
  }
  kc=ks+kc;

#if defined MPI
  if(parallel)
  {
    double buffer = kc;
    MPI_Allreduce(&buffer, &kc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif

  return kc;

}


#endif

#ifdef WITHLAPH
double scalar_prod_r_su3vect(su3_vector * const S,su3_vector * const R, const int N, const int parallel)
{
  int ix;
  static double ks,kc,ds,tr,ts,tt;
  su3_vector *s,*r;

  ks=0.0;
  kc=0.0;
  for (int ix = 0; ix < N; ++ix)
  {
    s = (su3_vector *) S + ix;
    r = (su3_vector *) R + ix;
  
    ds = creal(r->c0) * creal(s->c0) + cimag(r->c0) * cimag(s->c0) +
         creal(r->c1) * creal(s->c1) + cimag(r->c1) * cimag(s->c1) +
         creal(r->c2) * creal(s->c2) + cimag(r->c2) * cimag(s->c2);
  
    tr = ds + kc;
    ts = tr + ks;
    tt = ts-ks;
    ks = ts;
    kc = tr-tt;
  }
  kc = ks + kc;
#if defined MPI
  if(parallel)
  {
    MPI_Allreduce(&kc, &ks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return ks;
  }
#endif
  return kc;
}

#endif // WITHLAPH
