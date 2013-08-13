/***********************************************************************
*
* Copyright (C) 1995 Ulli Wolff, Stefan Sint
*               2001,2005 Martin Hasenbusch
*               2011,2012 Carsten Urbach
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
***********************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#ifdef SSE
# undef SSE
#endif
#ifdef SSE2
# undef SSE2
#endif
#ifdef SSE3
# undef SSE3
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#ifdef MPI
# include <mpi.h>
#endif
#ifdef OMP
# include <omp.h>
#endif
#include "global.h"
#include "su3.h"
#include "sse.h"
#include "su3adj.h"

double energy_density(gauge_field_t const gf)
{
  // NOTE This normalization is not obviously the correct one...
  static const double normalization = 1 / (6.0 /* Plaquette directions */ * 3.0 /* Nc */ * VOLUME);
  
#ifdef OMP
#pragma omp parallel
  {
#endif
    su3 ALIGN v1, v2, plaq;
    double ed = 0.0, rem = 0.0, tmp1 = 0.0;

  /*  compute the clover-leave */
  /*  l  __   __
        |  | |  |
        |__| |__|
        __   __
        |  | |  |
        |__| |__| k  */
  
#ifdef OMP
#pragma omp for
#endif
    for(int x = 0; x < VOLUME; x++)
    {
      for(int k = 0; k < 4; k++)
      {
        for(int l = k+1; l < 4; l++)
        {
          int xpk = g_iup[x][k];
          int xpl = g_iup[x][l];
          int xmk = g_idn[x][k];
          int xml = g_idn[x][l];
          int xpkml = g_idn[xpk][l];
          int xplmk = g_idn[xpl][k];
          int xmkml = g_idn[xml][k];
          const su3 *w1 = &gf[x][k];
          const su3 *w2 = &gf[xpk][l];
          const su3 *w3 = &gf[xpl][k];
          const su3 *w4 = &gf[x][l];
          _su3_times_su3(v1, *w1, *w2);
          _su3_times_su3(v2, *w4, *w3);
          _su3_times_su3d(plaq, v1, v2);
          w1 = &gf[x][l];
          w2 = &gf[xplmk][k];
          w3 = &gf[xmk][l];
          w4 = &gf[xmk][k];
          _su3_times_su3d(v1, *w1, *w2);
          _su3d_times_su3(v2, *w3, *w4);
          _su3_times_su3_acc(plaq, v1, v2);
          w1 = &gf[xmk][k];
          w2 = &gf[xmkml][l];
          w3 = &gf[xmkml][k];
          w4 = &gf[xml][l];
          _su3_times_su3(v1, *w2, *w1);
          _su3_times_su3(v2, *w3, *w4);
          _su3d_times_su3_acc(plaq, v1, v2);
          w1 = &gf[xml][l];
          w2 = &gf[xml][k];
          w3 = &gf[xpkml][l];
          w4 = &gf[x][k];
          _su3d_times_su3(v1, *w1, *w2);
          _su3_times_su3d(v2, *w3, *w4);
          _su3_times_su3_acc(plaq, v1, v2);
          _su3_dagger(v2, plaq); 
          _su3_minus_su3(plaq, plaq, v2); // At this point: anti-hermitian clover average in direction k, l
          
          _real_trace_su3_squared(tmp1, plaq); // This should actually be the energy density already...
          
          // Kahan summation for each thread
          tmp1 -= rem;
          double tmp2 = ed + tmp1;
          rem = (tmp2 - ed) - tmp1;
          ed = tmp2;
        }
      }   
    }
#ifdef OMP
    int thread_num = omp_get_thread_num();
    g_omp_acc_re[thread_num] = ed;
  } /* OpenMP closing brace */
  
  // Kahan summation on all threads
  double ed = 0.0, rem = 0.0;
  for(int i = 0; i < omp_num_threads; ++i)
  {
    double tmp1 = g_omp_acc_re[i] - rem;
    double tmp2 = ed + tmp1;
    rem = (tmp2 - ed) - tmp1;
    ed = tmp2;
  }
  
#endif
  // With or without OpenMP, ed now contains the Kahan summed value over the whole lattice
  return (normalization * ed);
}
