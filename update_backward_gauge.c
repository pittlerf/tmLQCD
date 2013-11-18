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
 ***********************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#ifdef OMP
#include <omp.h>
#endif
#include <stdlib.h>
#include "global.h"
#include "su3.h"
#include "update_backward_gauge.h"


#if defined _USE_HALFSPINOR
void update_backward_gauge(su3 ** const gf) {
#ifndef OMP
  #include "function_bodies/update_backward_gauge_halfspinor_body.ic"
  g_update_gauge_copy = 0;
#else
  if( omp_get_num_threads() > 1 ) {
    #include "function_bodies/update_backward_gauge_halfspinor_body.ic"
    #pragma omp single nowait
    {
      g_update_gauge_copy = 0;
    }
  } else {
    #pragma omp parallel
    {
      #include "function_bodies/update_backward_gauge_halfspinor_body.ic"
    }
    g_update_gauge_copy = 0;
  }
#endif
  return;
}

#elif _USE_TSPLITPAR 

void update_backward_gauge(su3 ** const gf) {
#ifndef OMP
  #include "function_bodies/update_backward_gauge_tsplitpar_body.ic"
  g_update_gauge_copy = 0;
#else
  if( omp_get_num_threads() > 1 ) {
    #include "function_bodies/update_backward_gauge_tsplitpar_body.ic"
    #pragma omp single nowait
    {
      g_update_gauge_copy = 0;
    }
  } else {
    #pragma omp parallel
    {
      #include "function_bodies/update_backward_gauge_tsplitpar_body.ic"
    }
    g_update_gauge_copy = 0;
  }
#endif
  return;
}

#else

void update_backward_gauge(su3 ** const gf) {
#ifndef OMP
  #include "function_bodies/update_backward_gauge_fullspinor_body.ic"
  g_update_gauge_copy = 0;
#else
  if( omp_get_num_threads() > 1 ) {
    #include "function_bodies/update_backward_gauge_fullspinor_body.ic"
    #pragma omp single nowait
    {
      g_update_gauge_copy = 0;
    }
  } else {
    #pragma omp parallel
    {
      #include "function_bodies/update_backward_gauge_fullspinor_body.ic"
    }
    g_update_gauge_copy = 0;
  }
#endif
  return;
}

#endif
