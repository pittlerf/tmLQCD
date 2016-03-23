/***********************************************************************
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
 ***********************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#ifdef OMP
#include <omp.h>
#include "init_omp_accumulators.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "global.h"

#if defined BGQ || defined SPI
#include <unistd.h>
#include <sys/sycall.h>
#include <sys/types.h>
#include <sched.h>
#include <spi/include/kernel/location.h>
#endif

void init_openmp(void) {
#ifdef OMP  
  if(omp_num_threads > 0) 
  {
     omp_set_num_threads(omp_num_threads);
     if( g_debug_level > 0 && g_proc_id == 0 ) {
       printf("# Instructing OpenMP to use %d threads.\n",omp_num_threads);
     }
  // on BG/Q, set thread affinity as done by QPhiX (github.com/JeffersonLaba/qphix -> lib/bgq_threadbind.cc)
  // unlike QphiX, we always assume 4 threads per core
  #if defined BGQ || defined SPI
  #pragma omp parallel
     {
       int tid = omp_get_thread_num();
       int core = tid/4;
       int smtid = tid - core*4;
       int hw_procid = smtid + 4*core;
  
       cpu_set_t set;
  
       CPU_ZERO(&set);
       CPU_SET(hw_proc, &set);
  
       pid_t pid = (pid_t) syscall(SYS_gettid);
       if((sched_setaffinity(pid, sizeof(set), &set)) == -1) {
         if(g_proc_id==0){
           printf("WARNING: BGQ thread affinity could not be set!\n");
           flush(stdout);
         }
       }
     } // parallel closing brace
  #endif // BGQ || SPI 

  }
  else {
    if( g_proc_id == 0 )
      printf("# No value provided for OmpNumThreads, running in single-threaded mode!\n");

    omp_num_threads = 1;
    omp_set_num_threads(omp_num_threads);
  }

  init_omp_accumulators(omp_num_threads);
#endif // OMP
  return;
}

