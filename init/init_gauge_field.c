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
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include "global.h"
#include "su3.h"
#include "sse.h"
#include "init_gauge_field.h"

su3 * gauge_field = NULL;
#ifdef _USE_BSM
su3 * smeared_gauge_field = NULL;
#endif
#ifdef _USE_TSPLITPAR
su3 * gauge_field_copyt = NULL;
su3 * gauge_field_copys = NULL;
#ifdef _USE_BSM
su3 * smeared_gauge_field_copyt = NULL;
su3 * smeared_gauge_field_copys = NULL;
#endif
#else
su3 * gauge_field_copy = NULL;
#ifdef _USE_BSM
su3 * smeared_gauge_field_copy = NULL;
#endif
#endif

int init_gauge_field(const int V, const int back) {
  int i=0;

#ifdef _USE_TSPLITPAR
  g_gauge_field_copyt = NULL;
  g_gauge_field_copys = NULL;
#ifdef _USE_BSM
  g_smeared_gauge_field_copyt = NULL;
  g_smeared_gauge_field_copys = NULL;
#endif
#else
  g_gauge_field_copy = NULL;
#ifdef _USE_BSM
  g_smeared_gauge_field_copy = NULL;
#endif
#endif

  if((void*)(g_gauge_field = (su3**)calloc(V, sizeof(su3*))) == NULL) {
    printf ("malloc errno : %d\n",errno); 
    errno = 0;
    return(1);
  }
#ifdef _USE_BSM
  if((void*)(g_smeared_gauge_field = (su3**)calloc(V, sizeof(su3*))) == NULL) { 
    printf ("malloc errno : %d\n",errno); 
    errno = 0;
    return(1);
  }
#endif
  if((void*)(gauge_field = (su3*)calloc(4*V+1, sizeof(su3))) == NULL) {
    printf ("malloc errno : %d\n",errno); 
    errno = 0;
    return(2);
  }
#ifdef _USE_BSM
  if((void*)(smeared_gauge_field = (su3*)calloc(4*V+1, sizeof(su3))) == NULL) {
    printf ("malloc errno : %d\n",errno);
    errno = 0;
    return(2);
  }
#endif
#if (defined SSE || defined SSE2 || defined SSE3)
  g_gauge_field[0] = (su3*)(((unsigned long int)(gauge_field)+ALIGN_BASE)&~ALIGN_BASE);
#ifdef _USE_BSM
  g_smeared_gauge_field[0] = (su3*)(((unsigned long int)(smeared_gauge_field)+ALIGN_BASE)&~ALIGN_BASE);
#endif
#else
  g_gauge_field[0] = gauge_field;
#ifdef _USE_BSM
  g_smeared_gauge_field[0] = smeared_gauge_field;
#endif

#endif
  for(i = 1; i < V; i++){
    g_gauge_field[i] = g_gauge_field[i-1]+4;
#ifdef _USE_BSM
    g_smeared_gauge_field[i] = g_smeared_gauge_field[i-1]+4;
#endif
  }

#  if defined _USE_HALFSPINOR
  if(back == 1) {
    /*
      g_gauge_field_copy[ieo][PM][sites/2][mu]
    */
    if((void*)(g_gauge_field_copy = (su3***)calloc(2, sizeof(su3**))) == NULL) {
      printf ("malloc errno : %d\n",errno); 
      errno = 0;
      return(3);
    }
#ifdef _USE_BSM
    if((void*)(g_smeared_gauge_field_copy = (su3***)calloc(2, sizeof(su3**))) == NULL) {
      printf ("malloc errno : %d\n",errno);
      errno = 0;
      return(3);
    }
#endif
    if((void*)(g_gauge_field_copy[0] = (su3**)calloc(VOLUME, sizeof(su3*))) == NULL) {
      printf ("malloc errno : %d\n",errno); 
      errno = 0;
      return(3);
    }
#ifdef _USE_BSM
    if((void*)(g_smeared_gauge_field_copy[0] = (su3**)calloc(VOLUME, sizeof(su3*))) == NULL) {
      printf ("malloc errno : %d\n",errno);
      errno = 0;
      return(3);
    }
#endif
    g_gauge_field_copy[1] = g_gauge_field_copy[0] + (VOLUME)/2;
#ifdef _USE_BSM 
    g_smeared_gauge_field_copy[1] = g_smeared_gauge_field_copy[0] + (VOLUME)/2;
#endif
    if((void*)(gauge_field_copy = (su3*)calloc(4*(VOLUME)+1, sizeof(su3))) == NULL) {
      printf ("malloc errno : %d\n",errno); 
      errno = 0;
      return(4);
    }
#ifdef _USE_BSM
    if((void*)(smeared_gauge_field_copy = (su3*)calloc(4*(VOLUME)+1, sizeof(su3))) == NULL) {
      printf ("malloc errno : %d\n",errno);
      errno = 0;
      return(4);
    }
#endif
#    if (defined SSE || defined SSE2 || defined SSE3)
    g_gauge_field_copy[0][0] = (su3*)(((unsigned long int)(gauge_field_copy)+ALIGN_BASE)&~ALIGN_BASE);
#ifdef _USE_BSM
    g_smeared_gauge_field_copy[0][0] = (su3*)(((unsigned long int)(smeared_gauge_field_copy)+ALIGN_BASE)&~ALIGN_BASE);
#endif
#else
    g_gauge_field_copy[0][0] = gauge_field_copy;
#ifdef _USE_BSM
    g_smeared_gauge_field_copy[0][0] = smeared_gauge_field_copy;
#endif
#endif
    for(i = 1; i < (VOLUME)/2; i++) {
      g_gauge_field_copy[0][i] = g_gauge_field_copy[0][i-1]+4;
    }
    g_gauge_field_copy[1][0] = g_gauge_field_copy[0][0] + 2*VOLUME; 
    for(i = 1; i < (VOLUME)/2; i++) {
      g_gauge_field_copy[1][i] = g_gauge_field_copy[1][i-1]+4;
    }
#ifdef _USE_BSM
  for(i = 1; i < (VOLUME)/2; i++) {
      g_smeared_gauge_field_copy[0][i] = g_smeared_gauge_field_copy[0][i-1]+4;
    }
    g_smeared_gauge_field_copy[1][0] = g_smeared_gauge_field_copy[0][0] + 2*VOLUME;
    for(i = 1; i < (VOLUME)/2; i++) {
      g_smeared_gauge_field_copy[1][i] = g_smeared_gauge_field_copy[1][i-1]+4;
    }
#endif
  }
#  elif defined _USE_TSPLITPAR
  if(back == 1) {
    if((void*)(g_gauge_field_copyt = (su3**)calloc((VOLUME+RAND), sizeof(su3*))) == NULL) {
      printf ("malloc errno : %d\n",errno); 
      errno = 0;
      return(3);
    }
    if((void*)(g_gauge_field_copys = (su3**)calloc((VOLUME+RAND), sizeof(su3*))) == NULL) {
      printf ("malloc errno : %d\n",errno); 
      errno = 0;
      return(3);
    }
    if((void*)(gauge_field_copyt = (su3*)calloc(2*(VOLUME+RAND)+1, sizeof(su3))) == NULL) {
      printf ("malloc errno : %d\n",errno); 
      errno = 0;
      return(4);
    }
    if((void*)(gauge_field_copys = (su3*)calloc(6*(VOLUME+RAND)+1, sizeof(su3))) == NULL) {
      printf ("malloc errno : %d\n",errno); 
      errno = 0;
      return(4);
    }
#ifdef _USE_BSM
   if((void*)(g_smeared_gauge_field_copyt = (su3**)calloc((VOLUME+RAND), sizeof(su3*))) == NULL) {
      printf ("malloc errno : %d\n",errno);
      errno = 0;
      return(3);
    }
    if((void*)(g_smeared_gauge_field_copys = (su3**)calloc((VOLUME+RAND), sizeof(su3*))) == NULL) {
      printf ("malloc errno : %d\n",errno);
      errno = 0;
      return(3);
    }
    if((void*)(g_smeared_gauge_field_copyt = (su3*)calloc(2*(VOLUME+RAND)+1, sizeof(su3))) == NULL) {
      printf ("malloc errno : %d\n",errno);
      errno = 0;
      return(4);
    }
    if((void*)(g_smeared_gauge_field_copys = (su3*)calloc(6*(VOLUME+RAND)+1, sizeof(su3))) == NULL) {
      printf ("malloc errno : %d\n",errno);
      errno = 0;
      return(4);
    }
#endif
#if (defined SSE || defined SSE2 || defined SSE3)
    g_gauge_field_copyt[0] = (su3*)(((unsigned long int)(gauge_field_copyt)+ALIGN_BASE)&~ALIGN_BASE);
    g_gauge_field_copys[0] = (su3*)(((unsigned long int)(gauge_field_copys)+ALIGN_BASE)&~ALIGN_BASE);
#ifdef _USE_BSM
    g_smeared_gauge_field_copyt[0] = (su3*)(((unsigned long int)(smeared_gauge_field_copyt)+ALIGN_BASE)&~ALIGN_BASE);
    g_smeared_gauge_field_copys[0] = (su3*)(((unsigned long int)(smeared_gauge_field_copys)+ALIGN_BASE)&~ALIGN_BASE);
#endif
#else
    g_gauge_field_copyt[0] = gauge_field_copyt;
    g_gauge_field_copys[0] = gauge_field_copys;
#ifdef _USE_BSM
    g_smeared_gauge_field_copyt[0] = smeared_gauge_field_copyt;
    g_smeared_gauge_field_copys[0] = smeared_gauge_field_copys;
#endif
#    endif
    for(i = 1; i < (VOLUME+RAND); i++) {
      g_gauge_field_copyt[i] = g_gauge_field_copyt[i-1]+2;
      g_gauge_field_copys[i] = g_gauge_field_copys[i-1]+6;
    }
#ifdef _USE_BSM
    for(i = 1; i < (VOLUME+RAND); i++) {
      g_smeared_gauge_field_copyt[i] = g_smeared_gauge_field_copyt[i-1]+2;
      g_smeared_gauge_field_copys[i] = g_smeared_gauge_field_copys[i-1]+6;
    }
#endif
  }
#  else  /* than _USE_HALFSPINOR or _USE_TSPLITPAR */
  if(back == 1) {
    if((void*)(g_gauge_field_copy = (su3**)calloc((VOLUME+RAND), sizeof(su3*))) == NULL) {
      printf ("malloc errno : %d\n",errno); 
      errno = 0;
      return(3);
    }
#ifdef _USE_BSM
    if((void*)(g_smeared_gauge_field_copy = (su3**)calloc((VOLUME+RAND), sizeof(su3*))) == NULL) {
      printf ("malloc errno : %d\n",errno);
      errno = 0;
      return(3);
    }
#endif
    if((void*)(gauge_field_copy = (su3*)calloc(8*(VOLUME+RAND)+1, sizeof(su3))) == NULL) {
      printf ("malloc errno : %d\n",errno); 
      errno = 0;
      return(4);
    }
#ifdef _USE_BSM
    if((void*)(smeared_gauge_field_copy = (su3*)calloc(8*(VOLUME+RAND)+1, sizeof(su3))) == NULL) {
      printf ("malloc errno : %d\n",errno);
      errno = 0;
      return(4);
    }
#endif

#  if (defined SSE || defined SSE2 || defined SSE3)
    g_gauge_field_copy[0] = (su3*)(((unsigned long int)(gauge_field_copy)+ALIGN_BASE)&~ALIGN_BASE);
#ifdef _USE_BSM
    g_smeared_gauge_field_copy[0] = (su3*)(((unsigned long int)(smeared_gauge_field_copy)+ALIGN_BASE)&~ALIGN_BASE);
#endif
#  else
    g_gauge_field_copy[0] = gauge_field_copy;
#ifdef _USE_BSM
    g_smeared_gauge_field_copy[0] = smeared_gauge_field_copy;
#endif
#  endif
    for(i = 1; i < (VOLUME+RAND); i++) {
      g_gauge_field_copy[i] = g_gauge_field_copy[i-1]+8;
    }
#ifdef _USE_BSM
    for(i = 1; i < (VOLUME+RAND); i++) {
      g_smeared_gauge_field_copy[i] = g_smeared_gauge_field_copy[i-1]+8;
    }
#endif
  }
#endif
  g_update_gauge_copy = 1;
  return(0);
}

void free_gauge_field() {
  free(gauge_field);
#ifdef _USE_BSM
  free(smeared_gauge_field);
#endif
  free(g_gauge_field);
#ifdef _USE_BSM
  free(g_smeared_gauge_field);
#endif
#  if defined _USE_TSPLITPAR
  free(gauge_field_copys);
#ifdef _USE_BSM
  free(smeared_gauge_field_copys);
#endif
  free(gauge_field_copyt);
#ifdef _USE_BSM
  free(smeared_gauge_field_copys);
#endif
#  else
  free(gauge_field_copy);
#ifdef _USE_BSM
  free(smeared_gauge_field_copy);
#endif
#  endif
}
