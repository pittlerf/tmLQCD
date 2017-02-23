/***********************************************************************
 *
 * Copyright (C) 2015 Mario Schroeck
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

#include <errno.h>
#include "global.h"
#include "scalar.h"
#if defined MPI
#include "buffers/utils_nonblocking.h"
#endif

extern int scalar_precision_read_flag;
// TODO consider that input scalar field could be in single prec.

int read_scalar_field(char * filename, scalar ** const sf) {

  FILE *ptr;

  int count = 4*VOLUME;
  int scalarreadsize = ( scalar_precision_read_flag==64 ? sizeof(double) : sizeof(float) );

  ptr = fopen(filename,"rb");  // r for read, b for binary
  // read into buffer
  void *buffer;
  if((buffer = malloc(count*scalarreadsize)) == NULL) {
    printf ("malloc errno : %d\n",errno);
    errno = 0;
    return(2);
  }

  if( count > fread(buffer,scalarreadsize,count,ptr) )
    return(-1);

  // copy to sf
  for( int s = 0; s < 4; s++ ) {
    for( int i = 0; i < VOLUME; i++ ) {
      if ( scalar_precision_read_flag == 64 )
	sf[s][i] = ((double*)buffer)[4*i+s];
      else
	sf[s][i] = ((float*)buffer)[4*i+s];
    }
  }

  return(0);
}
int read_scalar_field_parallel( char * filename, scalar ** const sf){
  int t;
  FILE *ptr;
  int count = 4*LX*N_PROC_X*LY*N_PROC_Y*LZ*N_PROC_Z;
  int scalarreadsize = ( scalar_precision_read_flag==64 ? sizeof(double) : sizeof(float) );

  ptr = fopen(filename,"rb");  // r for read, b for binary

  // read into buffer
  void *buffer;
  if((buffer = malloc(count*scalarreadsize)) == NULL) {
    printf ("malloc errno : %d\n",errno);
    errno = 0;
    return(2);
  }

  for (t=0; t< T_global; ++t){
    
      if ( g_proc_id == 0 ){
         
          int nread= fread(buffer, scalarreadsize, count, ptr);
          if ( nread != count ) { printf("Error in reading the scalar fields, exiting ...\n"); exit(1); }
  
      }
#if defined MPI
      MPI_Barrier(g_cart_grid);
      MPI_Bcast(buffer, count,  scalar_precision_read_flag==64 ? MPI_DOUBLE : MPI_FLOAT ,0, g_cart_grid );
#endif
      int ix, j;
      for (ix=0; ix< VOLUME; ++ix){
         if ( g_coord[ix][0] == t ){
            int ind = LY*N_PROC_Y*LZ*N_PROC_Z*g_coord[ix][1] + LZ*N_PROC_Z*g_coord[ix][2] + g_coord[ix][3];
            for (j=0; j<4; ++j){
               if ( scalar_precision_read_flag == 64 )
                 sf[j][ix] = ((double*)buffer)[4*ind+j];
               else
                 sf[j][ix] = ((float*)buffer)[4*ind+j];

            }
         }
      }
#if defined MPI
      MPI_Barrier(g_cart_grid);
#endif
           
  }
  free(buffer);
  return(0);
}
int unit_scalar_field( scalar **sf){
   int i;
   for (i=0; i<VOLUME; ++i){
      sf[0][i]=1.;
      sf[1][i]=0.;
      sf[2][i]=0.;
      sf[3][i]=0.;
   }
   return (0);
}
void smear_scalar_fields( scalar ** smearedfield, scalar ** const sf ) {

   int ix;
   int in;

   scalar *tmps1= (scalar *)malloc(sizeof(scalar)*VOLUMEPLUSRAND );
   scalar *tmps2= (scalar *)malloc(sizeof(scalar)*VOLUMEPLUSRAND );

   scalar *hyperc= (scalar *)malloc(sizeof(scalar)*VOLUMEPLUSRAND );
   scalar *nearen= (scalar *)malloc(sizeof(scalar)*VOLUMEPLUSRAND );

   int neit, neix, neiy, neiz;
#if defined MPI
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

   int count=0;
#endif

// hypercubic smearing 

   for (in = 0; in<4 ; ++in ){
      for (ix=0; ix<VOLUME; ++ix){
         smearedfield[in][ix]=0.0;
      } 
   }
   for (in=0; in<4; ++in) {
      for (ix=0; ix<VOLUME; ++ix){
         nearen[ix]=sf[in][ix];
         hyperc[ix]=sf[in][ix];
      }
      for (neit=0; neit<2; ++neit)
         for (neix=0; neix<2; ++neix)
            for (neiy=0; neiy<2; ++neiy)
               for (neiz=0; neiz<2; ++neiz){

#if defined MPI
                  count=0;
                  generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), neit ? TDOWN : TUP, request, &count );
                  MPI_Waitall( count, request, statuses);
#endif
                  for (ix=0; ix<VOLUME; ++ix)
                     tmps1[ix]= sf[in][    neit ? g_idn[ix][TUP] : g_iup[ix][TUP] ];

#if defined_MPI
                  count=0;
                  generic_exchange_direction_nonblocking(  tmps1, sizeof(scalar), neix ? XDOWN : XUP, request, &count );
                  MPI_Waitall( count, request, statuses);
#endif
                  for (ix=0; ix<VOLUME; ++ix)
                     tmps2[ix]= tmps1[ neix ? g_idn[ix][XUP] : g_iup[ix][XUP] ];

#if defined MPI
                  count=0;
                  generic_exchange_direction_nonblocking(  tmps2, sizeof(scalar), neiy ? YDOWN : YUP, request, &count );
                  MPI_Waitall( count, request, statuses);
#endif
                  for (ix=0; ix<VOLUME; ++ix)
                     tmps1[ix]= tmps2[ neiy ? g_idn[ix][YUP] : g_iup[ix][YUP] ];

#if defined MPI
                  count=0;
                  generic_exchange_direction_nonblocking(  tmps1, sizeof(scalar), neix ? ZDOWN : ZUP, request, &count );
                  MPI_Waitall( count, request, statuses);
#endif
                  for (ix=0; ix<VOLUME; ++ix)
                     tmps2[ix]= tmps1[ neiz ? g_idn[ix][ZUP] : g_iup[ix][ZUP] ];

                  for (ix=0; ix<VOLUME; ++ix)
                     hyperc[ix]+=tmps2[ix];
               }
      for (ix =0; ix<VOLUME; ++ix){
         hyperc[ix]/=17.0;
      }
#if defined MPI
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), TDOWN, request, &count );
      MPI_Waitall( count, request, statuses);
#endif
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_idn[ix][TUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];
      
#if defined MPI
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), TUP ,  request, &count );
      MPI_Waitall( count, request, statuses);     
#endif 
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_iup[ix][TUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];
      
#if defined MPI
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), XDOWN, request, &count );
      MPI_Waitall( count, request, statuses);   
#endif   
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_idn[ix][XUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];

#if defined MPI
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), XUP , request, &count );
      MPI_Waitall( count, request, statuses);
#endif
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_iup[ix][XUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];

#if defined_MPI
      count=0;    
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), YDOWN, request, &count );
      MPI_Waitall( count, request, statuses);
#endif      
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_idn[ix][YUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];
    
#if defined MPI
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), YUP, request, &count );
      MPI_Waitall( count, request, statuses);
#endif
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_iup[ix][YUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];

#if defined MPI
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), ZDOWN, request, &count );
      MPI_Waitall( count, request, statuses);  
#endif    
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_idn[ix][ZUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];

#if defined MPI
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), ZUP, request, &count );
      MPI_Waitall( count, request, statuses);
#endif
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_iup[ix][ZUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];


      for (ix =0; ix<VOLUME; ++ix){
         nearen[ix]/=9.0;
      }

      for (ix=0; ix<VOLUME; ++ix){
         smearedfield[in][ix]=0.5*(nearen[ix] + hyperc[ix] );
      }
   }
   free(tmps1);
   free(tmps2);

   free(hyperc);
   free(nearen);
#if defined MPI
   free(request);
#endif
}
void smear_scalar_fields_correlator( scalar **smearedfield, scalar ** const sf ) {

   int x0,y0,z0,t0;
   double **timeslicesum;
#if defined MPI
   double mpi_res;
#endif
   int j;
   for (j = 0; j<4 ; ++j ){
      for (x0=0; x0<VOLUME; ++x0){
         smearedfield[j][x0]=0.0;
      }
   }
   timeslicesum=(double **)malloc(sizeof(double *)*T);
   for (j=0; j<T; ++j)
      timeslicesum[j] = (double *)malloc(sizeof(double)*4);
   for (j=0; j<T; ++j){
      timeslicesum[j][0]=0.;
      timeslicesum[j][1]=0.;
      timeslicesum[j][2]=0.;
      timeslicesum[j][3]=0.;
   }
   for (j=0; j<VOLUME; ++j){
          /* get (t,x,y,z) from j */
      t0 = j/(LX*LY*LZ);
      x0 = (j-t0*(LX*LY*LZ))/(LY*LZ);
      y0 = (j-t0*(LX*LY*LZ)-x0*(LY*LZ))/(LZ);
      z0 = (j-t0*(LX*LY*LZ)-x0*(LY*LZ) - y0*LZ);
      timeslicesum[t0][0]+=sf[0][j];
      timeslicesum[t0][1]+=sf[1][j];
      timeslicesum[t0][2]+=sf[2][j];
      timeslicesum[t0][3]+=sf[3][j];
   }
#if defined MPI
   for (j=0; j<T; ++j){
      MPI_Allreduce(&timeslicesum[j][0], &mpi_res, 1, MPI_DOUBLE, MPI_SUM, g_mpi_time_slices);
      timeslicesum[j][0]=mpi_res;
      MPI_Allreduce(&timeslicesum[j][1], &mpi_res, 1, MPI_DOUBLE, MPI_SUM, g_mpi_time_slices);
      timeslicesum[j][1]=mpi_res;
      MPI_Allreduce(&timeslicesum[j][2], &mpi_res, 1, MPI_DOUBLE, MPI_SUM, g_mpi_time_slices);
      timeslicesum[j][2]=mpi_res;
      MPI_Allreduce(&timeslicesum[j][3], &mpi_res, 1, MPI_DOUBLE, MPI_SUM, g_mpi_time_slices);
      timeslicesum[j][3]=mpi_res;
   }
#endif
   for (j=0; j<T; ++j){
      timeslicesum[j][0]/=(double)LX*LY*LZ*N_PROC_X*N_PROC_Y*N_PROC_Z;
      timeslicesum[j][1]/=(double)LX*LY*LZ*N_PROC_X*N_PROC_Y*N_PROC_Z;
      timeslicesum[j][2]/=(double)LX*LY*LZ*N_PROC_X*N_PROC_Y*N_PROC_Z;
      timeslicesum[j][3]/=(double)LX*LY*LZ*N_PROC_X*N_PROC_Y*N_PROC_Z;
   }
   for (j=0; j<VOLUME; ++j){
          /* get (t,x,y,z) from j */
      t0 = j/(LX*LY*LZ);
      x0 = (j-t0*(LX*LY*LZ))/(LY*LZ);
      y0 = (j-t0*(LX*LY*LZ)-x0*(LY*LZ))/(LZ);
      z0 = (j-t0*(LX*LY*LZ)-x0*(LY*LZ) - y0*LZ);

      smearedfield[0][j]=timeslicesum[t0][0];
      smearedfield[1][j]=timeslicesum[t0][1];
      smearedfield[2][j]=timeslicesum[t0][2];
      smearedfield[3][j]=timeslicesum[t0][3];
   }

   for (j=0; j<T; ++j)
      free(timeslicesum[j]);
   free(timeslicesum);

}

