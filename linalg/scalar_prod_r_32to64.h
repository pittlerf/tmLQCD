#ifndef _SCALAR_PROD_R_32TO64_H
#define _SCALAR_PROD_R_32TO64_H

#include "su3.h"

/* Returns the real part of the scalar product (*R,*S) */
double scalar_prod_r_32to64(const spinor32 * const S, const spinor32 * const R, const int N, const int parallel);

#endif
