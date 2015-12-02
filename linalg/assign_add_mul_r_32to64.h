#ifndef _ASSIGN_ADD_MUL_32TO64_H
#define _ASSIGN_ADD_MUL_32TO64_H

#include "su3.h"

/*   (*P) = (*P) + c(*Q)        c is a real constant   */
void assign_add_mul_r_32to64(spinor * const R, spinor32 * const S, const float c, const int N);

#endif
