/***********************************************************************
 *
 * Copyright (C) 2009 Carsten Urbach
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

#include "gauge.ih"

void allocate_gauge_buffers(unsigned int count)
{
  if ((g_gauge_buffers.allocated + count) > g_gauge_buffers.max)
    fatal_error("Maximum number of allocated gauge fields exceeded.", "allocate_gauge_buffers");
  
  for (unsigned int ctr = 0; ctr < count; ++ctr)
  {
    void *raw = malloc(sizeof(void*) + ALIGN_BASE + sizeof(su3_tuple) * VOLUMEPLUSRAND + 1);
    if (raw == NULL)
      fatal_error("Could not allocate the requested amount of memory.", "allocate_gauge_buffers");
    size_t p = (size_t)raw + sizeof(void*);
    p = ((p + ALIGN_BASE) & ~ALIGN_BASE);
    ((void**)p)[-1] = raw;

    g_gauge_buffers.reserve[g_gauge_buffers.free] = (su3_tuple*)p;
    ++g_gauge_buffers.allocated;
    ++g_gauge_buffers.free;
  }
}
