#include <math.h>

#define MAX_DIM 512
#define SOBOL_BITS 31

__constant__ unsigned int SOBOL_DIR[MAX_DIM][SOBOL_BITS];

__device__
double sobol_sample(unsigned int index, int dim)
{
    unsigned int g = index ^ (index >> 1);

    unsigned int x = 0;

    #pragma unroll
    for(int b=0;b<SOBOL_BITS;b++)
    {
        if(g & (1u << b))
            x ^= SOBOL_DIR[dim][b];
    }

    double u = (double)x * 2.3283064365386963e-10;

    // clamp to avoid inverse normal singularities
    u = fmax(u,1e-12);
    u = fmin(u,1.0-1e-12);

    return u;
}
