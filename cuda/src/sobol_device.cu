#include <math.h>

#define MAX_DIM 512

__constant__ unsigned int SOBOL_DIR[MAX_DIM][32];

__device__
double sobol_sample(unsigned int index, int dim)
{
    unsigned int g = index ^ (index >> 1);

    unsigned int x = 0;

    for(int b=0;b<32;b++)
    {
        if(g & (1u << b))
            x ^= SOBOL_DIR[dim][b];
    }

    return x * 2.3283064365386963e-10;
}

__device__
double inverse_normal(double u)
{
    return sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * u);
}
