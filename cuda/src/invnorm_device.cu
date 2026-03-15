#include <math.h>

__device__
double inverse_normal(double p)
{
    const double a1=-39.69683028665376;
    const double a2=220.9460984245205;
    const double a3=-275.9285104469687;
    const double a4=138.3577518672690;
    const double a5=-30.66479806614716;
    const double a6=2.506628277459239;

    const double b1=-54.47609879822406;
    const double b2=161.5858368580409;
    const double b3=-155.6989798598866;
    const double b4=66.80131188771972;
    const double b5=-13.28068155288572;

    const double c1=-0.007784894002430293;
    const double c2=-0.3223964580411365;
    const double c3=-2.400758277161838;
    const double c4=-2.549732539343734;
    const double c5=4.374664141464968;
    const double c6=2.938163982698783;

    const double d1=0.007784695709041462;
    const double d2=0.3224671290700398;
    const double d3=2.445134137142996;
    const double d4=3.754408661907416;

    double q,r;

    if(p < 0.02425)
    {
        q = sqrt(-2*log(p));
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
               ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    }

    if(p > 1-0.02425)
    {
        q = sqrt(-2*log(1-p));
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                 ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    }

    q = p-0.5;
    r = q*q;

    return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
           (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
}
