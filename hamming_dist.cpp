#if !defined(_WIN32)
#define dgemm dgemm_
#endif
#include "mex.h"
#include "string.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    unsigned int n1 = mxGetM(prhs[0]);
    unsigned int* B1 = (unsigned int*)mxGetPr(prhs[0]);
    unsigned int n2 = mxGetM(prhs[1]);
    unsigned int* B2 = (unsigned int*)mxGetPr(prhs[1]);
    unsigned int nwords = mxGetN(prhs[1]);
    plhs[0] = mxCreateDoubleMatrix(n1, n2, mxREAL);
	double *Dh = (double*)mxGetPr(plhs[0]);
    //mexPrintf("%d,%d,%d,%d\n",n1,n2,B1[0],B2[0]);
    for(int i=0;i<n1;++i)
    {
        for(int j=0;j<n2;++j)
        {
            Dh[i+j*n1] = 0;
            for(int n=0;n<nwords;++n)
            {   
                unsigned int y = B1[i+n*n1] ^ B2[j+n*n2];
                Dh[i+j*n1] += __builtin_popcount(y);
                //mexPrintf("%d %d %d\n",y, B1[i+n*n1], B2[j+n*n2]); 
            }
            
        }
            
    }

}