#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
#define HX %(hx)d
#define HY %(hy)d
//#define BLOCK_SIZE %(block_size)d

__constant unsigned int bg[3] = {NY*NZ,NZ,1};
__constant float TINY = 1e-40;

float Abs(float X)
{
    return X >= 0.f ? X : -X;
}

#define lu(i,j) A[idx + (i*HY + j)*NX*NY*NZ]
#define b(i) B[idx + i*NX*NY*NZ]
#define x(i) X[idx + i*NX*NY*NZ]

__kernel //__attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)))
void Solve(__global float* A, __global float* B, __global float* X, float tsh)
{  
    unsigned int ix[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ix[0]*bg[0]+ix[1]*bg[1]+ix[2]*bg[2];
    
    float indx[HX];
    float vv[HX];
    float big, temp, d = 1.f;
    
    unsigned j,k,ip;
    int i;
    
    
    for (i=0; i<HX; i++) {
        big = 0.f;
        for (j=0; j<HX; j++)
            if ((temp=Abs(lu(i,j))) > big) big = temp;
        if (big == 0.f)
            big = 1.f;
        vv[i] = 1.f/big;
    }
    
    for (k=0; k<HX; k++) {
        big = 0.f;
        for (i=k; i<HX; i++) {
            temp = vv[i]*Abs(lu(i,k));
            if (temp > big) {
                big = temp;
                ip = i;
            }
        }
        if (k != ip) {
            for (j=0; j<HX; j++) {
                temp = lu(ip,j);
                lu(ip,j) = lu(k,j);
                lu(k,j) = temp;
            }
            d = -d;
            vv[ip] = vv[k];
        }
        indx[k] = ip;
        if (lu(k,k) == 0.f) lu(k,k) = TINY;
        for (i = k+1; i<HX; i++) {
            temp = (lu(i,k) /= lu(k,k));
            for (j=k+1; j<HX; j++)
                lu(i,j) -= temp*lu(k,j);
        }
    }
    
    k = 0;
    for (i=0; i<HX; i++) x(i) = b(i);
    
    for (i=0; i<HX; i++) {
        ip = indx[i];
        temp = x(ip);
        x(ip) = x(i);
        if (k != 0)
            for (j=k-1; j<i; j++) temp -= lu(i,j)*x(j);
        else if (temp != 0.f)
            k = i+1;
        x(i) = temp;
    }
    
    for (i=HX-1; i>=0; i--) {
        temp = x(i);
        for (j=i+1; j<HX; j++) temp -= lu(i,j)*x(j);
        x(i) = temp/lu(i,i);
    }
    
}
             
    
        

                     
                 
                 
    
