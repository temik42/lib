#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
#define HX %(hx)d
#define HY %(hy)d
//#define BLOCK_SIZE %(block_size)d

__constant unsigned int bg[3] = {NY*NZ,NZ,1};

#define el(i,j) A[idx + (i*HY + j)*NX*NY*NZ]
#define b(i) B[idx + i*NX*NY*NZ]
#define x(i) X[idx + i*NX*NY*NZ]

__kernel //__attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)))
void Solve(__global float* A, __global float* B, __global float* X, float tsh)
{  
    unsigned int ix[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ix[0]*bg[0]+ix[1]*bg[1]+ix[2]*bg[2];
    
    int i,j,k;
    float sum;
    
    for (i=0;i<HX;i++) {
        for (j=i;j<HX;j++) {
            for (sum=el(i,j),k=i-1;k>=0;k--) sum -= el(i,k)*el(j,k);
            if (i == j) {
                if (sum <= 0.f) 
                    sum = 1.f;
                el(i,i)=sqrt(sum);
            } else el(j,i)=sum/el(i,i);
        }
    }
    for (i=0;i<HX;i++) for (j=0;j<i;j++) el(j,i) = 0.f;
    
    
    for (i=0;i<HX;i++) { 
        for (sum=b(i),k=i-1;k>=0;k--) sum -= el(i,k)*x(k);
        x(i)=sum/el(i,i);
    }
    for (i=HX-1;i>=0;i--) { 
        for (sum=x(i),k=i+1;k<HX;k++) sum -= el(k,i)*x(k);
        x(i)=sum/el(i,i);
    }
    
    
}
             
    
        

                     
                 
                 
    
