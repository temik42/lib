#define NX %(nx)d
#define NY %(ny)d
#define NZ %(nz)d
#define HX %(hx)d
#define HY %(hy)d
//#define BLOCK_SIZE %(block_size)d

__constant unsigned int bg[3] = {NY*NZ,NZ,1};
__constant float eps = 1e-10;

float Abs(float X)
{
    return X >= 0.f ? X : -X;
}

float Sign(float a, float b)
{
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

float Pythag(float a, float b) 
{
    float absa=Abs(a), absb=Abs(b);
    return (absa > absb ? absa*sqrt(1.f+(absb/absa)*(absb/absa)) :
    (absb == 0.f ? 0.f : absb*sqrt(1.f+(absa/absb)*(absa/absb))));
}


#define u(i,j) A[idx + (i*HY + j)*NX*NY*NZ]
#define b(i) B[idx + i*NX*NY*NZ]
#define x(i) X[idx + i*NX*NY*NZ]

__kernel //__attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)))
void Solve(__global float* A, __global float* B, __global float* X, float tsh)
{  
    unsigned int ix[3] = {get_global_id(0),get_global_id(1),get_global_id(2)};
    unsigned int idx = ix[0]*bg[0]+ix[1]*bg[1]+ix[2]*bg[2];


    bool flag;
    int i,its,j,jj,k,l,nm;
    float anorm,c,f,g,h,s,scale,x,y,z;
    float rv1[HY], w[HY];
    g = scale = anorm = 0.f; 
    float v[HX][HY];
    
    for (i=0;i<HY;i++) {
    l=i+2;
    rv1[i]=scale*g;
    g=s=scale=0.f;
    if (i < HX) {
    for (k=i;k<HX;k++) scale += Abs(u(k,i));
    if (scale != 0.f) {
    for (k=i;k<HX;k++) {
    u(k,i) /= scale;
    s += u(k,i)*u(k,i);
    }
    f=u(i,i);
    g = -Sign(sqrt(s),f);
    h=f*g-s;
    u(i,i)=f-g;
    for (j=l-1;j<HY;j++) {
    for (s=0.f,k=i;k<HX;k++) s += u(k,i)*u(k,j);
    f=s/h;
    for (k=i;k<HX;k++) u(k,j) += f*u(k,i);
    }
    for (k=i;k<HX;k++) u(k,i) *= scale;
    }
    }
    w[i]=scale *g;
    g=s=scale=0.f;
    if (i+1 <= HX && i+1 != HY) {
    for (k=l-1;k<HY;k++) scale += Abs(u(i,k));
    if (scale != 0.f) {
    for (k=l-1;k<HY;k++) {
    u(i,k) /= scale;
    s += u(i,k)*u(i,k);
    }
    f=u(i,l-1);
    g = -Sign(sqrt(s),f);
    h=f*g-s;
    u(i,l-1)=f-g;
    for (k=l-1;k<HY;k++) rv1[k]=u(i,k)/h;
    for (j=l-1;j<HX;j++) {
    for (s=0.f,k=l-1;k<HY;k++) s += u(j,k)*u(i,k);
    for (k=l-1;k<HY;k++) u(j,k) += s*rv1[k];
    }
    for (k=l-1;k<HY;k++) u(i,k) *= scale;
    }
    }
    anorm=max(anorm,(Abs(w[i])+Abs(rv1[i])));
    }
    for (i=HY-1;i>=0;i--) {
    if (i < HY-1) {
    if (g != 0.f) {
    for (j=l;j<HY;j++)
    v[j][i]=(u(i,j)/u(i,l))/g;
    for (j=l;j<HY;j++) {
    for (s=0.0,k=l;k<HY;k++) s += u(i,k)*v[k][j];
    for (k=l;k<HY;k++) v[k][j] += s*v[k][i];
    }
    }
    for (j=l;j<HY;j++) v[i][j]=v[j][i]=0.f;
    }
    v[i][i]=1.f;
    g=rv1[i];
    l=i;
    }
    for (i=min(HX,HY)-1;i>=0;i--) {
    l=i+1;
    g=w[i];
    for (j=l;j<HY;j++) u(i,j)=0.f;
    if (g != 0.f) {
    g=1.f/g;
    for (j=l;j<HY;j++) {
    for (s=0.f,k=l;k<HX;k++) s += u(k,i)*u(k,j);
    f=(s/u(i,i))*g;
    for (k=i;k<HX;k++) u(k,j) += f*u(k,i);
    }
    for (j=i;j<HX;j++) u(j,i) *= g;
    } else for (j=i;j<HX;j++) u(j,i)=0.f;
    u(i,i)+= 1.f;
    }
    for (k=HY-1;k>=0;k--) {
    for (its=0;its<30;its++) {
    flag=true;
    for (l=k;l>=0;l--) {
    nm=l-1;
    if (l == 0 || Abs(rv1[l]) <= eps*anorm) {
    flag=false;
    break;
    }
    if (Abs(w[nm]) <= eps*anorm) break;
    }
    if (flag) {
    c=0.f; 
    s=1.f;
    for (i=l;i<k+1;i++) {
    f=s*rv1[i];
    rv1[i]=c*rv1[i];
    if (Abs(f) <= eps*anorm) break;
    g=w[i];
    h=Pythag(f,g);
    w[i]=h;
    h=1.f/h;
    c=g*h;
    s = -f*h;
    for (j=0;j<HX;j++) {
    y=u(j,nm);
    z=u(j,i);
    u(j,nm)=y*c+z*s;
    u(j,i)=z*c-y*s;
    }
    }
    }
    z=w[k];
    if (l == k) {
    if (z < 0.f) {
    w[k] = -z;
    for (j=0;j<HY;j++) v[j][k] = -v[j][k];
    }
    break;
    }
    //if (its == 29) throw("no convergence in 30 svdcmp iterations");
    x=w[l];
    nm=k-1;
    y=w[nm];
    g=rv1[nm];
    h=rv1[k];
    f=((y-z)*(y+z)+(g-h)*(g+h))/(2.f*h*y);
    g=Pythag(f,1.f);
    f=((x-z)*(x+z)+h*((y/(f+Sign(g,f)))-h))/x;
    c=s=1.f;
    for (j=l;j<=nm;j++) {
    i=j+1;
    g=rv1[i];
    y=w[i];
    h=s*g;
    g=c*g;
    z=Pythag(f,h);
    rv1[j]=z;
    c=f/z;
    s=h/z;
    f=x*c+g*s;
    g=g*c-x*s;
    h=y*s;
    y *= c;
    for (jj=0;jj<HY;jj++) {
    x=v[jj][j];
    z=v[jj][i];
    v[jj][j]=x*c+z*s;
    v[jj][i]=z*c-x*s;
    }
    z=Pythag(f,h);
    w[j]=z;
    if (z) {
    z=1.f/z;
    c=f*z;
    s=h*z;
    }
    f=c*g+s*y;
    x=c*y-s*g;
    for (jj=0;jj<HX;jj++) {
    y=u(jj,j);
    z=u(jj,i);
    u(jj,j)=y*c+z*s;
    u(jj,i)=z*c-y*s;
    }
    }
    rv1[l]=0.f;
    rv1[k]=f;
    w[k]=x;
    }
    }
    
    //float tsh = 0.5*sqrt((float)(HX+HY+1))*w[0]*eps;
    //float tsh = 1.f;
    
    for (j=0; j<HY; j++) {
        s = 0.f;
        if (w[j] > tsh) {
            for (i=0; i<HX; i++) s += u(i,j)*b(i);
            s /= w[j];
        }
        rv1[j] = s;
    }
    
    for (j=0; j<HY; j++) {
        s = 0.f;
        for (jj=0; jj<HY; jj++) s += v[j][jj]*rv1[jj];
        x(j) = s;
    }
                
    
}