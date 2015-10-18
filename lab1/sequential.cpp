#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    //clock_t start = clock();
    
    int n=10000;
    int num = 500;//iteration
    float h[(n+1)*(n+1)];
    float k[(n+1)*(n+1)];
    int i,j;
    for(i=0;i<(n+1)*(n+1);i++){
        h[i]=0.0;
        k[i]=0.0;
    }
    for(i=0;i<=n;i++){
        h[i]=80.0;
        k[i]=80.0;
    }
    for(i=1;i<=n;i++){
        h[i*(n+1)]=80.0;
        k[i*(n+1)]=80.0;

    }
    for(i=1;i<=n+1;i++){
        h[i*(n+1)-1]=80.0;
        k[i*(n+1)-1]=80.0;
    }
    for (i=n*(n+1); i<=(n+1)*(n+1)-1;i++){
        h[i]=80.0;
        k[i]=80.0;
    }
    if(n>=30){
    for(i=10;i<=30;i++){
        h[i]=150.0;
        k[i]=150.0;
    }
    }
    else if (n>=10&&n<30){
        for(i=10;i<=n;i++){
            h[i]=150.0;
            k[i]=150.0;
        }
    }
    
     for (i=n+2;i<n*(n+1)-1;i++){
        k[i]= (h[i-1]+h[i+1]+h[i-n-1]+h[i+n+1])/4;
        
    }

    for(i=0;i<=n;i++){
        k[i]=80.0;
    }
    for(i=1;i<=n;i++){
        k[i*(n+1)]=80.0;
        
    }
    for(i=1;i<=n+1;i++){
        k[i*(n+1)-1]=80.0;
    }
    for (i=n*(n+1); i<=(n+1)*(n+1)-1;i++){
        k[i]=80.0;
    }
    if(n>=30){
        for(i=10;i<=30;i++){
            k[i]=150.0;
        }
    }
    else if (n>=10&&n<30){
        for(i=10;i<=n;i++){
            k[i]=150.0;
        }
    }
    
    for(j=2;j<=num;j++){
        for(int i=0;i<(n+1)*(n+1);i++){
        h[i]=k[i];
       }
        for (i=n+2;i<n*(n+1)-1;i++){
            k[i]= (h[i-1]+h[i+1]+h[i-n-1]+h[i+n+1])/4;
            
        }
        for(i=0;i<=n;i++){
            k[i]=80.0;
        }
        for(i=1;i<=n;i++){
            k[i*(n+1)]=80.0;
            
        }
        for(i=1;i<=n+1;i++){
            k[i*(n+1)-1]=80.0;
        }
        for (i=n*(n+1); i<=(n+1)*(n+1)-1;i++){
            k[i]=80.0;
        }
        if(n>=30){
            for(i=10;i<=30;i++){
                k[i]=150.0;
            }
        }
        else if (n>=10&&n<30){
            for(i=10;i<=n;i++){
                k[i]=150.0;
            }
        }
    }
    /*
    for(i=0;i<(n+1)*(n+1);i++){
        printf("%f ",k[i]);
        if(i%(n+1)==n)
        printf("\n");
    }
    
    clock_t end = (clock() - start)/1000;
    printf("%d * %d, time: %ldms\n", n+1,n+1, end);  */ }
