# Affine transformation 
- for parallelization 
- improves locality 


## Some techniques 
- **Loop fusion**
- **Fission**
- **Re-indexing**
- **Scaling**


- Following loop can be transformed as 

        for( int i = 0 p; i <= N ; i++ ){
            y[i] = z[i]; 
            x[i] = z[i-1]
        }

- After transformation 

        if( N>= 1 ) x[1] = y[0]; 
        for( int i = 1; i < N ;i++ ){
            y[i] = z[i]; 
            x[i+1] = y[i]; 
        }
        if( N>= 1) x[N] = y[N-1]; 


## you can't parallelize  loop where we have iterative dependence 