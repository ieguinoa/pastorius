
/**
 *		 ******************************************************
 *		 ***********   FILL LJ POTENTIALS TABLE   ************* 
 *		 ******************************************************
 
 *    CALCULATE LJ_POTENTIAL OF PARTICLE (WITH SIGMA=s AND EPSILON=e) INTERACTION WITH OTHER PARTICLES ***FOR A SPECIFIC r VALUE***
 *    THE OTHER PARTICLE TYPE DEPENDES ON Thread
 *    EPS and SIG CONTAINS VALUES OF THE OTHER EPSILONS AND SIGMAS 
 *    
 */

__global__ void lennard_Kernel(float* LJ_POT, double* EPS, double* SIG, double e, double s, double var, int width, int height)
{
     /* x determines the assigned value of r */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    /*y value determines the type of the other particle(position in EPS and SIG arrays */
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(x >= width || y >= height) {return;}
      
    /* Get sig and epsilon values to use in LJ calculation(avarage of both types values) */ 
      double sig12 = (double) (s + SIG[y])/2;
      double eps12 = (double) sqrt(e * EPS[y]);
    /* get assigned r (based on thread x) */
      double r = (double) MIn+x*var;
      
      /* Calc. and save result */
      LJ_POT[y*width +x] = (float) 4.0*eps12*( pow((sig12/r),12) - pow((sig12/r),6));
}








/** **************************************************************** **/







  /**		************************************************
  * 		********    FILL DERIVATIVES TABLE   ***********
  *		************************************************ 
 
 
 *    CALCULATE ***DERIVATIVE**  OF LJ_POTENTIAL OF PARTICLE (WITH SIGMA=s AND EPSILON=e) INTERACTION WITH OTHER PARTICLES ***FOR A SPECIFIC r VALUE***
 *    THE OTHER PARTICLE TYPE DEPENDES ON Thread
 *    EPS and SIG CONTAINS VALUES OF THE OTHER EPSILONS AND SIGMAS 
 */

__global__ void derivatives_lennard_Kernel(float* dLJ_POT, double* EPS, double* SIG, double e, double s, double var, int width, int height)
{
    /* x determines the assigned value of r */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      
    /*y value determines the type of the other particle(position in EPS and SIG arrays */
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(x >= width || y >= height) {return;}
      
    /* Get sig and epsilon values to use in LJ calculation(avarage of both types values) */ 
      double sig12 = (double) (s + SIG[y])/2;
      double eps12 = (double) sqrt(e * EPS[y]);
    /* get assigned r (based on thread x) */
      double r = (double) MIn+x*var;
      
    /* Calc. and save result */
      dLJ_POT[y*width +x] = (float) 24.0*eps12*( pow(sig12,6)/ pow(r,7) - 2 * pow(sig12,12)/ pow(r,13));
}
      
      
      
      
      
/** **************************************************************** **/





/*		**************************************************************************
 *		********   DISTANCE BETWEEN CLOSEST IMAGE  OF ANY PARTICLE **************
 *		*************************************************************************
 */

//  THIS KERNEL CALCULATES THE CLOSEST IMAGE BETWEEN TWO PARTICLES IN A SIMULATION USING PERIODIC BOUNDARIES 
__global__ void close_distances_kernel(double* X, double* Y, double* Z, double* R, double* position_x, double* position_y, double* position_z, double box_x, double box_y, double box_z, int width, int height)
{
    /*   i,j determines the two particles particle to consider*/
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(i >= width || j >= height) {return;}
//       unsigned int pos = j*width+i;
      
      double _X = position_x[i] - position_x[j];
      double _Y = position_y[i] - position_y[j];
      double _Z = position_z[i] - position_z[j];
      
      _X = _X - box_x * round((double) _X/box_x);
      _Y = _Y - box_y * round((double) _Y/box_y);
      _Z = _Z - box_z * round((double) _Z/box_z);
      X[pos] = _X;
      Y[pos] = _Y;
      Z[pos] = _Z;
      R[pos] = (double) sqrt( _X*_X + _Y*_Y + _Z*_Z );
}







/** **************************************************************** **/









/*		***************************************************
 *		********   DISTANCE BETWEEN PARTICLES *************
 *		*********---  NON Periodic  --------***************   
 *		**************************************************
 */

__global__ void distances_kernel(double* R, double* X, double* Y, double* Z, double* x1, double* y1, double* z1, int width, int height)
{
//   PARTICLES INDEX
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(x >= width || y >= height) {return;}
      
      double x_ = x1[x] - x1[y];
      double y_ = y1[x] - y1[y];
      double z_ = z1[x] - z1[y];
      X[y*width+x] = x_;
      Y[y*width+x] = y_;
      Z[y*width+x] = z_;
      R[y*width+x] = (double) sqrt( x_*x_ + y_*y_ + z_*z_ );
}





/***************************************************************************/








/*		***************************************************
 *		*********** DERIVATIVES CALCULATION ****************
 *		**************************************************
 *		**************************************************
 */






 
 
/**********************************************************************
********   POTENTIALS-MODE CALCULATION ------ GLOBAL MEMORY **************/

__global__ void potentialsMode_memory_kernel(float* LJPot,double* dEr, double* r, double cut, int* item_to_type, int num_samples_r, int num_types, int width, int height )
{
     /*GET INDEX OF ELEMENT*/
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;   /** partic 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;   /** partic 1 **/
     /*   r VALUE  */ 
      double erre=r[y*width+x];
      double result;
      
      if(x >= width || y >= height) {return;}
     /* IF I AM TRYING TO CALCULATE POTENTIAL AGAINS ITSELF OR BETWEEN OUT OF RANGE PARTICLES POT=0 -> dPot=0*/
     if(x == y || erre >= cut) {
	//dEr[y*width+x] = 0; return;
	result=0;
	}
	else{

    /** type of particles **/
      int t_o_p_1 =  item_to_type[y] * num_types;     //this one decides which subMatrix to use
      int t_o_p_2 =  item_to_type[x] + t_o_p_1;  //this one decides which row on these
    
      int posInicial = t_o_p_2 * num_samples_r;
      /** Convierto r a subíndice de matriz de lennard-jones **/
     // float index_x = (float)((double) (r[y*width+x] - MIn) * (double) num_samples_r / DIST + 0.5);    // convert  r  to   x

      // int index=0;
      double superior=erre + (DIF_FINITAS_DELTA*DIST/num_samples_r);
      double inferior=erre - (DIF_FINITAS_DELTA*DIST/num_samples_r);
      int indexsup=posInicial + ((superior-MIn)*(num_samples_r/DIST));
      int indexinf=posInicial + ((inferior-MIn)*(num_samples_r/DIST));

      if(superior > MAx)
        indexsup=posInicial + num_samples_r - 1;
      if(superior<MIn)
        indexsup=posInicial;
      if(inferior<MIn)
        indexinf=posInicial;
      if(inferior>MAx)
        indexinf=posInicial + num_samples_r - 1;

      /* Get value in higher position*/
      double E_r_up = (double) LJPot[indexsup];
      /*Get value in lower position */
      double E_r_dwn = (double) LJPot[indexinf];

      /*CALCULATE DISCRETE DERIVATIVE*/
      double r_dif = DIST * 2 * (DIF_FINITAS_DELTA) / num_samples_r;
	result = (E_r_up - E_r_dwn) / (r_dif);
	}

      /*SAVE RESULT*/
	dEr[y*width+x]=result;
}






/*************************************************************************/









/*		***************************************************
 *		********   POTENTIALS-MODE  ----- TEXTURE *********
 *		**************************************************
 */

// THIS KERNEL CALCULATES FORCES(LJ DERIVATIVE) USING A TABLE OF POTENTIALS STORED IN DEVICEs TEXTURE MEMORY

__global__ void potentialsMode_texture_kernel(double* dEr, double* r, double cut, int* item_to_type, int num_samples_r, int num_types, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      double erre= r[y*width+x]; 
	double result;
      /* Dentro del bloque correspondiente */
      if(x >= width || y >= height) {return;}
      if(x == y || erre  >= cut) {
	//dEr[y*width+x] = 0; return;
	result=0;
	}
      
      else{
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particles **/
      float t_o_p_1 = (float) item_to_type[y] * num_types;	//this one decides which subMatrix to use
      float t_o_p_2 = (float) item_to_type[x] + 0.5 + t_o_p_1;	//this one decides which row on these
      
      /** Convierto r a subíndice de matriz de lennard-jones **/
      /** r = (MAX-MIN) * X / N  + MIN    **/
      /** x = (r-MIN) * N / (MAX-MIN)     **/
      float index_x = (float)((double) (erre - MIn) * (double) num_samples_r / DIST + 0.5);	// convert  r  to   x
      /*
	double rposta=r[y*width+x];
      if(rposta> MAx)
        rposta=MAx;
        else
                if(rposta<MIn)
                        rposta=MIn;

   float index_x = (float)((double) (rposta - MIn) * (double) num_samples_r / DIST + 0.5);  // convert  r  to   x      
*/

      double E_r_up = (double) tex2D( texRef, index_x + DIF_FINITAS_DELTA, t_o_p_2 );
      double E_r_dwn = (double) tex2D( texRef, index_x - DIF_FINITAS_DELTA, t_o_p_2 );
      
      
      double r_dif = DIST * 2 * (DIF_FINITAS_DELTA) / num_samples_r;
       result = (E_r_up - E_r_dwn) / (r_dif); 
        }
        dEr[y*width+x]= result;
     
      //dEr[y*width+x] = (E_r_up - E_r_dwn) / (r_dif);
}




/** **************************************************************** **/









/*		************************************************************************
 *              *******   DERIVATIVE-MODE CALCULATION ----- GLOBAL MEMORY **************
 *		************************************************************************
 */

// THIS KERNEL CALCULATES FORCES(LJ DERIVATIVE) USING A TABLE OF ***DERIVATIVE OF POTENTIALS**** STORED IN DEVICEs GLOBAL MEMORY

__global__ void derivativeMode_memory_kernel(float* dLJPot,double* dEr, double* r, double cut,int* item_to_type,
                    int num_samples_r, int num_types, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; /** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; /** particula 1 **/

	double erre=r[y*width+x];
	double result;
      /* Dentro del bloque correspondiente */
      if(x >= width || y >= height) {return;}
      if(x == y || erre  >= cut) {
	//dEr[y*width+x] = 0; return;
	result=0;
	}
	else{
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */

      /** type of particles **/
      //float t_o_p_1 = (float) item_to_type[y] * num_types;   //this one decides which subMatrix to use
      //float t_o_p_2 = (float) item_to_type[x] + 0.5 + t_o_p_1;        //this one decides which row on these


      int t_o_p_1 = item_to_type[y] * num_types;       //this one decides which subMatrix to use
      int t_o_p_2 =  item_to_type[x] + t_o_p_1; //this one decides which row on these
      int posInicial=t_o_p_2 * num_samples_r;   //comienzo de la fila??
      /** Convierto r a subíndice de matriz de lennard-jones **/
      /** r = (MAX-MIN) * X / N  + MIN    **/
      /** x = (r-MIN) * N / (MAX-MIN)     **/
      //float index_x = (float)((double) (r[y*width+x] - MIn) * (double) num_samples_r / DIST + 0.5);  // convert  r  to   x
      //int index=0;
      int posMax=num_samples_r -2;
      
      float sesgo=(erre-MIn) *(num_samples_r/DIST);
	if(sesgo>posMax)
		result = dLJPot[posInicial+ posMax];
	else
		if(sesgo<0)
			result = dLJPot[posInicial];
		else
			result = dLJPot[posInicial+(int)ceil(sesgo)];	
	}
       dEr[y*width+x]=result;

/*
      if(erre > MAx)
        dEr[y*width+x] = dLJPot[posInicial + num_samples_r - 1];
      else
        if(erre<MIn)
          dEr[y*width+x]=dLJPot[posInicial];
        else{
          int sesgoSup=ceil((erre-MIn)*(num_samples_r/DIST));
          int sesgoInf= floor((erre-MIn)*(num_samples_r/DIST));
          float y1=dLJPot[posInicial + sesgoSup];
          float y0=dLJPot[posInicial + sesgoInf];
	  double x0=sesgoInf*num_samples_r /DIST;
	  double a = (y1 - y0) / (1);
  	  double b = -a*x0 + y0;
   	  double ybuscado = a *((erre-MIn)*(num_samples_r/DIST))  + b;
	  dEr[y*width +x]= ybuscado;
		//dEr[y*width +x]=dLJPot[posInicial + sesgoSup];
        } 
	
*/
	 //dEr[y*width+x] = (double) tex2D( texRef, index_x, t_o_p_2 );
}






// *****************************************************************************************








/*		************************************************************************
 *              *******   DERIVATIVE-MODE CALCULATION ----- TEXTURE ********************
 *		************************************************************************
 */
 
/**********************************************************************
********   DERIVATIVE-MODE CALCULATION ---------- TEXTURE **************/

// THIS KERNEL CALCULATES FORCES(LJ DERIVATIVE) USING A TABLE OF ***DERIVATIVE OF POTENTIALS**** STORED IN DEVICEs TEXTURE MEMORY(DIRECT FETCH)
 
__global__ void direct_derivativeMode_E_r(double* dEr, double* r, double cut, int* item_to_type, int num_samples_r, int num_types, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      double result;
      /* Dentro del bloque correspondiente */
      double erre= r[y*width+x];
      if(x >= width || y >= height) {return;}
      if(x == y || erre >= cut) {
	result=0;
	//dEr[y*width+x] = 0; 
	//return;
	}
      else{
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particles **/
      float t_o_p_1 = (float) item_to_type[y] * num_types;	//this one decides which subMatrix to use
      float t_o_p_2 = (float) item_to_type[x] + 0.5 + t_o_p_1;	//this one decides which row on these
      
      /** Convierto r a subíndice de matriz de lennard-jones **/
      /** r = (MAX-MIN) * X / N  + MIN    **/
      /** x = (r-MIN) * N / (MAX-MIN)     **/
     float index_x = (float)((double) (erre - MIn) * (double) num_samples_r / DIST + 0.5);	// convert  r  to   x
     /* double rposta=r[y*width+x];
      if(rposta> MAx)
	rposta=MAx;
	else
		if(rposta<MIn)
			rposta=MIn;
   
   float index_x = (float)((double) (rposta - MIn) * (double) num_samples_r / DIST + 0.5);  // convert  r  to   x	
*/	

      result = (double) tex2D( texRef, index_x, t_o_p_2 );
	}
	dEr[y*width+x]= result;

}




/* ***************************************************************** **/










/*		************************************************************************
 *              *****************   ANALYTIC-MODE CALCULATION **************************
 *		************************************************************************
 */

/***************************************************************
***********  AnalyticMode CALCULATION *************************/


__global__ void analyticMode_kernel(double* dEr, double* r, double cut, int* item_to_type, int num_samples_r, double* EPS, double* SIG, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      /* Dentro del bloque correspondiente */
       double erre=r[y*width+x];
      double result; 
	 if(x >= width || y >= height) {return;}
  
    if(x == y || erre  >= cut) {
        //dEr[y*width+x] = 0; return;
         result=0;
         }
        else{
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particle 2 **/
      int type_i = item_to_type[x];
      int type_j = item_to_type[y];
      
      double sig12 = (double) (SIG[type_i] + SIG[type_j])/2;
      double eps12 = (double) sqrt(EPS[type_i] * EPS[type_j]);
      
      result = (double) 24.0*eps12*( pow(sig12,6)/ pow(erre,7) - 2 * pow(sig12,12)/ pow(erre,13));
	}	
      dEr[y*width+x]=result;

}






















/*		***************************************************
 *		*********** POTENTIALS CALCULATION ****************
 *		***************************************************
 *		***************************************************
 */


 
 













/*		************************************************************************
 *              *****************   ANALYTIC-MODE CALCULATION **************************
 *		************************************************************************
 */

__global__ void potential_analytic_kernel(double* Er, double* r, double cut, int* item_to_type, int num_samples_r,
	       double* EPS, double* SIG, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      /* Dentro del bloque correspondiente */
      if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {Er[y*width+x] = 0; return;}
      
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particle 2 **/
      int type_i = item_to_type[x];
      int type_j = item_to_type[y];
      
      double sig12 = (double) (SIG[type_i] + SIG[type_j])/2;
      double eps12 = (double) sqrt(EPS[type_i] * EPS[type_j]);
      
      Er[y*width+x] = (double) 4.0*eps12*( pow((sig12/r[y*width+x]),12) - pow((sig12/r[y*width+x]),6));
}










/** *********************************************************************************** **/



/*		************************************************************************
 *              *****************   POTENTIALS-MODE CALCULATION **************************
 *		************************************************************************
 */

// THIS KERNEL CALCULATES **POTENTIAL** VALUE FROM A TABLE IN TEXTURE MEMORY
__global__ void potentials_texture_kernel(double* Er, double* r, double cut, int* item_to_type,
	 int num_samples_r, int num_types, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      /* Dentro del bloque correspondiente */
      if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {Er[y*width+x] = 0; return;}
      
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particles **/
      float t_o_p_1 = (float) item_to_type[y];		//this one decides which subMatrix to use
      float t_o_p_2 = (float) item_to_type[x];		//this one decides which row on these
      float row =  t_o_p_2 + 0.5 + (t_o_p_1* num_types); 
      /** Convierto r a subíndice de matriz de lennard-jones **/
      /** r = (MAX-MIN) * X / N  + MIN **/
      /** x = (r-MIN) * N / (MAX-MIN)     **/
      float index_x = (float)((double) (r[y*width+x] - MIn) * (double) num_samples_r / DIST + 0.5);	// convert  r  to   x

/*
	 double rposta=r[y*width+x];
      if(rposta> MAx)
        rposta=MAx;
        else
                if(rposta<MIn)
                        rposta=MIn;

   float index_x = (float)((double) (rposta - MIn) * (double) num_samples_r / DIST + 0.5);  // convert  r  to   x      
    */  


      Er[y*width+x] = (double) tex2D( texRef, index_x, row );
}




/** **************************************************************** **/
