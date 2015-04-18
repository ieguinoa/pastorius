

/** **************************************************************** **/

// THIS FUNCTION CALCULATES FORCES(LJ DERIVATIVE) USING cpu AND A TABLE OF POTENTIALS STORED IN SYSTEMS MEMORY

void potentialsMode_cpu(float* LJPot,double* dEr, double* r, double cut, int* item_to_type, int num_samples_r, int num_types, int width, int height, int x, int y)
{
      /* Elemento de la matriz a calcular */
      //unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;   /** particula 2 **/
      //unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;   /** particula 1 **/

      /* Dentro del bloque correspondiente */
      //if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {dEr[y*width+x] = 0; return;}


      /** type of particles **/
      int t_o_p_1 =  item_to_type[y] * num_types;     //this one decides which subMatrix to use
      int t_o_p_2 =  item_to_type[x] + t_o_p_1;  //this one decides which row on these
      int posInicial = t_o_p_2 * num_samples_r;
      /** Convierto r a subíndice de matriz de lennard-jones **/
     // float index_x = (float)((double) (r[y*width+x] - MIn) * (double) num_samples_r / DIST + 0.5);    // convert  r  to   x

      // int index=0;
      double erre=r[y*width+x];
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
      	
      double E_r_up = (double) LJPot[indexsup];
      double E_r_dwn = (double) LJPot[indexinf];

      double r_dif = DIST * 2 * (DIF_FINITAS_DELTA) / num_samples_r;

      dEr[y*width+x] = (E_r_up - E_r_dwn) / (r_dif);
}





// THIS KERNEL CALCULATES FORCES(LJ DERIVATIVE) USING A TABLE OF ***DERIVATIVE OF POTENTIALS**** STORED IN DEVICEs GLOBAL MEMORY
 
void derivativeMode_cpu(float* dLJPot,double* dEr, double* r, double cut,int* item_to_type,
		    int num_samples_r, int num_types, int width, int height, int x, int y)
{
      /* Elemento de la matriz a calcular */
      //unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	/** particula 2 **/
      //unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	/** particula 1 **/
      
      
      /* Dentro del bloque correspondiente */
      //if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {dEr[y*width+x] = 0; return;}
      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */
      
      /** type of particles **/
      //float t_o_p_1 = (float) item_to_type[y] * num_types;	//this one decides which subMatrix to use
      //float t_o_p_2 = (float) item_to_type[x] + 0.5 + t_o_p_1;	//this one decides which row on these
      
      
      int t_o_p_1 = item_to_type[y] * num_types;	//this one decides which subMatrix to use
      int t_o_p_2 =  item_to_type[x] + t_o_p_1;	//this one decides which row on these
      int posInicial=t_o_p_2 * num_samples_r;   //comienzo de la fila??
      /** Convierto r a subíndice de matriz de lennard-jones **/
      /** r = (MAX-MIN) * X / N  + MIN    **/
      /** x = (r-MIN) * N / (MAX-MIN)     **/
      //float index_x = (float)((double) (r[y*width+x] - MIn) * (double) num_samples_r / DIST + 0.5);	// convert  r  to   x
      //int index=0;
      double erre=r[y*width+x];
      if(erre > MAx)
	dEr[y*width+x] = dLJPot[posInicial + num_samples_r - 1];
      else 
	if(erre<MIn)
	  dEr[y*width+x]=dLJPot[posInicial];
	else{
	  int sesgo=(erre-MIn)*(num_samples_r/DIST);
	  dEr[y*width +x]=dLJPot[posInicial + sesgo];
	}  //dEr[y*width+x] = (double) tex2D( texRef, index_x, t_o_p_2 );
}



// THIS FUNCTION CALCULATES FORCES(dPot) BY SOLVING THE EQUATION IN cpu
void analyticMode_cpu(double* dEr, double* r, double cut, int* item_to_type, int num_samples_r,
                    double* EPS, double* SIG, int width, int height, int x, int y)
{
      /* Elemento de la matriz a calcular */
      //unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;   /** particula 2 **/
      //unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;   /** particula 1 **/

      /* Dentro del bloque correspondiente */
      //if(x >= width || y >= height) {return;}
      if(x == y || r[y*width+x] >= cut) {dEr[y*width+x] = 0; return;}

      /* valor del Potencial para la distancia r,
       *  para el tipo de partícula correspondiente */

      /** type of particle 2 **/
      int type_i = item_to_type[x];
      int type_j = item_to_type[y];

      double sig12 = (double) (SIG[type_i] + SIG[type_j])/2;
      double eps12 = (double) sqrt(EPS[type_i] * EPS[type_j]);

      dEr[y*width+x] = (double) 24.0*eps12*( pow(sig12,6)/ pow(r[y*width+x],7) - 2 * pow(sig12,12)/ pow(r[y*width+x],13));
}




/* ***************************************************************** **/


