
/* **************************************************************
 * THIS FILE CONTAINS ALL KERNEL FUNCTIONS ASOCIATED WITH MD EXECUTION:
	-FORCE CALCULATION
	-FORCE SUMS
	-KINETIC ENERGY CALCULATION
	-TOTAL KINETIC ENERGY CALCULATION
	-PERIODIC BOUNDARIES PARTICLE RELOCATION
	-CALCULATE FORCE ACTION ON VELOCITY
	-UPDATE VELOCITIES
	-UPDATE POSITIONS
 * *************************************************************/

 
 
 
 
 
 
 
 
 
 
 
 
 
 

/* **************************************************************
 * ******  CALCULATE FORCE BETWEEN 2 PARTICLES  *****************
 * *************************************************************/

  /*   Fx =  dE(r) / dr  *  (x1-x2) / r               */
__global__ void Parcial_Forces_Kernel(double* force, double* dEr, double* dif, double* r, int width, int height)
{
      /* Elemento de la matriz a calcular */
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
      
      if(x >= width || y >= height) {return;}
      if(x == y) {force[y*width+x] = 0; return;}
      
	//force[y*width+x] = dEr[y*width+x] *  dif[y*width+x] ;
 
      force[y*width+x] = dEr[y*width+x] * dif[y*width+x] / r[y*width+x];
}











/* **************************************************************
 * ******  SUM ALL FORCES ACTING ON A PARTICLE  *****************
 * *************************************************************/

__global__ void Resultant_Forces_Kernel(double* result, double* forces, int cant)
{
   
      unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(x >= cant) {return;}
      
      int i = 0;
      double tmp = 0;
      int row = x*cant;
      for(; i < cant; i++){
	 tmp += forces[row + i];
      }
      result[x] = tmp;
}











/* ***********************************************************************
 * *****************  CALCULATE NEW VELOCITIES  *************************
 * **********************************************************************/


/*  V(t + Dt/2) = V(t - Dt/2) +  [ F(t) * Dt ] / m  */  
__global__ void Resultant_Velocities_Kernel(double* velocity, double* old_velocity, double* force, double* m,
				 int* item_to_type, double delta_time, int cant_particles)
{      
      /* Elemento de la matriz a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i >= cant_particles) {return;}
      
      double Vt = old_velocity[i];
      int type = item_to_type[i];
      double dtx = delta_time*20.454999999999;
      //double dtx=delta_time;

	/* Result */
      velocity[i] = Vt + ( (force[i]*dtx) / m[type] );
}










/* *************************************************
 * **********  UPDATE VELOCITIES  ******************
 * ************************************************/

__global__ void Corrected_Velocities_Kernel(double* vold, double* v, double lambda, int cant){
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i>= cant) {return;}
      vold[i] = v[i];
      //vold[i] = v[i] * lambda;
}









/* **************************************************************
 * *****************  UPDATE POSITIONS  *************************
 * *************************************************************/
/*       P(t + Dt) = P(t) +  V(t + Dt/2) * Dt       */

__global__ void Resultant_Positions_Kernel(double* positions, double* velocity, double delta_time, int cant)
{
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i >= cant) {return;}
      double dtx = delta_time*20.454999999999;
      //double dtx=delta_time;
	positions[i] = positions[i] + (velocity[i] * dtx);
}











/* ******************************************************************
 * ******  RELOCATE POSITION IN THE CORRECT IMAGE BOX  *************
 * ***********    (ONLY FOR PERIODIC MODE)   		***********
 * ****************************************************************/

/*  -BOX_MAX              0              BOX_MAX   */
/*      |-----------------|-----------------|      */

__global__ void Adjustin_Positions_Kernel(double* position, double box_max, int cant)
{
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i >= cant) {return;}
      double pos = position[i] - box_max;
      if(pos > 0){
	position[i] = -box_max + fmod(pos, (double) (2*box_max));
      }
      if(pos < -2*box_max){
	position[i] = box_max + fmod(pos, (double) (2*box_max));
      }
      
}









/* **************************************************************
 * ******  CALCULATE KINETIC ENERGY FOR A PARTICLE  *************
 * *************************************************************/
/*            Ek = |v|^2  *  m / 2                  */
/*            Ek_x = (v_x)^2  *  m / 2              */
__global__ void Kinetic_Energy_Kernel(double* kE, double* vold, double* v, double* m, int* item_to_type, int cant)
{
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i>= cant) {return;}
      
    double vi = vold[i] + v[i];
      //   double vi=v[i];
	 int type = item_to_type[i];
      
     // kE[i] = vi * vi * m[type] / 2;

     kE[i] = vi * vi * m[type] / 8;
}











/* *************************************************
 * ******  SUMS ALL KINETIC ENERGIES  *************
 * ************************************************/

__global__ void Total_Kinetic_Energy_Kernel(double* kE, double* Ke_x, double*  Ke_y, double* Ke_z, int cant)
{
      /* Elemento del vector a calcular */
      unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
      
      if(i>= cant) {return;}
      
      kE[i] = Ke_x[i] + Ke_y[i] + Ke_z[i];
}




