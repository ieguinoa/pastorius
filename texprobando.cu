/* Filename:  main.cu **************************************************************************** /
 *
 * INPUT:
 *   -Particulas.in:
 *     cantParticles
 *     type   x   y   z   Vx   Vy   Vz   q	; where
 *     dt					; (x,y,z)	= posición respecto de algún (0,0,0)
 *     temp0					; (Vx,Vy,Vz)	= Velocidades iniciales
 *     tempi					; q		= carga
 *     tautp					; dt		= delta_tiempo
 *     						; temp0		= temperatura target
 *     						; tempi		= temperatura inicial (No se usa aún)
 *     						; tautp		= factor de corrección de velocidades
 *     
 *     
 *     
 *   -TablaCoeficientesLennard
 *     type   sigma   epsilon   mass	min   max	; donde min y max indican de qué valor
 *     							; a qué valor hay que densificar las muestras
 *     							; (NO ESTA IMPLEMENTADO AUN)
 *
 * ALGORITMO:
 *   1-Levantar Coeficientes
 *   2-Armar matriz de lennard para cant_samples_r muestras
 *	Para cada tipo de partícula:
 *	    Calcular en funcion de los coeficientes el potencial para cant_samples_r valores r
 *   3-Levantar partículas
 *	Ordenar y armar índices
 *   Para cada iteración de MD:
 *	4-Calcular distancias:
 *	    Cada partícula contra todas las otras
 *	    Armar matriz de distancias
 *      5-Calcular las derivadas respecto de r para cada par de partículas
 *	6-Calcular fuerza para cada particula:
 *	    Cada partícula contra todas las otras: matriz 3D
 *	    Obtener fuerza resultante para cada partícula: vector 3D
 *	7-Calcular nuevas posiciones: vector 3D
 *
 ***************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <unistd.h>
#include <iomanip>
#include <sys/time.h>


/** **************************************************************** **/
/** ************* DEFAULT GLOBAL VARIABLES VALUES ****************** **/
#define BLOCK_SIZE_X		32
#define BLOCK_SIZE_Y		16
#define BLOCK_SIZE		(BLOCK_SIZE_X*BLOCK_SIZE_Y)

#define TEXTURE_MEM_SIZE	6500
#define DIF_FINITAS_DELTA	4

/** Variables físicas **/
#define CANT_TYPES		37
#define MAx			15		
#define MIn			0.3		
#define DIST			(MAx - MIn)

#define DELTA_TIEMPO		0.001
#define TEMP			100
#define TAO			0.1

#define BOX_MAX			12	// distancia máxima del 0 para cada coordenada
					// Determinamos un cubo de volumen = (2*BOX_MAX) ^3

//#define DEBUG			1

/** Filenames **/
char* lennardTableFileName = "Input_Mache/TablaCoeficientesLennard";
char* particlesFileName = "Input_Mache/particles.in";
char* debugOutputFilename = "Output_Mache/debug.out";
char* outputFilename = "Output_Mache/results.out";
char* crdFilename = "Output_Mache/mdcrd";
char* timeFilename = "Output_Mache/times.out";

using namespace std;
// streamsize ss = cout.precision();

/** **************************************************************** **/
/** ******************** GLOBAL VARIABLES ************************** **/
texture <float, 2,cudaReadModeElementType> texRef;
double delta_tiempo = DELTA_TIEMPO;
double temp0 = TEMP;
double tempi;
double tao = TAO;
double Boltzmann_cte = 0.0019872041;
double box_max = BOX_MAX;
int cant_steps = 1;
int cant_types = CANT_TYPES;


bool derivative = false;
bool analytic = false;


/** **************************************************************** **/
/** ************************* DEVICE ******************************* **/



__global__ void kernelCroto(float* dato, float index){


 //dato[0]= tex1D( texRef, index+0.5);

	dato[0]=tex2D(texRef, index+0.5, 0.5);
}



//comienzo

/** **************************************************************** **/
/** *************************** HOST ******************************* **/

int main( int argc, char* argv[] )
{
   
  int cantDatos=atoi(argv[1]);
  float position=atof(argv[2]);
   /*
    int cant_samples_r = TEXTURE_MEM_SIZE/(sizeof(float));	// cant of original sample values (máximo permitido por mem de textura)
    double var = DIST / ((double) cant_samples_r);		// variation of r
    size_t cant_samples_r_size = cant_samples_r * sizeof(float);
   */
    
    
//cudaArray* d_prueba;
//cudaChannelFormatDesc descriptor = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
//cudaMallocArray(&d_prueba, &descriptor,200*2000);

  float* datos;
   datos=(float*) malloc(sizeof(float)*cantDatos );


   for(int a = 0; a<cantDatos; a++){
	   datos[a]=a;
	
      }
  
  

cudaArray* d_datos;
    //     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>( );

     cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
      cudaMallocArray(&d_datos, &channelDesc, cantDatos,1);
      
      texRef.addressMode[0] = cudaAddressModeClamp;
 texRef.addressMode[1] = cudaAddressModeClamp;
 texRef.addressMode[2] = cudaAddressModeClamp;
      
texRef.filterMode = cudaFilterModeLinear; //cudaFilterModePoint; //		//Tipo de interpolación
      
     cudaMemcpyToArray(d_datos, 0, 0, datos, sizeof(float)*cantDatos , cudaMemcpyHostToDevice);
      cudaBindTextureToArray(texRef, d_datos, channelDesc); 
  

dim3 dimBloque(1,1);
dim3 dimGrid(1,1);
int size=5;
float* datosHost = (float*)malloc(size * sizeof(float));
datosHost[0] = 0.0;
datosHost[1] = 0.0;
datosHost[2] = 0.0;
datosHost[3] = 0.0;
datosHost[4] = 5.0;

float* datosDevice;
cudaMalloc(&datosDevice, 5 * sizeof(float));

cudaError_t err = cudaMemcpy(datosDevice,datosHost,5 * sizeof(float), cudaMemcpyHostToDevice);
if(err !=0){
    printf("Cudamemcpy threw error\n");
    getchar();  
}

kernelCroto<<<dimGrid,dimBloque>>>(datosDevice, position);


err = cudaMemcpy(datosHost,datosDevice,5 * sizeof(float), cudaMemcpyDeviceToHost);
if(err !=0){
    printf("Cudamemcpy threw error\n");
    getchar();
}


int r1,r2;

r1=10;
r2=100;

float eps12=0.0860;
float sig12=3.3996695087738837;
float result1=(float) 24.0*eps12*( pow(sig12,6)/ pow(r1,7) - 2 * pow(sig12,12)/ pow(r1,13));

float result2=(float) 24.0*eps12*( pow(sig12,6)/ pow(r2,7) - 2 * pow(sig12,12)/ pow(r2,13));

printf("El resultado 1 = %g\n", result1);
printf("El resultado 2 = %g\n", result2);

printf("La suma total daria = %g\n", 240*240*result2);

//printf("El dato es %f\n", datosHost[0]);
//  sleep(20); 
	    cudaUnbindTexture(texRef);
	    cudaFree(datosDevice);
	    cudaFreeArray(d_datos);
	//	cudaFreeArray(d_prueba);
	    free(datosHost);

  return 0;
}
