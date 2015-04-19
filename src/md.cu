/* Filename:  md.cu **************************************************************************** /
 *  THIS FILE CONTAINS THE GENERAL LOOP FOR MD(MOLECULAR DYNAMICS)
 *
 *
 * INPUT FILES (2):
 * 
 *   -LennardCoeffs
 *         type   sigma   epsilon   mass     (one entry line for each TYPE)
 * 
 *
 *   -particles.in:   
 *     cantParticles                            | Total number of particles in the system
 *     type   x   y   z   Vx   Vy   Vz   q	| One entry for each particle
 *     box params.  x	y   z                   | Optional
 *     nstlim					| Limit in number of steps  
 *     dt					|  
 *     temp0					| Target temperature
 *     tempi					| Initial temp.
 *     tautp					| Velocity correction coeff. 
 *     cut					| cutoff value				
 *     
 *     
 *     
 *
 *
 * ALGORITHM:
 *   1- Read coefficients
 *   2- Generate Lennard-Jones 3D matrix with values (place it in texture memory):
 *		For each kind of particle(1D):
 * 			Generate a total of num_samples_r of interaction against N number of particle types(using particle coeff.) (2D matrix)
 *   
 *   3-Read particle positions
 *	**********?????Ordenar y armar índices
 *   For each iteration (MD step) :
 *      4-Calcular distancias:
 *	    Cada partícula contra todas las otras
 *	    Armar matriz de distancias
 *      5- IF REQUIRED: Evaluate L-J potential (based on distance and pair of coefficients)
 * 	6- Evaluate forces between each pair of particles (derivativeMode of L-J potential -> based on distance and pair of coefficients):
 * 		Calculating the derivativeMode equation
 *	 OR 	Estimate forces using table of derivativeModes samples(closest distance entry)
 *	 OR 	Estimate forces from table lookup(using table of potential energy(higher)
 *	7- Sum partial forces for each particle.
 *	8- Get new positions (based on previous position, velocity and forces )
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
#include <iomanip>
#include <sys/time.h>


/** **************************************************************** **/
/** ************* DEFAULT GLOBAL VARIABLES VALUES ****************** **/
#define BLOCK_SIZE_X		512	
#define BLOCK_SIZE_Y		1
#define BLOCK_SIZE		(BLOCK_SIZE_X*BLOCK_SIZE_Y)

// #define textureSize		50000
#define DIF_FINITAS_DELTA	4

/** Variables físicas **/
#define NUM_TYPES		70
#define MAx			90		
#define MIn			0.01
#define DIST			(MAx - MIn)

#define DELTA_TIEMPO		0.0001
#define TEMP			100
#define TAO			0.1

#define BOX_MAX			999	// distancia máxima del 0 para cada coordenada
					// Determinamos un cubo de volumen = (2*BOX_MAX) ^3

/** Filenames **/
char* lennardTableFileName = "../Input/Pastorius/LennardCoeffs";
char* particlesFileName = "../Input/Pastorius/particles.in";
char* debugOutputFilename = "../Output/Pastorius/debug.out";  //?????????
char* outputFilename = "../Output/Pastorius/results.out";     //Simulation termodynamic vars.( Pot.E, temp, ..    (standard output path)
char* crdFilename = "../Output/Pastorius/mdcrd";              //Trajectory output   (standard output path)
char* frcFilename = "../Output/Pastorius/frc";              //Forces output   (standard output path)
char* timeFilename = "../Output/Pastorius/times/times.out";   //Performance output  (standard output path)

using namespace std;
// streamsize ss = cout.precision();



//cudaSetDevice(1);
/** **************************************************************** **/
/** ******************** GLOBAL VARIABLES ************************** **/
texture <float, cudaTextureType2D,cudaReadModeElementType> texRef;
double delta_time = DELTA_TIEMPO;
double temp0 = TEMP;
double tempi;
double tautp = TAO;

double Boltzmann_cte = 0.0019872041;
double box_max_x = BOX_MAX;
double box_max_y = BOX_MAX;
double box_max_z = BOX_MAX;
bool box = true;
int num_types = NUM_TYPES;
double energyDiff,  totalEnergyXstep , totInitialEnergy;  //used to save total Energy(if required)



//execution config 
bool cpu=false;  //RUN ON cpu ONLY
bool gpu=true;
bool texture=true;
double textureSize=50000;
bool globalmem=false;
bool periodicity = false;   //USE PERIODIC BOUNDARIES
bool potentials=false;  //create table of potentials
double cut = 12;
int dimBlockX= 1024;
int dimBlockY=1;
int cant_steps = 1;

// Execution modes (how derivatives are calculated)
bool derivativeMode = false;   //Use table of derivatives
bool potentialMode= true;      //Use table of potentials 
bool analyticMode = false;     //Solve equations each step  


//OUTPUT
bool results = false; 
int outputSteps=1;
bool time = true;  //Execution to get time performance (oposite to get results)
bool amberResults = false;
bool coordinatesOut = false;
bool forcesOut = false;











/** **************************************************************** **/
/** *************************** MD Main ******************************* **/
/** **************************************************************** **/



/* INPUT PARAMETERS :
 * 
 * General input:
 * 	-cut 			Define cutoff value
 * 	-energy			Calculate total energy (sum of potentialMode)
 * 	-periodic		Periodic boundaries
 * 	-textureSize [size]	Define texture size
 * 	-dimX			Define dimBlockX
 * 	-dimY			Define dimBlockY
 * 
 * 
 * Modes for calculating derivatives (main interest of algorithm):
 * 	-a		Run using equation to solve derivativeModes
 * 	-d		Run using table of derivativeModes to calculate derivativeModes
 * 	-p		Run using table of potentialMode to calculate derivativeModes
 * 	
 * 
 * GPU or cpu: 	
 * 	-cpu 	Execute all calculations on cpu
 *	-gpu	Execute on gpu (when possible)  **STANDARD MODE**
 *
 * Global memory(device or cpu) or Texture (only in gpu mode):
 *	-global		Store tables on global memory
 *	-texture	Store tables on texture memory 
 * 
 * Output:
 * 	-output			Write output to file(STANDARD INFORMATION)
 * 	-coord 			Write trajectory to output
 * 	-forces			Write forces to output
 * 	-t [timesFile]		Specify file to write execution times
 * 
*/	


/*  ****STANDARD MODE*****:  
 *             gpu, texture, NO-energy calculationsm NO-output at all except execution time */





int main( int argc, char* argv[] )
{
  

  //**********************
  // PROCESS INPUT PARAMS.
  //**********************
  
  
  

for(uint i = 0; i < argc; i++){
  
    
//        **********************
// 	  GENERAL CONFIGURATION
//        **********************
  
  
  
  
   /**  THIS VALUE IS READ FROM INPUT FILE  (MDIN)*/
//  DEFINE CUTOFF 
//       if(strcmp(argv[i], "-cut") == 0){
// 	cut = atoi(argv[i+1]);
//       }



//  DO ENERGY CALCULATIONS (CALCULATE POTENTIAL BETWEEN ALL PAIRS AND SUM THEM)
      if(strcmp(argv[i], "-energy") == 0){  
	energyCalc = true;   
	potentials=true;
      }
        
//  USE PERIODIC BOUNDARIES
      if(strcmp(argv[i], "-periodic") == 0){  
	/* Periodicity */
	periodicity = true;
      }
      if(strcmp(argv[i], "-textSize") == 0){    
	textureSize = atoi(argv[i+1]);
      }
      
//  GRID DIMENSIONS
      if(strcmp(argv[i], "-dimX") == 0){    
	dimBlockX = atoi(argv[i+1]);
      }
      if(strcmp(argv[i], "-dimY") == 0){   
	dimBlockY = atoi(argv[i+1]);
      }
      
      /**  THIS VALUE IS READ FROM INPUT FILE  (MDIN)*/
// MAX. STEPS
//       if(strcmp(argv[i], "-steps") == 0){    
// 	cant_steps = atoi(argv[i+1]);
//       }
      


      
      
      
//        **********************
// 	  EXECUTION MODES !!!!
//        **********************

  /* ANALYTIC mode */
      if(strcmp(argv[i], "-a") == 0){   //solve equations every step to calculate derivativeModes and potential(when required)
	analyticMode = true;
	potentialMode=false;   
	derivativeMode=false;
	texture=false;
	globalmem=false;
      }
  
  /* DERIVATIVE mode */
      if(strcmp(argv[i], "-d") == 0){   //Use tables of derivatives to estimate derivative values
	derivativeMode = true;   
	potentialMode=false;   
      }
  
  /* POTENTIAL mode */
      if(strcmp(argv[i], "-p") == 0){   //Use tables of potentials to estimate derivative valuess
	potentialMode = true;   
	derivativeMode=false;  
	potentials=true;  //will need potentials table
      }
   
     
     
     
//        **********************
//	  ****  OUTPUT !!!! **** 
//        **********************
  
// DEFINE IF I WANT TO SAVE RESULTS(LATER I DEFINE WHAT IS SAVED) OR ONLY MEASURE TIME PERFORM.  
      if(strcmp(argv[i], "-output") == 0){  //save results (POT. ENERGY, TEMP. , ETC)
	/* RESULTS or TIMER mode */    //THIS METHOD IS USED TO CHECK THE SIMULATION IS CORRECT (COMPARED TO AMBER)
	results = true;
	outputSteps=atoi(argv[i+1]);
	time=false;  //incompatible to write both results and time(output to file breaks performance)
      }
      
//       ONLY MEASURE TIME
      if(strcmp(argv[i], "-time") == 0){   //set performance mode of execution and define times output file
	/* outputTimeFilename  */
	time=true;
	timeFilename = argv[i+1];
      }
      
//       SAVE TRAJECTORY(mdcrd) IN AN INDEPENDENT FILE
      if(strcmp(argv[i], "-coord") == 0){    //Save traj. in file
	coordinatesOut = true;
	time=false; //incompatible to write both results and time(output to file breaks performance)
      }
      
      
//       SAVE FORCES FOR EACH STEP IN AN INDEPENDENT FILE
      if(strcmp(argv[i], "-forces") == 0){    
	forcesOut = true;
	time=false; //incompatible to write both results and time(output to file breaks performance)
      }
      
    
    

//     CPU vs GPU

      if(strcmp(argv[i], "-cpu") == 0){  //run calculations using cpu only
	cpu = true;
	gpu = false;
	texture=false;
      }
      
      if(strcmp(argv[i], "-gpu") == 0){  //run calculations using GPU (when possible)
	cpu = false;
	gpu = true;
      }
      
      
//    	TEXTURE vs GLOBAL-MEM   

      if(strcmp(argv[i], "-tex") == 0){   //use texture memory to store tables
       texture = true;
       }
     
      if(strcmp(argv[i], "-global") == 0){   //use global memory(device or cpu) to store tables
       globalmem = true;
       texture=false;  //dont use texture
      } 
 
   

 }
    
    
//CHECK IF PARAMETERS OF EXECUTION ARE COMPATIBLE WITH EACHOTHER    
//MOST OF THE CHECKING IS UP TO THE USER (SO, DONT USE OPPOSITE PARAMETERS)  
    
if (cpu){
  if(texture){
    cout << "ERROR: cant execute on cpu using texture device memory " << endl;
    return 0;
  }
}


    
    
//****************************************************************        
//****************************************************************    

//Print details of execution

    cout << "Execution conditions:" << endl;
    if (cpu)
      cout << "		Execution using CPU" << endl;
    if (gpu)
      cout << "		Execution using GPU" << endl;
    if (potentialMode)
      cout << "		Potentials table mode ON: Use potentials table" << endl;
    if (derivativeMode)
      cout << "		Derivatives table mode ON: Use derivatives table" << endl;
    if (analyticMode)
      cout << "		Analytic mode ON: solve equations to evaluate interactions" << endl;
    if(results){
      cout << "		DEBUG mode: ??????? - PRINT OUTPUT EVERY ***x*** STEPS" << endl;
    }
    if(coordinatesOut){
      cout << "		Coordinates mode ON: save trajectory" << endl;
    }
    if(forcesOut){
      cout << "		Forces output mode ON: save forces" << endl;
    }
    if(time){
      cout << "		Performance mode ON: dont print any output. Measure and write performance" << endl;
      }
    if(texture){
      cout << "		Texture mode ON: use texture memory to store tables" << endl;
    }
    if(globalmem){
      cout << "		Global memory mode ON: use global memory to store table" << endl;
    }
    
    
    

//****************************************************************        
//****************************************************************    
//****************************************************************        
    
   
   
   
   

   
   
   
   
//*************************************************************** */    
//**           CONFIGURE OUTPUT                                   */
//*************************************************************** */    


    // Termodynamic vars. output
    fstream out;    
    if(results){
	/* Output file */
	out.open(outputFilename,fstream::out);
	streamsize ss = out.precision();
	out << setprecision(20);
    }
    
    // Trajectory output stream
    fstream crd;    
    if(coordinatesOut){
	/* CRD output file */
	crd.open(crdFilename,fstream::out);
	crd << setprecision(3);
	crd.setf( std::ios::fixed, std:: ios::floatfield );
	crd << "  POS(x)  POS(y)  POS(z)" << endl;
    }
    
     // Forces output stream
    fstream frcOut;    
    if(forcesOut){
	/* FRC output file */
	frcOut.open(frcFilename,fstream::out);
	frcOut << setprecision(3);
	frcOut.setf( std::ios::fixed, std:: ios::floatfield );
	frcOut << " *********** " << endl;
    }


    // Execution time output stream
    struct timeval  tv1, tv2;
    fstream timeout;
    if(time){    //timer mode ON
	/* Time output file */
	timeout.open(timeFilename, fstream::app | fstream::out);
	timeout << setprecision(20);
    }
    


    
    
    
    
    
    
//*************************************************************** */    
//**           LOAD DATA FROM INPUT FILES 		          */
//*************************************************************** */    
     
          
/* LOAD LENNARD COEFFS TABLE */
   ifstream table (lennardTableFileName);
    table >> num_types;
    /**Variables y memoria*/
    size_t num_types_size = num_types * sizeof(double);
    
    vector<string> h_type;
    h_type.resize(num_types);
    double* h_sigma = (double*) ( malloc(num_types_size));
    double* h_epsilon = (double*) ( malloc(num_types_size));
    double* h_mass = (double*) ( malloc(num_types_size));
    
     
    for(int j = 0; j<num_types ; j++){
      table >> h_type[j];
      table >> h_sigma[j];
      table >> h_epsilon[j];
      table >> h_mass[j];
    }
  table.close();
    
  
   
  
/*READ PARTICLES POS and VEL FROM INPUT FILE*/

fstream particles;
    particles.open(particlesFileName);
    
    uint cant_particles;
    double* h_position_x;
    double* h_position_y;
    double* h_position_z;
    double* h_velocity_x;
    double* h_velocity_y;
    double* h_velocity_z;
    double* h_velocity_old_x;
    double* h_velocity_old_y;
    double* h_velocity_old_z;
    double* h_chargue;
    double h_box_x;
    double h_box_y;
    double h_box_z;
    double h_box_alpha;
    double h_box_beta;
    double h_box_gamma;
    vector<string> h_particle_type;
    particles >> cant_particles;    //PRIMER LINEA DE particles.in ES EL NUMERO DE PARTICULAS QUE HAY
    size_t cant_particles_size = cant_particles * sizeof(double);
    h_position_x = (double*)malloc(cant_particles_size);
    h_position_y = (double*)malloc(cant_particles_size);
    h_position_z = (double*)malloc(cant_particles_size);
    h_velocity_x = (double*)malloc(cant_particles_size);
    h_velocity_y = (double*)malloc(cant_particles_size);
    h_velocity_z = (double*)malloc(cant_particles_size);
    h_velocity_old_x = (double*)malloc(cant_particles_size);
    h_velocity_old_y = (double*)malloc(cant_particles_size);
    h_velocity_old_z = (double*)malloc(cant_particles_size);
    h_chargue = (double*)malloc(cant_particles_size);
    h_particle_type.resize(cant_particles);
    
    
    /** Read data : coord, velocities, type, charge **/
    for(uint i = 0; i < cant_particles ; i++) {
      particles >> h_particle_type[i];
      
      particles >> h_position_x[i];
      particles >> h_position_y[i];
      particles >> h_position_z[i];
      
      particles >> h_velocity_old_x[i];
      particles >> h_velocity_old_y[i];
      particles >> h_velocity_old_z[i];
      
      particles >> h_chargue[i];
    }
    
    
     
    /** PERIODICITY VALUES **/ 
    //TODO: por ahora usamos cubo,
    //situamos el cero en el centro del mismo
    //Recibimos en orden x, y, z
    particles >> box;
    if(box){
      cout << " Levantamos caja" << endl;
      particles >> h_box_x;
      particles >> h_box_y;
      particles >> h_box_z;
      particles >> h_box_alpha;
      particles >> h_box_beta;
      particles >> h_box_gamma;
      if( h_box_alpha != 90 or h_box_beta != 90 or h_box_gamma != 90){
	    cout << " Se forzaron los angulos para que sea un CUBO: " << endl;
      }
      box_max_x = h_box_x/2;
      box_max_y = h_box_y/2;
      box_max_z = h_box_z/2;
    }
    
    
    
    /** OTHER PARAMETERS**/
    particles >> cant_steps;
    particles >> delta_time;
    particles >> temp0;
    particles >> tempi;
    particles >> tautp;
    particles >> cut;
    
    particles.close();
    
 
  
    /*******  FINISH READING PARTICLES FILE *******/
   /*********************************************/
  
   
   
   
   
   
   
    
  
  /*******************************************************************************/  
 /*** CALCULATE DERIVATIVES/POTENTIALS VALUES TO FILL TABLES IN HOSTs MEMORY ****/
/*******************************************************************************/
/*********************** CALCULATION IS ALWAYS DONE ON GPU         ************/
/*****************************************************************************/


    int num_samples_r = textureSize/(sizeof(float));	// cant of original sample values (máximo permitido por mem de textura)
    double var = DIST / ((double) num_samples_r);	// variation of r
    size_t num_samples_r_size = num_samples_r * sizeof(float);
   
    //TOTAL NUMBER OF THREADS:  ThreadsX=num_samples_r    -   ThreadsY=num_types 
    int width = num_samples_r;
    int height = num_types;
    dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
    dim3 dimGrid( (int) ceil((double)width / (double)dimBlock.x), (int) ceil((double)height / (double)dimBlock.y) );
    
    
    
    
 /******************************* FILL TABLES USING GPU ********************/
 /********************   AND STORE ALL RESULTS IN HOSTs MEMORY ************/
    
 
    /* COPY EPSILON AND SIGMA VALUES TO DEVICE (WILL BE NEEDED TO CALCULATE d/POTENTIALS)*/
    double* d_EPS;     //EPSILON values on device
    double* d_SIG;      //SIGMA values on device
    cudaMalloc(&d_EPS, num_types_size);    
    cudaMalloc(&d_SIG, num_types_size);   
    cudaMemcpy(d_EPS, h_epsilon, num_types_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_SIG, h_sigma, num_types_size, cudaMemcpyHostToDevice);
    
         
 
    if(derivativeMode) {  //WILL NEED DERIVATIVES TABLE
      float* h_dLJPot;   // TABLE FOR DERIVATIVES VALUES (IN HOST MEMORY)
      float* d_LJPot;    //devices table
      h_dLJPot = (float*) malloc(num_samples_r_size*num_types*num_types);	// #samples * #particles * #particles (*float)
      cudaMalloc(&d_dLJPot, num_samples_r_size * num_types);
      for(int a = 0; a<num_types; a++){
	derivativeModes_lennard_Kernel<<<dimGrid, dimBlock>>>(d_dLJPot, d_EPS, d_SIG, h_epsilon[a], h_sigma[a], var, width, height);
        cudaMemcpy( (float*) &(h_dLJPot[(a*num_samples_r*num_types)]), d_dLJPot, num_types * num_samples_r_size, cudaMemcpyDeviceToHost);
       }
         cudaFree(&d_dLJPot);
    }
    
    
    if(potentials) {  //POTENTIALS TABLE IS REQUIRED 
      float* h_LJPot;    // TABLE FOR POTENTIAL VALUES (IN HOST MEMORY)
      float* d_dLJPot;  //devices table
      h_LJPot = (float*) malloc(num_samples_r_size*num_types*num_types);	// #samples * #particles * #particles (*float)
      cudaMalloc(&d_LJPot, num_samples_r_size * num_types);   //STORE DEVICEs MEMORY FOR EACH CALCULATION STEP
      for(int a = 0; a<num_types; a++){
	lennard_Kernel<<<dimGrid, dimBlock>>>(d_LJPot, d_EPS, d_SIG, h_epsilon[a], h_sigma[a], var, width, height);
      	cudaMemcpy( (float*) &(h_LJPot[(a*num_samples_r*num_types)]), d_LJPot, num_types * num_samples_r_size, cudaMemcpyDeviceToHost);
        
      }
      cudaFree(&d_LJPot);   
    }
	
    
  /*FREE DEVICE MEMORY*/
    cudaFree(&d_EPS);
    cudaFree(&d_SIG);

    
    
    
    
    
      /* ************************************************************************ */
     /*        AT THIS POINT THE TABLES ARE LOCATED IN HOST MEMORY      	 */
    /*      RELOCATE THE TABLES BASED ON THE DERIVATIVES CALCULATION MODE	*/
   /* ************************************************************************ */
    
 
// //  ********   GPU MODE ON ***********
if(gpu) {  
cudaError err;
  if (texture){
      //cudaArray* cuLennard_i;
      cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
      cudaMallocArray(&cuLennard_i, &channelDesc, num_samples_r, num_types*num_types);		//width x height
      texRef.addressMode[0] = cudaAddressModeClamp;
     //texRef.addressMode[0] = cudaAddressModeBorder;  
      texRef.filterMode = cudaFilterModeLinear; //cudaFilterModePoint; //		//Tipo de interpolación

      
//*TODO:  THE TEXTURES OF POTENTIALS AND DERIVATIVES MUST BE INDEPENDENT REFERENCES SO THEY CAN BE USED BOTH AT THE SAME TIME*/      
    if (derivativeMode){
	cudaMemcpyToArray(cuLennard_i, 0, 0, h_dLJPot, num_types * num_types * num_samples_r_size, cudaMemcpyHostToDevice);
	 /** BIND TEXTURE **/
	cudaBindTextureToArray(texRef, cuLennard_i, channelDesc); 
      }
    else{
     if (potential){
       cudaMemcpyToArray(cuLennard_i, 0, 0, h_LJPot, num_types * num_types * num_samples_r_size, cudaMemcpyHostToDevice);
       /** BIND TEXTURE **/
       cudaBindTextureToArray(texRef, cuLennard_i, channelDesc); 
       }   
    }
  	  
    /** WAIT FOR TEXTURE BIND **/
    cudaDeviceSynchronize();
  }
  else{   //COPY ALL TO DEVICES MEMORY (DONT USE TEXTURE)
     if(derivativeMode){
      err= cudaMalloc(&d_dLJPot, num_samples_r_size * num_types*num_types); 
		if( err != cudaSuccess)
			{
     			printf("CUDA error paso1: %s\n", cudaGetErrorString(err));
			}	    
		err=cudaMemcpy( d_dLJPot, h_dLJPot, num_types * num_types * num_samples_r_size, cudaMemcpyHostToDevice);
		if( err != cudaSuccess)
			{
			     printf("CUDA error paso2: %s\n", cudaGetErrorString(err));
			}
     }
     if(potential){
	      err= cudaMalloc(&d_LJPot, num_samples_r_size * num_types*num_types);
		if( err != cudaSuccess)
		{
		     printf("CUDA error paso3: %s\n", cudaGetErrorString(err));
		}	
		err=	 cudaMemcpy( d_LJPot, h_LJPot, num_types * num_types * num_samples_r_size, cudaMemcpyHostToDevice);
		if( err != cudaSuccess)
		{
		     printf("CUDA error paso4: %s\n", cudaGetErrorString(err));
		} 

      }
    }  
}
       






/** PRINT TABLE CONTENT  ********   DEBUG ********* */
// if(results){
// 	    if(derivativeMode)
// 	      out << " derivativeMode LENNARD " << endl;
// 	    else
// 	      out << " LENNARD " << endl;
// 	    for(int a = 0; a<num_types; a++){
// 	      out << " Type = " << h_type[a] << endl << "  ";
// 	      for(int i = 0; i<num_types; i++){
// 		for(int j = 0; j<num_samples_r; j+= num_samples_r/8){
// 		  if(derivativeMode)
// 		    out << h_dLJPot[(a*num_types*num_samples_r)+(i*num_samples_r)+j] << ", ";
// // 		  if(potentials)
// // 		    out << h_LJPot[(a*num_types*num_samples_r)+(i*num_samples_r)+j] << ", ";
// 		}
// 		out << endl << "  ";
// 	      }
// 	      out << "***********************************************************************************"  << endl;
// 	    }
//     }

    
    
    
    
    
    
     

    
    
   
  
  
  
  
//     if(results){
//       /** DEBUG **/
// 	    out << " INITIAL VALUES" << endl;
// 	    for(int i = 0; i<cant_particles; i++){
// 	      out << "  Type: " << h_particle_type[i] << " | Pos: (" << h_position_x[i] << " , " << h_position_y[i] << " , " << h_position_z[i] << ")";
// 	      out << " | Vel: (" << h_velocity_old_x[i] << " , " << h_velocity_old_y[i] << " , " << h_velocity_old_z[i] << ")" << endl;
// 	    }
// 	    out << endl;
// 
//       /** DEBUG **/
//     }
//     if(results){
    //   /** DEBUG **/
    // 	out << " CANT of TYPES" << endl;
    // 	for(int i = 0; i < h_type.size(); i++){
    // 	  out << "  " << h_type[i] << " " << cant_of_typ[i] << endl;
    // 	}
    // 	out << endl;
      /** DEBUG **/
//     }  
  
    /* Armamos estructura de items para saber de qué tipo
    /* es la partícula en la que estamos en CUDA */
    /** h_particle_type =  H H H H H K K K K K O O O O O O O O O ... **/
    /** h_item_particle =  1 1 1 1 1 3 3 3 3 3 9 9 9 9 9 9 9 9 9 ... **/
    
    
    
//CREATE ARRAY TO ASSOCIATE PARTICLES WITH ITS TYPE
    int * h_item_particle = (int*)malloc(cant_particles * sizeof(int));
    
    /**  FILL ARRAY : EACH TYPE IS DEFINED WITH AN INTEGER **/
    for(int i = 0; i< cant_particles; i++){
      for(int j = 0; j< h_type.size(); j++){
	if(h_type[j] == h_particle_type[i]){
	    h_item_particle[i] = j;
	    break;
	}
      }
    }
    
    
    int * d_item_particle;
    cudaMalloc(&d_item_particle, cant_particles * sizeof(int));
    cudaMemcpy(d_item_particle, h_item_particle, cant_particles * sizeof(int), cudaMemcpyHostToDevice);


//     if(results){
//       /** DEBUG **/
// 	    out << " ITEM to TYPE" << endl;
// 	    for(int i = 0; i < cant_particles; i++){
// 	      out << "  Particle[" << i << "]  | Type: " << h_type[h_item_particle[i]] << " (index :" << h_item_particle[i] << ") " << endl;
// 	    }
// 	    out << endl;
//       /** DEBUG **/
//     }  
  
  
    
    
    
    
    
    
    
  /* ************************************************ */
  /*     	ALLOCATE MEMORY ON DEVICE	      */
  /* ************************************************ */
    size_t s_size = cant_particles_size * cant_particles;
    
    /** Positions **/
    double* d_position_x;
    double* d_position_y;
    double* d_position_z;
    cudaMalloc(&d_position_x, cant_particles_size);
    cudaMalloc(&d_position_y, cant_particles_size);
    cudaMalloc(&d_position_z, cant_particles_size);
    cudaMemcpy(d_position_x, h_position_x, cant_particles_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_position_y, h_position_y, cant_particles_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_position_z, h_position_z, cant_particles_size, cudaMemcpyHostToDevice);
    
    /** Positions **/
    double* d_pos_close_x;
    double* d_pos_close_y;
    double* d_pos_close_z;
    cudaMalloc(&d_pos_close_x, cant_particles_size);
    cudaMalloc(&d_pos_close_y, cant_particles_size);
    cudaMalloc(&d_pos_close_z, cant_particles_size);
    
    /** Particle's mass **/
    double* d_mass;
    cudaMalloc(&d_mass, num_types_size);
    cudaMemcpy(d_mass, h_mass, num_types_size, cudaMemcpyHostToDevice);
    
    /** Velocities **/
    double* d_velocity_x;
    double* d_velocity_y;
    double* d_velocity_z;
    double* d_velocity_old_x;
    double* d_velocity_old_y;
    double* d_velocity_old_z;
    cudaMalloc(&d_velocity_x, cant_particles_size);
    cudaMalloc(&d_velocity_y, cant_particles_size);
    cudaMalloc(&d_velocity_z, cant_particles_size);
    cudaMalloc(&d_velocity_old_x, cant_particles_size);
    cudaMalloc(&d_velocity_old_y, cant_particles_size);
    cudaMalloc(&d_velocity_old_z, cant_particles_size);
    cudaMemcpy(d_velocity_old_x, h_velocity_old_x, cant_particles_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_old_y, h_velocity_old_y, cant_particles_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity_old_z, h_velocity_old_z, cant_particles_size, cudaMemcpyHostToDevice);
    
    /** Distances **/
    double* d_distance_x;
    double* d_distance_y;
    double* d_distance_z;
    double* d_distance_r;
    cudaMalloc(&d_distance_x, s_size);
    cudaMalloc(&d_distance_y, s_size);
    cudaMalloc(&d_distance_z, s_size);
    cudaMalloc(&d_distance_r, s_size);
    
    /** Derivatives **/
    double* d_dEr;
    cudaMalloc(&d_dEr, s_size);
    
    /** VDWAALS **/
    double* d_Er;
    cudaMalloc(&d_Er, s_size);
    
    /** Forces **/
    double* d_Force_x;
    double* d_Force_y;
    double* d_Force_z;
    cudaMalloc(&d_Force_x, s_size);
    cudaMalloc(&d_Force_y, s_size);
    cudaMalloc(&d_Force_z, s_size);
    
    double* d_Force_x_resultant;
    double* d_Force_y_resultant;
    double* d_Force_z_resultant;
    cudaMalloc(&d_Force_x_resultant, cant_particles_size);
    cudaMalloc(&d_Force_y_resultant, cant_particles_size);
    cudaMalloc(&d_Force_z_resultant, cant_particles_size);
    
    /** Kinetic Energy **/
    double* d_kinetic_energy;
    double* d_kinetic_energy_x;
    double* d_kinetic_energy_y;
    double* d_kinetic_energy_z;
    cudaMalloc(&d_kinetic_energy, cant_particles_size);
    cudaMalloc(&d_kinetic_energy_x, cant_particles_size);
    cudaMalloc(&d_kinetic_energy_y, cant_particles_size);
    cudaMalloc(&d_kinetic_energy_z, cant_particles_size);
    
  
    
    
    
    
    
    
    
  /* ************************************************ */
  /*          		HOST MEMORY	              */
  /* ************************************************ */
    /** Distances **/
    double (*h_distance_x)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    double (*h_distance_y)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    double (*h_distance_z)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
   // double (*h_distance_r)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
   double *h_distance_r = (double *) ( malloc(s_size*cant_particles));
 
    /** Forces **/
    double (*h_Force_x)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    double (*h_Force_y)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    double (*h_Force_z)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
    
    double* h_Force_x_resultant = (double*)malloc(cant_particles_size);
    double* h_Force_y_resultant = (double*)malloc(cant_particles_size);
    double* h_Force_z_resultant = (double*)malloc(cant_particles_size);
    
    /** Kinetic Energy **/
    double* h_kinetic_energy = (double*)malloc(cant_particles_size);
    double* h_kinetic_energy_x = (double*)malloc(cant_particles_size);
    double* h_kinetic_energy_y = (double*)malloc(cant_particles_size);
    double* h_kinetic_energy_z = (double*)malloc(cant_particles_size);
  
    
    
    
    
    
    
    
    
    
  /* ************************************************ */
  /*          TARGET KINETIC ENERGY????	              */
  /* ************************************************ */
  /*            Ek = Kb * T (3N - Nc) / 2             */
    double Nc = 5;
    double factor_conv_T_Ek = 2 / (Boltzmann_cte * (3 *cant_particles - Nc) );

    if(results){
	double kinetic_Energy = Boltzmann_cte * temp0 * (3*cant_particles - Nc) / 2;
	
	/** DEBUG **/
	    out << " THEORETICAL VALUES:" << endl << endl;
	    out << "  * Kb = " << Boltzmann_cte << endl << endl;
	    out << "  * Temperature = " << temp0 << endl << endl;
	    out << "  * Kinetic Energy = " << kinetic_Energy << endl << endl;
	    out << "  * Factor_conv_T_Ek = " << factor_conv_T_Ek << endl << endl;
	/** DEBUG **/
    }    
    
    
    
  
  
  
  
  
 if(results){
	out << endl << "   PROGRAM RUNNING...." << endl;
	out << "    NUMBER OF ITERATIONS = " << cant_steps << endl << endl;
    }
	  
 if(time){    //timer mode ON
	/** Arrancamos medicion del tiempo **/
	gettimeofday(&tv1, NULL);
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
/* ********************************************************************************************************** */
/* ****************************************** INICIO Iteracion DM ******************************************* */
/* ********************************************************************************************************** */
	    
    
    for(int step = 0; step < cant_steps; step++){
	
	    
	if(amberResults){
	    out << "/* ************************************************************************************************ */" << endl;
	    out << "/* ************************************* STARTING ITERATION " << step << " ************************ */" << endl;
	    out << "/* ************************************************************************************************ */" << endl;
	}
	
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    
	  /* *************************************************** */
	  /* 		CALCULATE DISTANCES BETWEEN PARTICLES	 */
	  /* *************************************************** */
	    
// ****	GRID DIMENSIONS ****	  
	    width = cant_particles;
	    height = cant_particles;
	    dimGrid.x = ceil((double)width / (double)dimBlock.x);
	    dimGrid.y = ceil((double)height / (double)dimBlock.y);
	    
	    
 if(!periodicity){
    distances_kernel<<<dimGrid, dimBlock>>>(d_distance_r, d_distance_x, d_distance_y, d_distance_z, d_position_x, d_position_y, d_position_z, width, height);
 	      
  } else {   
    close_distances_kernel<<<dimGrid, dimBlock>>>(d_distance_x, d_distance_y, d_distance_z, d_distance_r, d_position_x, d_position_y, d_position_z, h_box_x, h_box_y, h_box_z, width, height);
	      
 	    }
	    
	
	
// 	DOWNLOAD DISTANCES MATRIX 
	if (cpu)
	  cudaMemcpy(h_distance_r, d_distance_r, s_size, cudaMemcpyDeviceToHost);
	
	
	
	
	//if(results){
	    /** DEBUG **/
		/*cudaMemcpy(h_distance_r, d_distance_r, s_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_distance_x, d_distance_x, s_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_distance_y, d_distance_y, s_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_distance_z, d_distance_z, s_size, cudaMemcpyDeviceToHost);

		if (step %10000 == 0){
	
		out << " DISTANCES -  R" << endl << "  ";
        	 for(int i = 0; i<cant_particles; i++){
                  out << " " << i << "  | ";
                  for(int j = 0; j<cant_particles; j++){
                    out << h_distance_r[i][j] << "\t";
                  }
                  out << endl << "  ";
                }
                out << endl;

            }*/
/*
	   out << " DISTANCES -  X" << endl << "  ";
               for(int i = 0; i<cant_particles; i++){
                  out << " " << i << "  | ";
                  for(int j = 0; j<cant_particles; j++){
                    out << h_distance_x[i][j] << "\t";
                  }
                  out << endl << "  ";
                }
                out << endl;

	   out << " DISTANCES -  Y" << endl << "  ";
                 for(int i = 0; i<cant_particles; i++){
                  out << " " << i << "  | ";
                  for(int j = 0; j<cant_particles; j++){
                    out << h_distance_y[i][j] << "\t";
                  }
                  out << endl << "  ";
                }
                out << endl;
	

	   out << " DISTANCES -  Z" << endl << "  ";
                 for(int i = 0; i<cant_particles; i++){
                  out << " " << i << "  | ";
                  for(int j = 0; j<cant_particles; j++){
                    out << h_distance_z[i][j] << "\t";
                  }
                  out << endl << "  ";
                }
                out << endl;
*/
	/*	double (*matriz)[cant_particles] = (double (*)[cant_particles]) h_distance_r;
		for(int i = 0; i<cant_particles; i+= cant_particles){
		  out << " " << i << "  | ";
		  for(int j = 0; j<cant_particles; j+= cant_particles){
		    out << matriz[i][j] << "\t";
		  }
		  out << endl << "  ";
		}
		out << endl;
	  */ 
	 /** DEBUG **/ 
	//}    
	    
  
  //if(cpu)
	//double (*h_dEr)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
//  if(cpu)     
  //  double *h_dEr=(double *) malloc(s_size*cant_particles);
	  /* ************************************************ */
	  /*              Calculamos Derivadas                */
	  /* ************************************************ */
	    /** Variables y memoria **/


            dimBlock.x = BLOCK_SIZE_X;
            dimBlock.y = BLOCK_SIZE_Y;

	    width = cant_particles;
	    height = cant_particles;
	    dimGrid.x = ceil((double)width / (double)dimBlock.x);
	    dimGrid.y = ceil((double)height / (double)dimBlock.y);
	    
    if(analyticMode){
		if(cpu){   //VERSION ANALITICA SOBRE cpu
			out << "Entro a cpu analyticMode"<<endl;
			 double *h_dEr=(double *) malloc(s_size*cant_particles);
		        //double (*h_dEr)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
			if(h_dEr==NULL)	
				out<<"error malloc"<<endl;
			int x,y;
			for (x=0;x<cant_particles;x++)
	                     for(y=0;y<cant_particles;y++){
				analyticMode_cpu(h_dEr, h_distance_r, cut, h_item_particle, num_samples_r, h_epsilon, h_sigma, width, height,x,y); 
		    		//out<< "Iteracion"<<x*y<<endl;	
				}
			//mando los resultados a gpu
			//out<<"falta copiar"<<endl;   
                  err=cudaMemcpy( d_dEr,h_dEr, s_size, cudaMemcpyHostToDevice);
               		free(h_dEr);
		if( err != cudaSuccess)
                        {
                        printf("CUDA error paso1: %s\n", cudaGetErrorString(err));
                        }
		  
			out<<"termino todo"<< endl;
			}
			else 
				analyticMode_kernel<<<dimGrid, dimBlock>>>(d_dEr, d_distance_r, cut, d_item_particle, num_samples_r, d_EPS, d_SIG, width, height);
	       //if(step %100 ==0)
	       potential_analytic_kernel<<<dimGrid, dimBlock>>>(d_Er, d_distance_r, cut, d_item_particle, num_samples_r, d_EPS, d_SIG, width, height);
 		
   } else {  
           if(derivativeMode){
	          if (cpu){    //TABLA DE DERIVADAS SOBRE cpu
			 double *h_dEr=(double *) malloc(s_size*cant_particles);
		        int x,y;
		        for (x=0;x<cant_particles;x++)
		          for(y=0;y<cant_particles;y++)
		            derivativeMode_cpu(h_dLJPot,h_dEr, h_distance_r,cut,h_item_particle, num_samples_r,num_types,width,height, x, y );

		       //mando los resultados a gpu   
		        cudaMemcpy( d_dEr,h_dEr, s_size, cudaMemcpyHostToDevice);
		        free(h_dEr);
 
			} 
		        else
			    if(text)
			        derivativeMode_texture_kernel<<<dimGrid, dimBlock>>>(d_dEr, d_distance_r, cut, d_item_particle, num_samples_r, num_types, width, height);
			   else{
			derivativeMode_memory_kernel<<<dimGrid, dimBlock>>>(d_dLJPot, d_dEr, d_distance_r, cut, d_item_particle, num_samples_r, num_types, width, height);				
			//out<<"Salio"<<endl;  
		          	}
		 // if(step % 100 == 0)
                potential_analytic_kernel<<<dimGrid, dimBlock>>>(d_Er, d_distance_r, cut, d_item_particle, num_samples_r, d_EPS, d_SIG, width, height); 

		} else {    //PotentialsMode
		  if(cpu){    //TABLAS HACIENDO EL CALCULO SOBRE cpu
			  // out<< "ENTRO A cpu por tablas"<<endl;
			 double *h_dEr=(double *) malloc(s_size*cant_particles);
                          int x,y;
                          for (x=0;x<cant_particles;x++)
                            for(y=0;y<cant_particles;y++)
			      potentialsMode_cpu(h_LJPot,h_dEr, h_distance_r, cut, h_item_particle, num_samples_r, num_types,width, height,x,y);	
		        //mando los resultados a gpu   
                        cudaMemcpy( d_dEr,h_dEr, s_size, cudaMemcpyHostToDevice);
			free(h_dEr);

			}
		     
                       else{   //TABLAS HACIEND TODO SOBRE GPU
		            
                            if(text){    //USO LA MEMORIA DE TEXTURA
				//out<<"Entro a text - tabla "<<endl;
			        potentialsMode_texture_kernel<<<dimGrid, dimBlock>>>(d_dEr, d_distance_r, cut, d_item_particle, num_samples_r, num_types, width, height);
			      //out<<"salio text tabla"<<endl;
			     }
			     else     //USO DIRECTAMENTE EL ARREGLO EN MEMORIA DE GPU
 			      potentialsMode_memory_kernel<<<dimGrid, dimBlock>>>(d_LJPot, d_dEr, d_distance_r, cut, d_item_particle, num_samples_r, num_types, width, height);
			}	
               //if(step %100 == 0)
		 potentials_texture_kernel<<<dimGrid, dimBlock>>>(d_Er, d_distance_r, cut, d_item_particle, num_samples_r, d_EPS, d_SIG, width, height);
 
		//E_r<<<dimGrid, dimBlock>>>(d_Er, d_distance_r, cut, d_item_particle, num_samples_r, num_types, width, height);
 		}
 	    }
	    
	   // if(amberResults){
		//if(!derivativeMode){
		  /** DEBUG **/
		      //out << " Lennard-Jones" << endl << "  ";
		      double vdwaals = 0;
		      double (*h_Er)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
		      cudaError errato=cudaMemcpy(h_Er, d_Er, s_size, cudaMemcpyDeviceToHost);
			 if( errato != cudaSuccess)
                        {
                        printf("CUDA error paso1: %s\n", cudaGetErrorString(err));
                        }

		      for(int i = 0; i<cant_particles; i++){
 		//	out << " " << i << "  | ";
			for(int j = 0; j<cant_particles; j++){
 		//	  out << h_Er[i][j] << "\t";
			  if(i<=j)
			      vdwaals += h_Er[i][j];
			}
 		//	out << endl << "  ";
		      }  

// 		      out << endl;
		if(step == 0)
			totInitialEnergy= vdwaals;

		//if(step % 10000 == 0){
			totalEnergyXstep=vdwaals;
			//out << " STEP = " << step  << endl;
			//out << " VDWAALS = " << vdwaals << endl << endl;
		//	}
		      free(h_Er);
		  /** DEBUG **/
	//	}
	    //}
	    
	    
//	    if(results){
		  /** DEBUG **/
/*		      out << " DERIVATIVES" << endl << "  ";
		      double (*h_dEr)[cant_particles] = (double (*)[cant_particles]) ( malloc(s_size));
		      cudaMemcpy(h_dEr, d_dEr, s_size, cudaMemcpyDeviceToHost);
		      for(int i = 0; i<cant_particles; i+= cant_particles/8){
			out << " " << i << "  | ";
			for(int j = 0; j<cant_particles; j+= cant_particles/8){
			  out << h_dEr[i][j] << "\t";
			}
			out << endl << "  ";
		      }
		      out << endl;
		      free(h_dEr);
*/		  /** DEBUG **/ 
	   // }
	    if(results){
		  /** DEBUG **/
			cudaMemcpy(h_velocity_old_x, d_velocity_old_x, cant_particles_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_velocity_old_y, d_velocity_old_y, cant_particles_size, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_velocity_old_z, d_velocity_old_z, cant_particles_size, cudaMemcpyDeviceToHost);
			out << " OLD VELOCITIES" << endl;
			for(int i = 0; i<cant_particles; i++){
			  out << i+1 << ": (" << h_velocity_old_x[i] << " , " << h_velocity_old_y[i] << " , " << h_velocity_old_z[i] << ")" << endl;
			}
			out << endl;
		  /** DEBUG **/
	    }
	  





















	/* ************************************************ */
	  /*          Calculamos FUERZAS resultantes          */
	  /* ************************************************ */
	  /*   Fx =  dE(r) / dr  *  (x1-x2) / r               *
	  *   Fy =  dE(r) / dr  *  (y1-y2) / r               *
	  *   Fz =  dE(r) / dr  *  (z1-z2) / r               */
	   


            dimBlock.x = 1024;
            dimBlock.y = 1;
 
	    /* Calculo de vectores parciales */
	    /**Variables y memoria*/
	    width = cant_particles;
	    height = cant_particles;
	    dimGrid.x = ceil((double)width / (double)dimBlock.x);
	    dimGrid.y = ceil((double)height / (double)dimBlock.y);
	    
	    /** Calculo del vector F **/
	    Parcial_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_x, d_dEr, d_distance_x, d_distance_r, width, height);
	    Parcial_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_y, d_dEr, d_distance_y, d_distance_r, width, height);
	    Parcial_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_z, d_dEr, d_distance_z, d_distance_r, width, height);

	    //if(results){
		/** DEBUG **/
		      /*double fuerzaTot=0;
			cudaMemcpy(h_Force_x, d_Force_x, s_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_Force_y, d_Force_y, s_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_Force_z, d_Force_z, s_size, cudaMemcpyDeviceToHost);
			  out << " FORCES" << endl << "  ";
			  for(int i = 0; i<cant_particles; i++){
			    for(int j = 0; j<cant_particles; j++){
				 if(i<=j)
					fuerzaTot+=h_Force_x[i][j] + h_Force_y[i][j] + h_Force_z[i][j];
			     
				out << h_Force_x[i][j] << "\n" << h_Force_y[i][j] << "\n" << h_Force_z[i][j] << "\n";
				// out << "(" << h_Force_x[i][j] << " , " << h_Force_y[i][j] << " , " << h_Force_z[i][j] << ")\t";
			    }
			    out << endl << "  ";
			  }
			  out << endl;
		*/
		/** DEBUG **/
	    //}
	    
//		 out << "LA SUMA TOTAL DE FUERZAS ES: " << fuerzaTot << endl; 
	    /* Calculo del vector F */
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    
	    Resultant_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_x_resultant, d_Force_x, cant_particles);
	    Resultant_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_y_resultant, d_Force_y, cant_particles);
	    Resultant_Forces_Kernel<<<dimGrid, dimBlock>>>(d_Force_z_resultant, d_Force_z, cant_particles);

//	   if(results){
		/** DEBUG **/
/*
		      cudaMemcpy(h_Force_x_resultant, d_Force_x_resultant, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_Force_y_resultant, d_Force_y_resultant, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_Force_z_resultant, d_Force_z_resultant, cant_particles_size, cudaMemcpyDeviceToHost);
		      //out << " RESULTANT FORCES" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << h_Force_x_resultant[i] <<"\n" <<h_Force_y_resultant[i] << "\n"  << h_Force_z_resultant[i] <<  endl;

			//out << i+1 << ": (" << h_Force_x_resultant[i] << " , " << h_Force_y_resultant[i] << " , " << h_Force_z_resultant[i] << ")" << endl;
		      }
		      //out << endl;
*/
		/** DEBUG **/
//	    }
	    
	    
	    
	  /* ************************************************ */
	  /*       Calculamos VELOCIDADES Resultantes         */
	  /* ************************************************ */
	  /*  V(t + Dt/2) = V(t - Dt/2) +  [ F(t) * Dt ] / m  */  
	    
	    /**Variables y memoria*/
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    //out <<  "dtx= " << delta_time*20.455 << endl;
	    /** Piso las velocidades acumuladas al tiempo t con las nuevas de t+Dt */
	    Resultant_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_x, d_velocity_old_x, d_Force_x_resultant, d_mass, d_item_particle, delta_time, cant_particles);
	    Resultant_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_y, d_velocity_old_y, d_Force_y_resultant, d_mass, d_item_particle, delta_time, cant_particles);
	    Resultant_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_z, d_velocity_old_z, d_Force_z_resultant, d_mass, d_item_particle, delta_time, cant_particles);

	    if(results){
		/** DEBUG **/
		      cudaMemcpy(h_velocity_x, d_velocity_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_velocity_y, d_velocity_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_velocity_z, d_velocity_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      out << " RESULTANT VELOCITIES" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << i+1 << ": (" << h_velocity_x[i] << " , " << h_velocity_y[i] << " , " << h_velocity_z[i] << ")" << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }
	    
	  /* ************************************************ */
	  /*        Calculamos POSICIONES Resultantes         */
	  /* ************************************************ */
	  /*       P(t + Dt) = P(t) +  V(t + Dt/2) * Dt       */
	  /* (TODO: ajustar condiciones de perioricidad       */
	    
	    /**Variables y memoria*/
	    
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	   


 
	    Resultant_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_x, d_velocity_x, delta_time, cant_particles);
	    Resultant_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_y, d_velocity_y, delta_time, cant_particles);
	    Resultant_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_z, d_velocity_z, delta_time, cant_particles);

	    if(results){
		/** DEBUG **/
		      cudaMemcpy(h_position_x, d_position_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_position_y, d_position_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_position_z, d_position_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      out << " RESULTANT POSITIONS" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << i+1 << ": (" << h_particle_type[i] << " (" << h_position_x[i] << " , " << h_position_y[i] << " , " << h_position_z[i] << ")" << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }

	    
	    if(periodicity){
		    /* ************************************************ */
		    /*     Calculamos POSICIONES con PERIORICIDAD       */
		    /* ************************************************ */
		    /*       P(t + Dt) = P(t) +  V(t + Dt/2) * Dt       */
		      
		      /**Variables y memoria*/
		      dimBlock.x = 1024;
		      dimBlock.y = 1;
		      dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
		      dimGrid.y = 1;
		      
		      Adjustin_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_x, box_max_x, cant_particles);
		      Adjustin_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_y, box_max_y, cant_particles);
		      Adjustin_Positions_Kernel<<<dimGrid, dimBlock>>>(d_position_z, box_max_z, cant_particles);
	    }
	    if(coordinatesOut){
		/** DEBUG **/
		      cudaMemcpy(h_position_x, d_position_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_position_y, d_position_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_position_z, d_position_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      if(results){
			      out << " RESULTANT POSITIONS in the CUBE" << endl;
			      for(int i = 0; i<cant_particles; i++){
				out << i+1 << ": (" << h_particle_type[i] << " (" << h_position_x[i] << " , " << h_position_y[i] << " , " << h_position_z[i] << ")" << endl;
			      }
			      out << endl;
		      }
		      for(int i = 0; i<cant_particles; i+=2){
			crd << "  " << h_position_x[i] << "  " << h_position_y[i] << "  " << h_position_z[i];
			if(i+1 < cant_particles){
			  crd << "  " << h_position_x[i+1] << "  " << h_position_y[i+1] << "  " << h_position_z[i+1] << endl;
			} else
			  crd << endl;
		      } 
		      
		/** DEBUG **/
	    }
	  

	  /* ************************************************ */
	  /*        Calculamos Ek de cada partícula           */
	  /* ************************************************ */
	  /* Ek = |vp|^2  *  m / 2        con vp = (vold+v)/2 */
	  /*            Ek_x = (v_x)^2  *  m / 2              */
	    /**Variables y memoria*/
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    
	    /** Calculamos la energía cinética para las tres coordenadas de cada partícula      **/
	    /** Puede hacerse directamente así, sin calcular módulo por propiedades algebraicas **/
	    Kinetic_Energy_Kernel<<<dimGrid, dimBlock>>>(d_kinetic_energy_x, d_velocity_old_x, d_velocity_x, d_mass, d_item_particle, cant_particles);
	    Kinetic_Energy_Kernel<<<dimGrid, dimBlock>>>(d_kinetic_energy_y, d_velocity_old_y, d_velocity_y, d_mass, d_item_particle, cant_particles);
	    Kinetic_Energy_Kernel<<<dimGrid, dimBlock>>>(d_kinetic_energy_z, d_velocity_old_z, d_velocity_z, d_mass, d_item_particle, cant_particles);

	    if(results){
		/** DEBUG **/
		      cudaMemcpy(h_kinetic_energy_x, d_kinetic_energy_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_kinetic_energy_y, d_kinetic_energy_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_kinetic_energy_z, d_kinetic_energy_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      out << " KINETIC ENERGY" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << " " << i << "  | ";
			out << i+1 << ": (" << h_kinetic_energy_x[i] << " , " << h_kinetic_energy_y[i] << " , " << h_kinetic_energy_z[i] << ")" << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }
	    
	  /* ************************************************ */
	  /*            Calculamos Ek Resultante              */
	  /* ************************************************ */
	  /*               Ek_TOT = sum (Ek_i)                */  
	    
	    /**Variables y memoria*/
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    
	    /** Calculamos la Energía cinética total de cada partícula **/
	    Total_Kinetic_Energy_Kernel<<<dimGrid, dimBlock>>>(d_kinetic_energy, d_kinetic_energy_x, d_kinetic_energy_y, d_kinetic_energy_z, cant_particles);
	    
	    
	    /*  */
	    /** Calculamos la Energía cinética total del sistema **/
	    cudaMemcpy(h_kinetic_energy, d_kinetic_energy, cant_particles_size, cudaMemcpyDeviceToHost);
	    double Ek_TOT = 0;
	    for(int i = 0; i<cant_particles; i++){
		Ek_TOT += h_kinetic_energy[i];
	    }
	    
	    if(results){
		/** DEBUG **/
		      out << " KINETIC ENERGY" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << " " << i << "  | ";
			out << "  " << h_kinetic_energy[i] << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }
	    
	    //if(amberResults){
	if(step==0)
		totInitialEnergy=totInitialEnergy + Ek_TOT;

	if (step %10 == 0){
		totalEnergyXstep=totalEnergyXstep + Ek_TOT;
	      energyDiff= totalEnergyXstep - totInitialEnergy;	
	      //out << " Total Kinetic Energy(t) = " << Ek_TOT << endl << endl;
	      out << energyDiff << endl; 	        	
		}


	//   }

	    
	  /* ************************************************ */
	  /*        Calculamos Temperatura Resultante         */
	  /* ************************************************ */
	  /*          T(t) = 2*Ek_TOT / (Kb*(3N-Nc))          */
	    
	    double Temp_TOT = Ek_TOT *  factor_conv_T_Ek;

	    //if(amberResults){
		/** DEBUG **/
//		    if(step % 10000 == 0)
		//	out << " Temp(t) = " << Temp_TOT << endl << endl;
		/** DEBUG **/
	   // }
	    
	    
	  /* *********************************************** */
	  /*       Calculamos Factor de Correccion           */
	  /* *********************************************** */
	  /*   lambda = sqrt( 1 + 2 * dt / tautp * (T/T(t) -1) )   */
	    
	    double lambda = sqrt( 1 + delta_time / tautp * (temp0/Temp_TOT -1)  );


	    if(amberResults){
		/** DEBUG **/
		    out << " lambda(t) = " << lambda << endl << endl;
		/** DEBUG **/
	    }
	    
	  /* ************************************************ */
	  /*        Calculamos Velocidades Corregidas         */
	  /* ************************************************ */
	  /*                vi = lambda * vi                  */
	    
	    /**Variables y memoria*/
	    dimBlock.x = 1024;
	    dimBlock.y = 1;
	    dimGrid.x = ceil((double)cant_particles / (double)dimBlock.x);
	    dimGrid.y = 1;
	    
	    /** Piso las velocidades acumuladas al tiempo t+Dt con las nuevas de t+Dt corregidas */
	    Corrected_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_old_x, d_velocity_x, lambda, cant_particles);
	    Corrected_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_old_y, d_velocity_y, lambda, cant_particles);
	    Corrected_Velocities_Kernel<<<dimGrid, dimBlock>>>(d_velocity_old_z, d_velocity_z, lambda, cant_particles);

	    if(results){
		/** DEBUG **/
		      cudaMemcpy(h_velocity_x, d_velocity_old_x, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_velocity_y, d_velocity_old_y, cant_particles_size, cudaMemcpyDeviceToHost);
		      cudaMemcpy(h_velocity_z, d_velocity_old_z, cant_particles_size, cudaMemcpyDeviceToHost);
		      out << " CORRECTED RESULTANT VELOCITIES" << endl;
		      for(int i = 0; i<cant_particles; i++){
			out << i << ": (" << h_velocity_x[i] << " , " << h_velocity_y[i] << " , " << h_velocity_z[i] << ")" << endl;
		      }
		      out << endl;
		/** DEBUG **/
	    }
	    
	    dimBlock.x = BLOCK_SIZE_X;
	    dimBlock.y = BLOCK_SIZE_Y;  
	    
	    
	/* ********************************************************************************************************** */
	/* ******************************************* End of MD iteration ****************************************** */
	/* ********************************************************************************************************** */
    }
    
    
    
    
    
    
    
    
    
    
  string titulo("--");
  if(cpu)
	titulo +=  "cpu-";
  if(derivativeMode)
	titulo += "deriv-";
	else
	  if(analyticMode)
		titulo += "analyticMode-"; 
  	  else
		titulo += "tabla-";
  if(text)
	titulo += "texture";
	else
		titulo += "memory";


 
    if(time){    //timer mode ON
	gettimeofday(&tv2, NULL);
	timeout << cut << " " << (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec)<< " " << titulo << endl;
    }
    
//     if(!analyticMode){
	/** Unbindeamos Textura y liberamos memoria **/
	    cudaUnbindTexture(texRef);
	    cudaFreeArray(cuLennard_i);
//     }
    if(results or amberResults){
      out.close();
    }
    if(coordinatesOut){
      crd.close();
    }
  
  
  
  
  
  
  
  /* ************************************************ */
  /*         Free device memory		              */
  /* ************************************************ */
    cudaFree(&d_item_particle);
    
    /** Positions **/
    cudaFree(&d_position_x);
    cudaFree(&d_position_y);
    cudaFree(&d_position_z);
    
    /** Distances **/
    cudaFree(&d_distance_x);
    cudaFree(&d_distance_y);
    cudaFree(&d_distance_z);
    cudaFree(&d_distance_r);
    
    /** Particle's mass **/
    cudaFree(d_mass);
    
    /** Velocities **/
    cudaFree(d_velocity_x);
    cudaFree(d_velocity_y);
    cudaFree(d_velocity_z);
    
    /** Derivatives **/
    cudaFree(&d_dEr);
    cudaFree(&d_Er); 

    cudaFree(&d_LJPot);
    cudaFree(&d_dLJPot);

    /** Forces **/
    cudaFree(&d_Force_x);
    cudaFree(&d_Force_y);
    cudaFree(&d_Force_z);
    
    cudaFree(d_Force_x_resultant);
    cudaFree(d_Force_y_resultant);
    cudaFree(d_Force_z_resultant);
    
    /** Kinetic Energy **/
    cudaFree(d_kinetic_energy);
    cudaFree(d_kinetic_energy_x);
    cudaFree(d_kinetic_energy_y);
    cudaFree(d_kinetic_energy_z);
    
    
    
    
    
    
  /* ************************************************ */
  /*             Free host memory	              */
  /* ************************************************ */  
    free(h_sigma);
    free(h_epsilon);
    free(h_mass);
    
    /** Lennard Jones matrix**/
    if(derivativeMode)    
      free(h_dLJPot);
    else
      free(h_LJPot);
    
    free(h_item_particle);
    
    /** Positions **/
    free(h_position_x);
    free(h_position_y);
    free(h_position_z);
    
    /** Distances **/
    free(h_distance_x);
    free(h_distance_y);
    free(h_distance_z);
    free(h_distance_r);
    
    /** Velocities **/
    free(h_velocity_x);
    free(h_velocity_y);
    free(h_velocity_z);
    
    /** Chargue **/
    free(h_chargue);
    
    /** Forces **/
    free(h_Force_x);
    free(h_Force_y);
    free(h_Force_z);
    
    free(h_Force_x_resultant);
    free(h_Force_y_resultant);
    free(h_Force_z_resultant);
    
    /** Kinetic Energy **/
    free(h_kinetic_energy);
    free(h_kinetic_energy_x);
    free(h_kinetic_energy_y);
    free(h_kinetic_energy_z);
    
  return 0;
// }
