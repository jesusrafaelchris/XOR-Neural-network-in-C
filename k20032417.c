#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "libann.h"

int trIn[NUMTRAIN][NUMINPUT] = {0};
int trOut[NUMTRAIN] = {0};
double current_o[NUMTRAIN] = {0};

/*  ---------------------- DO NOT MODIFY ABOVE THIS LINE! ---------------------- */

/* 
 * Function to read training data.  Data from "in.csv" is stored in array
 * "trIn", and that from "out.csv" in array "trOut".
 *
 * Returns zero if success, or one if there is an error.
 */
int read_data(void) {
	/* --------------- Begin Answer to Task 1 Here ------------ */
    //Declare in and out files
    FILE *in;
    FILE *out;
    
    //Make trIn and trOut arrays
    //int trIn[4][2] = {0}; // size (0,0)
    //int trOut[4] = {0};   // size (0)
    
    in = fopen("in.csv", "r"); // open in.csv in read mode
    out = fopen("out.csv", "r"); // open out.csv in read mode
    
    if (in == NULL) { // if any errors opening the in file return 1 and exit
        fprintf(stderr, "Error reading file\n");
        return 1;
    }
    
    if (out == NULL) { // if any errors opening the out file return 1 and exit
        fprintf(stderr, "Error reading file\n");
        return 1;
    }

    //make the in array
    int c;
    int i = 0;
    int j = 0;
    int numberOfColumns = 2;
    int numberOfRows = 4;
    
    //hold each row as a string
    char row1[10];
    char row2[10];
    char row3[10];
    char row4[10];
    
    //get each row from the csv file
    fgets(row1,10,in);
    fgets(row2,10,in);
    fgets(row3,10,in);
    fgets(row4,10,in);
    
    //loop through all the rows and make the array
    for(i=0;i<numberOfRows;++i) {
        //printf("i= %d\n",i);
        for(j=0;j<numberOfColumns;++j){
            //printf("j= %d\n",j);
            if (i == 0){
                if (j == 1) {
                    trIn[i][j] = atoi(&row1[2]);
                }
                else {
                    trIn[i][j] = atoi(&row1[j]);
                }
            }
            else if (i == 1){
                if (j == 1) {
                    trIn[i][j] = atoi(&row2[2]);
                }
                else {
                    trIn[i][j] = atoi(&row2[j]);
                }
            }
            else if (i == 2){
                if (j == 1) {
                    trIn[i][j] = atoi(&row3[2]);
                }
                else {
                    trIn[i][j] = atoi(&row3[j]);
                }
            }
            else if (i == 3){
                if (j == 1) {
                    trIn[i][j] = atoi(&row4[2]);
                }
                else {
                    trIn[i][j] = atoi(&row4[j]);
                }
            }
        }
    }
    
    printf("trIn is ");
    for(i=0;i<numberOfRows;++i) {
        for(j=0;j<numberOfColumns;++j){
            printf("%d,",trIn[i][j]);
        }
    }
    printf("\n");
    //make the out array
    
    char outrow1[10];
    char outrow2[10];
    char outrow3[10];
    char outrow4[10];
    
    //get each row from the csv file
    fgets(outrow1,10,out);
    fgets(outrow2,10,out);
    fgets(outrow3,10,out);
    fgets(outrow4,10,out);
    

    //loop through all the rows and make the array
    for(i=0;i<numberOfRows;++i) {
            //printf("i= %d\n",i);
            if (i == 0){
                trOut[i] = atoi(&outrow1[0]);
            }
            else if (i == 1){
                trOut[i] = atoi(&outrow2[0]);
            }
            else if (i == 2){
                trOut[i] = atoi(&outrow3[0]);
            }
            else if (i == 3){
                trOut[i] = atoi(&outrow4[0]);
                
            }
        }
    printf("trOut is ");
    for(i=0;i<numberOfRows;++i) {
        printf("%d,",trOut[i]);
    }
    printf("\n");
    //close both files
    fclose(in);
    fclose(out);
    
    
	return 0;
	/* --------------- End Answer to Task 1 Here -------------- */
}

/* Function to produce output of one neuron */
double neuron(const int num_in, const double input[num_in], const double weight[num_in], const double bias) {
	/* --------------- Begin Answer to Task 2 Here ------------ */
    
    //num_in d number of inputs
    //input[num_in] array input values x1
    //weight[num_in] contains weights for inputs w1
    //bias bias term wj0
    
    float resultant_value = 0;
    int i;
    int j;
    float sum = 0;
    
    for (i=0;i<num_in;++i) {
        sum = ((input[i] * weight[i]) + bias);
        resultant_value += sum;
    }
    
    float activation_function_output = 1 / (1+ exp(-resultant_value));
    
    printf("      %f      \n", activation_function_output);

	return activation_function_output;
	/* --------------- End Answer to Task 2 Here -------------- */
}

void learn(void) {
	int i, k, p;
	double DeltaWeightIH[NUMHIDDEN][NUMINPUT], 
		   DeltaWeightHO[NUMOUTPUT][NUMHIDDEN],
		   DeltaBiasIH[NUMHIDDEN],
		   DeltaBiasHO[NUMOUTPUT];
	double WeightIH[NUMHIDDEN][NUMINPUT], WeightHO[NUMOUTPUT][NUMHIDDEN];
	double DeltaO[NUMOUTPUT+1], SumDOW[NUMHIDDEN+1], DeltaH[NUMHIDDEN+1];
	double biasIH[NUMHIDDEN], biasHO[NUMOUTPUT];
	double eta = 0.5, alpha = 0.9, smallwt = 0.5;
	int index[NUMTRAIN]={0};
	double error;

	init_network(DeltaWeightIH, DeltaWeightHO, DeltaBiasIH, DeltaBiasHO,
			WeightIH, WeightHO, biasIH, biasHO, smallwt);

	/* --------------- Begin Answer to Task 3 Here ------------ */

    int j;
    int m;
    
    //make index have 0-3
    //shuffle index
    
    for(i=0;i<4;++i) {
        index[i] = i;
        printf("%d", index[i]);
    }
    
    printf("\n");

    for(j=0;j<4;++j) {
        printf("\x1B[31m      Epoch %d      \n\n\x1B[0m", j);
        error = 0;
       for(m=0;m<1001;++m) {
           printf("\x1B[34m Step %d\n\x1B[0m", m);
	error = update_network(DeltaWeightIH, DeltaWeightHO, 
			        DeltaBiasIH, DeltaBiasHO,
					WeightIH, WeightHO, biasIH, biasHO,
					DeltaO, DeltaH, SumDOW, eta, alpha, index[j]);
           shuffle_index(index);
           
        }
        
    }

	/* --------------- End Answer to Task 3 Here -------------- */

	printf("\n\nPat\t") ;   // print network outputs
	for( i = 1 ; i <= NUMINPUT ; i++ ) {
		fprintf(stdout, "Input%-1d\t", i) ;
	}
	for( k = 1 ; k <= NUMOUTPUT ; k++ ) {
		fprintf(stdout, "Target%-1d\tOutput%-1d\t", k, k) ;
	}
	for( p = 0 ; p < NUMTRAIN ; p++ ) {
		fprintf(stdout, "\n%d\t", p) ;
		for( i = 0 ; i < NUMINPUT ; i++ ) {
			fprintf(stdout, "%d\t", trIn[p][i]) ;
		}
		for( k = 1 ; k <= NUMOUTPUT ; k++ ) {
			fprintf(stdout, "%d\t%f\t", trOut[p], current_o[p]) ;
		}
	}
	printf("\n");
	return;
}

/*  ---------------------- DO NOT MODIFY BELOW THIS LINE! ---------------------- */
