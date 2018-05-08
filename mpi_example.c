#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "genann.h"
#include <time.h>
#include <mpi.h>

double *input, *class;
unsigned int samples;
const char *class_names[] = {"0","1","2","3","4","5","6","7","8","9"};


void load_mnist(char *images_fname, char *labels_fname)
{
    mnist_data *data_t, *temp;
    unsigned int cnt;
    int ret;
    
    if (ret = mnist_load(images_fname, labels_fname, &data_t, &cnt)) {
        printf("An error occured: %d\n", ret);
    } else {
        printf("image count: %d\n", cnt);
    }
    //cnt = 500; // was used for debugging with smaller data, to reduce run time
    /* Allocate memory for input and output data. */
    input = (double *) malloc(sizeof(double) * cnt * 28*28);
    if (input == NULL)
    {
        printf("Input malloc error");
        exit(-1);
    }
    class = (double *) malloc(sizeof(double) * cnt * 10);
    if (class == NULL)
    {
        printf("class malloc error");
        exit(-1);
    }
    

    temp = data_t;
    int i, j,k;
    for (i = 0; i <cnt; ++i) {
        double *p = input + i * 28*28;
        double *c = class + i * 10;
        c[0] = c[1] = c[2] = c[4] = c[5] = c[6] = c[7] = c[8] = c[9] = 0.0;
        //printf("pointers allocated for data row %d \n",i);
        for (j = 0; j < 28*28; ++j) {
               //printf("data line %d, j %d, image row %d,image col %d value = %f \n",i,j,j/28,j%28, temp->data[j/28][j%28]);
               *(p + j) = temp->data[j/28][j%28];
            }

        *(c + (int)temp->label) = 1.0;
        temp = temp + 1;
    }
    samples = cnt;
    //printf("image count %d", cnt);
    free(data_t);
}

int correct_predictions(genann *ann) {
    int correct = 0, j =0;
    for (j = 0; j < samples; ++j) 
    {
        const double *guess = genann_run(ann, input + j*28*28);
        double max = 0.0;
        int k =0, actual =0, max_cls = 0;
        for (k =0; k < 10; k++)
        {
            if (guess[k]> max) {
                max = guess[k];
                max_cls = k;
            }
            if (class[j*10 + k]== 1.0) actual = k;
        } 
//        printf(" predicted %d, actual %d \n",max_cls, actual);
        if (class[j*10 + (int)max_cls] == 1.0) ++correct;
        //else {printf("Logic error.\n"); exit(1);
    }
    return correct;
}

/* example function to access weights

void genann_randomize(genann *ann) {
    int i;
#pragma omp parallel for
    for (i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();
        /* Sets weights from -0.5 to 0.5.
        ann->weight[i] = r - 0.5;
    }
}
*/


int main(int argc, char *argv[])
{
//    printf("GENANN example 4.\n");
//    printf("Train an ANN on the MNIST dataset using backpropagation.\n");

    /* Load the data from file to train */
    MPI_Init(&argc, &argv);
    int w_size;
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
      load_mnist("mnist/train-images-idx3-ubyte","mnist/train-labels-idx1-ubyte");
    }
      /* Initialize time elements */
      double ts, te;     
      ts = MPI_Wtime();
      MPI_Bcast(&samples, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    


    /* scatter data to all the other nodes */

    int *count, *disp, *count_c, *disp_c,sum =0, sum_c = 0;
    count =(int *)  malloc(sizeof(int)*w_size); //to define limits of transfer total floats
    disp = (int *) malloc(sizeof(int)*w_size);
    count_c = (int *)  malloc(sizeof(int)*w_size);
    disp_c = (int *) malloc(sizeof(int)*w_size); 

    int s_size= samples/w_size;
    if (rank < samples%w_size) s_size++;
    
    for (int i = 0; i < w_size; i++) { //find the number of elems to send to each processor
        count[i] = samples/w_size*28*28;
        count_c[i] = samples/w_size*10;
        if (i < samples%w_size) {  count[i] += 28*28; count_c[i] +=10;  }
        disp[i] = sum; disp_c[i] = sum_c;
        sum += count[i]; sum_c +=count_c[i];
    }  
    double *s_data, *s_class;
    s_data = (double *) malloc(sizeof(double) * count[rank]);
    s_class = (double *) malloc(sizeof(double) * count_c[rank]);

    MPI_Scatterv(input,count, disp, MPI_DOUBLE,s_data,count[rank],MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatterv(class,count_c, disp_c, MPI_DOUBLE,s_class,count_c[rank],MPI_DOUBLE,0,MPI_COMM_WORLD);
//    printf(" rank %d, cls[20]  =%f, %f, %f, %f, %f, %f, %f, %f, %f, %f \n",rank,s_class[20],s_class[21],s_class[22],s_class[23],s_class[24],s_class[25],s_class[26],s_class[27],s_class[28],s_class[29]);
    /* 28*28 inputs.
     * 3 hidden layer(s) of 10 neurons.
     * 10 outputs (1 per class)
     */
//    printf("load done\n");
    genann *ann = genann_init(28*28, 3, 10, 10);

    int i, j;
    int loops = 20;

    /* Train the network with backpropagation. */
//    printf("Training for %d loops over data by rank %d\n", loops, rank);
    for (i = 0; i < loops; ++i) {
        for (j = 0; j < s_size; ++j) {
            genann_train(ann, s_data + j*28*28,s_class + j*10, .1);
        }
//        printf("before reduce rank %d, ann->weight[20] : %f \n",rank,ann->weight[20]);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE,ann->weight,ann->total_weights,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        for (j=0;j<ann->total_weights;j++) { ann->weight[j] = ann->weight[j]/w_size; }
        MPI_Barrier(MPI_COMM_WORLD);
//        printf("after reduce rank %d, ann->weight[20] : %f \n",rank,ann->weight[20]);
    }
    
    te = MPI_Wtime();
    double cpu_time_used = (double) (te - ts);
    if (rank == 0) { printf("train time taken : %f \n",cpu_time_used);}

    free(input);
    free(class);
    
    if (rank == 0)
    {
    /* Load data from file to test */
    load_mnist("mnist/t10k-images-idx3-ubyte","mnist/t10k-labels-idx1-ubyte");
    
    /* find accuracy */
    int correct = correct_predictions(ann);
    printf("\n\n %d/%d correct (%0.1f%%).\n", correct, samples, (double)correct / samples * 100.0);
    }
    MPI_Finalize();
    free(s_data);
    free(s_class);
    genann_free(ann);

    return 0;
}
