#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "genann.h"
#include <time.h>
#include<omp.h>
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
    for (j = 0; j < samples; ++j) {
        const double *guess = genann_run(ann, input + j*28*28);
        double max = 0.0, max_cls = 0;
        int k =0, actual =0;
        for (k =0; k < 10; k++)
        {
            if (guess[k]> max) {
                max = guess[k];
                max_cls = k;
            }
            if (class[j*10 + k]== 1.0) actual = k;
    
        }
        if (class[j*10 + (int)max_cls] == 1.0) ++correct;
        //else {printf("Logic error.\n"); exit(1);
    }
    return correct;
}


int main(int argc, char *argv[])
{
    printf("GENANN example 4.\n");
    printf("Train an ANN on the MNIST dataset using backpropagation.\n");

    /* Load the data from file to train */
    load_mnist("mnist/train-images-idx3-ubyte","mnist/train-labels-idx1-ubyte");
    printf("load done\n");
    
    /* Initialize time elements */
//    clock_t start, end;
//    double cpu_time_used;
     
//    start = clock();
    double start_time = omp_get_wtime(); 
    /* 28*28 inputs.
     * 3 hidden layer(s) of 5 neurons.
     * 10 outputs (1 per class)
     */
    genann *ann = genann_init(28*28, 3, 10, 10);

    int i, j;
    int loops = 40;

    /* Train the network with backpropagation. */
    printf("Training for %d loops over data.\n", loops);
    for (i = 0; i < loops; ++i) {
            genann_train_omp(ann, input, class, .1, 28*28, 10,samples);
        }
    double time = omp_get_wtime() - start_time;
//    end = clock();
//    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
//    printf("train time taken time.h: %f \n",cpu_time_used);
    printf("train time omp : %f\n",time);
  
    /* Load data from file to test */
    free(input);
    free(class);
    load_mnist("mnist/t10k-images-idx3-ubyte","mnist/t10k-labels-idx1-ubyte");
    
    /* find accuracy */
    int correct = correct_predictions(ann);
    printf("\n\n %d/%d correct (%0.1f%%).\n", correct, samples, (double)correct / samples * 100.0);

    genann_free(ann);

    return 0;
}

