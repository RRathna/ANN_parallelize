#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "genann.h"

/* This example is to illustrate how to use GENANN.
 * It is NOT an example of good machine learning techniques.
 */


const char *path_to_data = "example/iris.data"; //ignore

double *input, *class;
int samples;
const char *class_names[] = {"0","1","2","3","4","5","6","7","8","9"};

void load_data() {  //ignore this function, not using it
    /* Load the iris data-set. */
    FILE *in = fopen("example/iris.data", "r");
    if (!in) {
        printf("Could not open file: %s\n", path_to_data);
        exit(1);
    }

    /* Loop through the data to get a count. */
    char line[1024];
    while (!feof(in) && fgets(line, 1024, in)) {
        ++samples;
    }
    fseek(in, 0, SEEK_SET);

    printf("Loading %d data points from %s\n", samples, path_to_data);

    /* Allocate memory for input and output data. */
    input = malloc(sizeof(double) * samples * 4);
    class = malloc(sizeof(double) * samples * 3);

    /* Read the file into our arrays. */
    int i, j;
    for (i = 0; i < samples; ++i) {
        double *p = input + i * 4;
        double *c = class + i * 3;
        c[0] = c[1] = c[2] = 0.0;

        if (fgets(line, 1024, in) == NULL) {
            perror("fgets");
            exit(1);
        }

        char *split = strtok(line, ",");
        for (j = 0; j < 4; ++j) {
            p[j] = atof(split);
            split = strtok(0, ",");
        }

        split[strlen(split)-1] = 0;
        if (strcmp(split, class_names[0]) == 0) {c[0] = 1.0;}
        else if (strcmp(split, class_names[1]) == 0) {c[1] = 1.0;}
        else if (strcmp(split, class_names[2]) == 0) {c[2] = 1.0;}
        else {
            printf("Unknown class %s.\n", split);
            exit(1);
        }

        /* printf("Data point %d is %f %f %f %f  ->   %f %f %f\n", i, p[0], p[1], p[2], p[3], c[0], c[1], c[2]); */
    }

    fclose(in);
} //ignore function, not using it

void load_mnist()
{
    mnist_data *data_t, *temp;
    unsigned int cnt;
    int ret;
    
    if (ret = mnist_load("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", &data_t, &cnt)) {
        printf("An error occured: %d\n", ret);
    } else {
        printf("image count: %d\n", cnt);
    }
    /* Allocate memory for input and output data. */
    input = malloc(sizeof(double) * cnt * 28*28);
    class = malloc(sizeof(int) * cnt * 10);
    temp = data_t;
    int i, j,k;
    for (i = 0; i < cnt; ++i) {
        double *p = input + i * 28*28;
        double *c = class + i * 10;
        c[0] = c[1] = c[2] = c[4] = c[5] = c[6] = c[7] = c[8] = c[9] = 0.0;
        printf("pointers allocated for data row %d \n",i);
        for (j = 0; j < 28; ++j) {
            //printf("row %d, image row %d \n",i,j);
            for ( k =0; k<28; ++k)
            {
                printf("row %d, image row %d, image col %d , value = %f \n",i,j,k, temp->data[j][k]);
                *(p + j*28 + k) = temp->data[j][k];
            }
        }
        printf("row %d saved\n",i);
        c[temp->label] = 1.0;
        temp = temp + sizeof(mnist_data);
    }
    samples = cnt;
    free(data_t);
}


int main(int argc, char *argv[])
{
    printf("GENANN example 4.\n");
    printf("Train an ANN on the MNIST dataset using backpropagation.\n");

    /* Load the data from file. */
    //load_data();
    load_mnist();
    /* 4 inputs.
     * 1 hidden layer(s) of 4 neurons.
     * 3 outputs (1 per class)
     */
    printf("load done\n");
    genann *ann = genann_init(28*28, 3, 5, 10);

    int i, j;
    int loops = 500;

    /* Train the network with backpropagation. */
    printf("Training for %d loops over data.\n", loops);
    for (i = 0; i < loops; ++i) {
        for (j = 0; j < samples; ++j) {
            genann_train(ann, input + j*4, class + j*3, .01);
        }
        /* printf("%1.2f ", xor_score(ann)); */
    }

    int correct = 0;
    for (j = 0; j < samples; ++j) {
        const double *guess = genann_run(ann, input + j*4);
        if (class[j*3+0] == 1.0) {if (guess[0] > guess[1] && guess[0] > guess[2]) ++correct;}
        else if (class[j*3+1] == 1.0) {if (guess[1] > guess[0] && guess[1] > guess[2]) ++correct;}
        else if (class[j*3+2] == 1.0) {if (guess[2] > guess[0] && guess[2] > guess[1]) ++correct;}
        else {printf("Logic error.\n"); exit(1);}
    }

    printf("%d/%d correct (%0.1f%%).\n", correct, samples, (double)correct / samples * 100.0);



    genann_free(ann);

    return 0;
}
