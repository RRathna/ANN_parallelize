/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015, 2016 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * Edited by R. Rathna on 05/09/2018
 *   1. Removed dependency on previous iteration from loop
 *   2. Added omp parallelism
 *   3. Changed design of genann_train()
 */

#include "genann.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define LOOKUP_SIZE 4096

double genann_act_sigmoid(double a) {
    if (a < -45.0) return 0;
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}


double genann_act_sigmoid_cached(double a) {
    /* If you're optimizing for memory usage, just
     * delete this entire function and replace references
     * of genann_act_sigmoid_cached to genann_act_sigmoid
     */
    const double min = -15.0;
    const double max = 15.0;
    static double interval;
    static int initialized = 0;
    static double lookup[LOOKUP_SIZE];
    
    /* Calculate entire lookup table on first run. */
    if (!initialized) {
        interval = (max - min) / LOOKUP_SIZE;
        int i;
        for (i = 0; i < LOOKUP_SIZE; ++i) {
            lookup[i] = genann_act_sigmoid(min + interval * i);
        }
        /* This is down here to make this thread safe. */
        initialized = 1;
    }
    
    int i;
    i = (int)((a-min)/interval+0.5);
    if (i <= 0) return lookup[0];
    if (i >= LOOKUP_SIZE) return lookup[LOOKUP_SIZE-1];
    return lookup[i];
}


double genann_act_threshold(double a) {
    return a > 0;
}


double genann_act_linear(double a) {
    return a;
}


genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;
    
    
    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);
    
    const int total_neurons = (inputs + hidden * hidden_layers + outputs);
    
    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;
    
    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;
    
    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;
    
    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;
    
    genann_randomize(ret);
    
    ret->activation_hidden = genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid_cached;
    
    return ret;
}


genann *genann_read(FILE *in) {
    int inputs, hidden_layers, hidden, outputs;
    int rc;
    
    errno = 0;
    rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if (rc < 4 || errno != 0) {
        perror("fscanf");
        return NULL;
    }
    
    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);
    
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        errno = 0;
        rc = fscanf(in, " %le", ann->weight + i);
        if (rc < 1 || errno != 0) {
            perror("fscanf");
            genann_free(ann);
            
            return NULL;
        }
    }
    
    return ann;
}


genann *genann_copy(genann const *ann) {
    const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;
    
    memcpy(ret, ann, size);
    
    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;
    
    return ret;
}


void genann_randomize(genann *ann) {
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        ann->weight[i] = r - 0.5;
    }
}


void genann_free(genann *ann) {
    /* The weight, output, and delta pointers go to the same buffer. */
    free(ann);
}


double const *genann_run(genann const *ann, double const *inputs) {
    double const *w = ann->weight;
    double *o = ann->output + ann->inputs;
    double const *i = ann->output;
    
    /* Copy the inputs to the scratch area, where we also store each neuron's
     * output, for consistency. This way the first layer isn't a special case. */
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);
    
    int h, j, k;
    
    const genann_actfun act = ann->activation_hidden;
    const genann_actfun acto = ann->activation_output;
    //printf("ann->hidden_layers : %f; ann->hidden : %f \n",ann->hidden_layers, ann->hidden);
    /* Figure hidden layers, if any. */
    for (h = 0; h < ann->hidden_layers; ++h) {
        for (j = 0; j < ann->hidden; ++j) {
            double sum = *w++ * -1.0;
            //printf("sum : %f",sum);
            for (k = 0; k < (h == 0 ? ann->inputs : ann->hidden); ++k) {
                sum += *w++ * i[k];
            }
            *o++ = act(sum);
        }
        
        
        i += (h == 0 ? ann->inputs : ann->hidden);
    }
    
    double const *ret = o;
    
    /* Figure output layer. */
    for (j = 0; j < ann->outputs; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < (ann->hidden_layers ? ann->hidden : ann->inputs); ++k) {
            sum += *w++ * i[k];
        }
        *o++ = acto(sum);
    }
    
    /* Sanity check that we used all weights and wrote all outputs. */
    assert(w - ann->weight == ann->total_weights);
    assert(o - ann->output == ann->total_neurons);
    
    return ret;
}


void genann_train_omp(genann const *ann, double const *input, double const *desired_output, double learning_rate, unsigned int size_i, unsigned int size_c, unsigned int count) {
    int I = 0;
#pragma omp parallel
    {
#pragma omp for
        for ( I =0; I < count; I++)
        {
            double const *inputs = input + I*size_i;
            double const *desired_outputs = desired_output + I*size_c;
            /* To begin with, we must run the network forward. */
            genann_run(ann, inputs);
            
            int h, j, k,t;
            /* First set the output layer deltas. */
            {
                double const *o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers; /* First output. */
                double *d = ann->delta + ann->hidden * ann->hidden_layers; /* First delta. */
                double const *t = desired_outputs; /* First desired output. */
                //double const *oo = ann->output + ann->inputs + ann->hidden * ann->hidden_layers; /* First output. */
                //double *dd = ann->delta + ann->hidden * ann->hidden_layers; /* First delta. */
                //double const *tt = desired_outputs; /* First desired output. */
                
                
                /* Set output layer deltas. */
                if (ann->activation_output == genann_act_linear) {
                    //#pragma omp parallel for
                    for (j = 0; j < ann->outputs; ++j) {
                        d[j] = t[j] - o[j];
                        //*dd++ = *tt++ - *oo++;
                    }
                    d += ann->outputs;
                    t += ann->outputs;
                    o += ann->outputs;
                } else {
                    //#pragma omp parallel for
                    for (j = 0; j < ann->outputs; ++j) {
                        d[j] = (t[j] - o[j]) * o[j] * (1.0 - o[j]);
                        //*dd++ = (*tt - *oo) * *oo * (1.0 - *oo);
                        //++oo; ++tt;
                        //printf(" d[%d] : %f, *(dd-1) : %f\n",j,d[j],*(dd-1));
                    }
                    d += ann->outputs;
                    t += ann->outputs;
                    o += ann->outputs;
                }
            }
            
            
            
            /* Set hidden layer deltas, start on last layer and work backwards. */
            /* Note that loop is skipped in the case of hidden_layers == 0. */
            double delta = 0;
            t = ann->outputs;
            //int t = (h == ann->hidden_layers-1 ? ann->outputs : ann->hidden);
            h = ann->hidden_layers -1;
            
//#pragma omp for collapse(2)
            for (j = 0; j < ann->hidden; ++j) {
                for (k = 0; k < t; ++k) {
                    /* Find first output and delta in this layer. */
                    double const *o = ann->output + ann->inputs + (h * ann->hidden);
                    double *d = ann->delta + (h * ann->hidden);
                    
                    /* Find first delta in following layer (which may be hidden or output). */
                    double const * const dd = ann->delta + ((h+1) * ann->hidden);
                    
                    /* Find first weight in following layer (which may be hidden or output). */
                    double const * const ww = ann->weight + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));
                    if (k == 0) delta = 0;
                    const double forward_delta = dd[k];
                    const int windex = k * (ann->hidden + 1) + (j + 1);
                    const double forward_weight = ww[windex];
                    delta += forward_delta * forward_weight;
                    if (k == t-1)
                    {
                        d[j] = o[j] * (1.0-o[j]) * delta;
                    }
                }
            }
            
            t = ann->hidden;
            
//#pragma omp for collapse(3)
            for (h = ann->hidden_layers - 2; h >= 0; --h) {
                for (j = 0; j < ann->hidden; ++j) {
                    for (k = 0; k < t; ++k) {
                        /* Find first output and delta in this layer. */
                        double const *o = ann->output + ann->inputs + (h * ann->hidden);
                        double *d = ann->delta + (h * ann->hidden);
                        
                        /* Find first delta in following layer (which may be hidden or output). */
                        double const * const dd = ann->delta + ((h+1) * ann->hidden);
                        
                        /* Find first weight in following layer (which may be hidden or output). */
                        double const * const ww = ann->weight + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));
                        if (k == 0) delta = 0;
                        const double forward_delta = dd[k];
                        const int windex = k * (ann->hidden + 1) + (j + 1);
                        const double forward_weight = ww[windex];
                        delta += forward_delta * forward_weight;
                        if (k == t-1)
                        {
                            d[j] = o[j] * (1.0-o[j]) * delta;
                        }
                    }
                }
            }
            
            /* Train the outputs. */
            
            /* Find first output delta. */
            double const *d = ann->delta + ann->hidden * ann->hidden_layers; /* First output delta. */
            
            /* Find first weight to first output delta. */
            double *w = ann->weight + (ann->hidden_layers
                                       ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * ann->hidden * (ann->hidden_layers-1)): (0));
            
            /* Find first output in previous layer. */
            double const * const i = ann->output + (ann->hidden_layers
                                                    ? (ann->inputs + (ann->hidden) * (ann->hidden_layers-1)): 0);
            t = (ann->hidden_layers ? ann->hidden : ann->inputs) + 1;
            
            /* Set output layer weights. */
//#pragma omp for collapse(2)
            for (j = 0; j < ann->outputs; ++j) {
                for (k = 0; k < t; ++k) {
                    if (k == 0) {
                        w[j*t + k] += d[j] * learning_rate * -1.0;
                    } else {
                        w[j*t + k] += d[j] * learning_rate * i[k-1];
                    }
                }
            }
            w += ann->outputs*t;
            d+=ann->outputs;
            
            assert(w - ann->weight == ann->total_weights);
            
            
            /* Train the hidden layers. */
            t = ann->hidden + 1;
            
            
//#pragma omp for collapse(3)
            for (h = ann->hidden_layers - 1; h > 0; --h) {
                
                for (j = 0; j < ann->hidden; ++j) {
                    for (k = 0; k < t; ++k) {
                        /* Find first delta in this layer. */
                        double const *d = ann->delta + (h * ann->hidden);
                        
                        /* Find first input to this layer. */
                        double const *i = ann->output + (h? (ann->inputs + ann->hidden * (h-1)): 0);
                        
                        /* Find first weight to this layer. */
                        double *w = ann->weight + (h? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1)): 0);
                        
                        
                        if (k == 0) {
                            w[j*t + k] += d[j] * learning_rate * -1.0;
                        } else {
                            w[j*t + k] += d[j] * learning_rate * i[k-1];
                        }
                    }
                }
                
            }
            h = 0;
            /* Find first delta in this layer. */
            d = ann->delta + (h * ann->hidden);
            
            /* Find first input to this layer. */
            double const *ii = ann->output + (h? (ann->inputs + ann->hidden * (h-1)): 0);
            
            /* Find first weight to this layer. */
            w = ann->weight + (h? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1)): 0);
            
            
            t = ann->inputs + 1;
//#pragma omp for collapse(2)
            for (j = 0; j < ann->hidden; ++j) {
                for (k = 0; k < t; ++k) {
                    if (k == 0) {
                        
                        
                        w[j*t + k] += d[j] * learning_rate * -1.0;
                    } else {
                        w[j*t + k] += d[j] * learning_rate * ii[k-1];
                    }
                }
                
            }
        } //end of for loop I
    } //end of parallel
}


void genann_write(genann const *ann, FILE *out) {
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);
    
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}



