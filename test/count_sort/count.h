#ifndef COUNT_H
#define COUNT_H

#include <cstdio>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define MAX_VALUE 256
#define BLOCKS 32
#define THREADS 128

__global__ void count(int*, int*, int);
__global__ void merge(int*);
void print(const char*, int*, int);

#endif