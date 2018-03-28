#
# Student makefile for Cache Lab
# Note: requires a 64-bit x86-64 system 
#
CC = gcc
OMPFLAGS = -g -Wall -std=c99 -m64 -fopenmp
#CFLAGS = -g -Wall -std=c99 -m64


all: serialomp
	
serialomp:
	$(CC) $(OMPFLAGS) -o serialomp serialomp.c

#csim: csim.c cachelab.c cachelab.h
#	$(CC) $(CFLAGS) -o csim csim.c cachelab.c -lm 

#test-trans: test-trans.c trans.o cachelab.c cachelab.h
#	$(CC) $(CFLAGS) -o test-trans test-trans.c cachelab.c trans.o 

#tracegen: tracegen.c trans.o cachelab.c
#	$(CC) $(CFLAGS) -O0 -o tracegen tracegen.c trans.o cachelab.c

#trans.o: trans.c
#	$(CC) $(CFLAGS) -O0 -c trans.c

#
# Clean the src dirctory
#
clean:
	rm -f serialomp
	rm -rf *.o
	rm -f *.tar
	rm -f *.tmp
