NVCC = nvcc
NVCCFLAGS = -c -O2 --compiler-bindir /usr/bin
LIB = -lm

all: count clean-o

clean: clean-o
	rm -rf count

clean-o:
	rm -rf *.o

count: count.o
	$(NVCC) -o $@ $^ $(LIB)

count.o: count.cu count.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<
