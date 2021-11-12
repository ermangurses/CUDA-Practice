NCXX=nvcc
all:
	$(NCXX) --ptxas-options=-v -o driver 6_occupancy_test.cu 
run:
	./driver
clean:
	/bin/rm driver
