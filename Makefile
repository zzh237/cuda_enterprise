IDIR=./
COMPILER=nvcc

# Since we're using OpenCV, we need to link against its libraries. 
# The NPP and FreeImage libraries can be commented out or removed if they're not used.
#LIBRARIES += -lcudart -lcuda -lnppisu -lnppif -lnppc -lculibos -lfreeimage
LIBRARIES += -lcuda -lcudart -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

COMPILER_FLAGS=-I/usr/local/cuda/include -I/usr/local/cuda/lib64 \
	-I./Common -I./Common/UtilNPP ${LIBRARIES} --std c++17 \
	-g

build: bin/sobel_filter

bin/sobel_filter: src/sobel_filter.cu
	$(COMPILER) $(COMPILER_FLAGS) $< -o $@

clean:
	rm -f bin/*

run:
	./bin/sobel_filter.exe $(ARGS)