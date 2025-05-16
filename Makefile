EXE = sycl-app

comp: 
	icpx -g -O3 -fsycl -fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda main.cpp -o sycl-app -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

all: $(EXE)

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)

full:comp run

