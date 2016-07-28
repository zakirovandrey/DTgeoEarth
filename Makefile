#ARCH ?= #k100#geocluster #gpupc1 #D
#USE_AIVLIB_MODEL ?= 1
#MPI_ON ?= 1
#USE_DOUBLE ?= 1

ifeq      ($(COMP),k100)
ARCH := sm_20
else ifeq ($(COMP),gpupc1)
ARCH := sm_35
else ifeq ($(COMP),geocluster)
ARCH := sm_50
else ifeq ($(COMP),D)
ARCH := sm_35
else ifeq ($(COMP),ion)
ARCH := sm_50
else ifeq ($(COMP),supermic)
ARCH := sm_61
else
ARCH := sm_35
endif

ifdef MPI_ON
ifeq ($(COMP),k100)
GCC  ?= /usr/mpi/gcc/mvapich2-1.5.1-qlc/bin/mpicc
else
GCC  ?= mpic++
endif
else
GCC  ?= g++
endif

ifeq      ($(COMP),k100)
NVCC := /common/cuda-6.5/bin/nvcc 
else ifeq ($(COMP),geocluster)
NVCC := nvcc
else ifeq ($(COMP),gpupc1)
NVCC := nvcc
else ifeq ($(COMP),D)
#NVCC := /home/zakirov/cuda-7.5/bin/nvcc 
NVCC := /home/zakirov/progs/cuda-8.0rc/bin/nvcc 
else ifeq ($(COMP),ion)
NVCC := /mnt/D/home/zakirov/cuda-7.5/bin/nvcc
else ifeq ($(COMP),supermic)
NVCC := /usr/local/cuda/bin/nvcc
else
NVCC := nvcc
endif 
GENCODE_SM := -arch=$(ARCH)
#NTIME = 0
ALL_DEFS := NP NS NA NB NV NTIME DYSH MPI_ON USE_AIVLIB_MODEL USE_DOUBLE

CDEFS := $(foreach f, $(ALL_DEFS), $(if $($f),-D$f=$($f)))

# internal flags
NVCCFLAGS   := -ccbin $(GCC) -O3 -Xptxas="-v" #-Xcudafe "--diag_suppress=declared_but_not_referenced,set_but_not_used"
CCFLAGS     := -O3 -fopenmp -fPIC $(CDEFS) 
NVCCLDFLAGS :=
LDFLAGS     := -L./ -L/home/zakirov/cuda/lib64 -L/usr/mpi/gcc/mvapich2-1.5.1/lib/ -L/common/cuda-6.5/lib64/ -L/home/zakirov/cuda-7.5/lib64

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_NVCCLDFLAGS ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?= #-std=c++11

ifeq ($(COMP),k100)
INCLUDES  := -I/usr/mpi/gcc/mvapich2-1.5.1/lib/include/ -I./png/
LIBRARIES := -lmpich -lcudart -lglut -lGL -lcufft -lpng -lgomp -lpthread
else ifeq ($(COMP),geocluster)
INCLUDES  := -I/usr/mpi/gcc/mvapich-1.2.0/include/
LIBRARIES := -lcudart -lGL -lcufft -lpng -lgomp -lpthread -lpyaiv2
ifdef MPI_ON
LIBRARIES := -lmpich $(LIBRARIES)
else
LIBRARIES := -lglut $(LIBRARIES)
endif
else
INCLUDES  := 
LIBRARIES := -lcudart -lglut -lGL -lcufft -lpng -lgomp -lpthread
endif

ifdef USE_AIVLIB_MODEL
LIBRARIES := $(LIBRARIES) -laiv
endif

################################################################################
GENCODE_SM20  := #-gencode arch=compute_20,code=sm_21
GENCODE_SM30  := #-gencode arch=compute_30,code=sm_30
GENCODE_SM35  := #-gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50  := #-arch=sm_20
GENCODE_FLAGS := $(GENCODE_SM50) $(GENCODE_SM35) $(GENCODE_SM30) $(GENCODE_SM20) $(GENCODE_SM)
ALL_CCFLAGS   := --compiler-options="$(CCFLAGS) $(EXTRA_CCFLAGS)" 
ALL_LDFLAGS   := --linker-options="$(LDFLAGS) $(EXTRA_LDFLAGS)"
################################################################################

# Target rules
all: build

DTgeo_wrap.cxx: geo.i params.h py_consts.h texmodel.cuh surfcut.cuh
	swig -python -c++ -o DTgeo_wrap.cxx $<
DTgeo_wrap.o: DTgeo_wrap.cxx params.h py_consts.h texmodel.cuh surfcut.cuh
	$(GCC) $(INCLUDES) $(CCFLAGS) -c $< -fPIC -I/usr/include/python/ -I/usr/include/python2.7/ -I/usr/include/python2.6/ -I./python2.6/ -o $@
#_DTgeo.so: DTgeo_wrap.o cudaDTgeo.so
ifdef USE_AIVLIB_MODEL
_DTgeo.so: DTgeo_wrap.o cudaDTgeo.so spacemodel/src/space_model.o spacemodel/src/middle_model.o
	$(GCC) $(INCLUDES) $(CCFLAGS) -Wl,-rpath=./ -L./ $(LDFLAGS) $< cudaDTgeo.so -o $@ -shared
else
_DTgeo.so: DTgeo_wrap.o cudaDTgeo.so 
	$(GCC) $(INCLUDES) $(CCFLAGS) -L./ $(LDFLAGS) $< cudaDTgeo.so -o $@ -shared
endif

build: DTgeo _DTgeo.so

kerTFSF.o: kerTFSF.inc.cu cuda_math.h params.h py_consts.h texmodel.cuh surfcut.cuh copyrags.cuh defs.h signal.hpp
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
kerTFSF_pmls.o: kerTFSF_pmls.inc.cu cuda_math.h params.h py_consts.h texmodel.cuh surfcut.cuh copyrags.cuh defs.h signal.hpp
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
kerITFSF.o: kerITFSF.inc.cu cuda_math.h params.h py_consts.h texmodel.cuh surfcut.cuh copyrags.cuh defs.h signal.hpp
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
kerITFSF_pmls.o: kerITFSF_pmls.inc.cu cuda_math.h params.h py_consts.h texmodel.cuh surfcut.cuh copyrags.cuh defs.h signal.hpp
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

#AsyncTYPES := D_pmls S_pmls I_pmls X_pmls DD_pmls D S I X DD
AsyncTYPES := D_pmls S_pmls Is_pmls Id_pmls Xs_pmls Xd_pmls D S Is Id Xs Xd
obj_files = $(foreach a,$(AsyncTYPES), ker$a.o)

ker%.o: ker%.inc.cu
ker%.o: %.inc.cu
$(obj_files): cuda_math.h params.h py_consts.h defs.h copyrags.cuh texmodel.cu surfcut.cuh 
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $(subst .o,.inc.cu,$@)

dt.o: DTgeo.cu diamond.cu drop.cu im3D.hpp im2D.h cuda_math.h params.h py_consts.h texmodel.cuh surfcut.cuh init.h signal.hpp window.hpp
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

TEXMODEL_DEPS := texmodel.cu texmodel.cuh params.h py_consts.h cuda_math.h
ifdef USE_AIVLIB_MODEL
TEXMODEL_DEPS := $(TEXMODEL_DEPS) spacemodel/include/access2model.hpp
obj_files := spacemodel/src/space_model.o spacemodel/src/middle_model.o $(obj_files)
endif

SURFCUT_DEPS := surfcut.cu surfcut.cuh params.h py_consts.h cuda_math.h
texmodel.o: $(TEXMODEL_DEPS)
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
surfcut.o: $(SURFCUT_DEPS)
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<
DTgeo: surfcut.o texmodel.o dt.o im3D.o kerTFSF.o kerTFSF_pmls.o kerITFSF.o kerITFSF_pmls.o $(obj_files)
ifndef USE_AIVLIB_MODEL
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LDFLAGS) -o $@ $+ $(LIBRARIES)
endif
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(ALL_LDFLAGS) $(GENCODE_FLAGS) $(LDFLAGS) -o cudaDTgeo.so $+ $(LIBRARIES) --shared

im3D.o: im3D.cu im3D.hpp cuda_math.h fpal.h im2D.h
	$(EXEC) $(NVCC) $(NVCCFLAGS) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -dc $<

cudaDTgeo.so: DTgeo
#DTgeo: texmodel.o dt.o im3D.o kerTFSF.o kerTFSF_pmls.o $(obj_files)

generate:
	python genDT.py

run: build
	$(EXEC) ./DTgeo

clean:
	$(EXEC) rm -f dt.o surfcut.o texmodel.o im3D.o ker*.o DTgeo _DTgeo.so cudaDTgeo.so DTgeo_wrap* DTgeo.py DTgeo.pyc

clobber: clean

