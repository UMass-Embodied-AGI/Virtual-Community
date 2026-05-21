UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
    OMP_FLAGS := -Xpreprocessor -fopenmp -I$(shell brew --prefix libomp)/include
    OMP_LIBS  := -L$(shell brew --prefix libomp)/lib -lomp
else
    OMP_FLAGS := -fopenmp
    OMP_LIBS  :=
endif

CXXFLAGS := -fpic -O3 -ffast-math -lpthread -std=c++17 $(OMP_FLAGS)
LDFLAGS  := $(OMP_LIBS)

all:
	g++ $(CXXFLAGS) builder.cpp -o libbuilder.so -shared $(LDFLAGS)
	g++ $(CXXFLAGS) region.cpp  -o libregion.so  -shared $(LDFLAGS)
