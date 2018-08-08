ARCH ?= x86
CROSS_COMPILE ?= 
#--------------------------------------------------
build_objects = main
build_libs = libeis.so
#--------------------------------------------------
ifeq ($(ARCH), x86)
INC_PATH += -I./include -I/opt/OpenCV/include -I/opt/Eigen/include
LIB_PATH += -L./lib -L/opt/OpenCV/lib -L.
LIB      += -lOpenCL -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio
CFLAGS   += -fPIC -Wall -g
CPPFLAGS += -fPIC -Wall -g
LDFLAGS  += -lpthread -Wl,-rpath=./
endif

ifeq ($(ARCH), arm)
INC_PATH += 
LIB_PATH += 
LIB      += 
CFLAGS   +=  
CPPFLAGS += 
LDFLAGS  += 
endif
#--------------------------------------------------
CFLAGS   += 
CPPFLAGS += -std=c++11 
LDFLAGS  += -std=c++11 

ifeq ($(ARCH), x86)
CC       = gcc
CPP      = g++
LINK     = g++
endif

ifeq ($(ARCH), arm)
CC       = $(CROSS_COMPILE)gcc
CPP      = $(CROSS_COMPILE)g++
LINK     = $(CROSS_COMPILE)g++
endif

build_libs_link = $(build_libs:lib%.so=-l%)
#---------------------------------------------------
all: $(build_libs) $(build_objects)
	rm *.o

main : main.o
	$(LINK) $^ $(LIB_PATH) $(LIB) $(build_libs_link) $(LDFLAGS) -o $@

libeis.so : eis.o
	$(LINK) -shared $^ $(LIB_PATH) $(LIB) $(LDFLAGS) -o $@

%.o:%.c
	$(CC) -c $< $(INC_PATH) $(CFLAGS) -o $@

%.o:%.cpp
	$(CPP) -c $< $(INC_PATH) $(CPPFLAGS) -o $@

.PHONY: clean
clean:
	rm -f *.o $(build_libs) $(build_objects)

