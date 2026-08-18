# Usage:
# make        # compile all binary
# make clean  # remove ALL binaries and objects
# $@ --> target , $< first prerequisite , $^ --> prerequisites

.PHONY = all clean

CC = g++ -std=c++23		# compiler to use
LIBS = -lm	# external libraries
OPT = -O3 -march=native -mtune=native -fopenmp
CFLAGS = -Wall # -g	# Use -g just for debug			# compiler flags: all warnings + debugger meta-data

BINS := Ecosystem_Dynamics_on_RRG_zero_Temperature    # There are three options here, uncomment the one you want. Default is the RRG zero temperature.
# BINS := Ecosystem_Dynamics_on_DirectedER_zero_Temperature
# BINS := Ecosystem_Dynamics_on_Graphs_in_Temperature
MAIN_SRCS := $(BINS).cpp
# Ecosystem_Dynamics_on_Graphs.cpp
TOOLBOX_SRCS := basic_toolbox.cpp \
            generate_graph_from_sequence_toolbox.cpp \
            ecosystem_numerical_integration_toolbox.cpp
SRCS := $(MAIN_SRCS) $(TOOLBOX_SRCS)
OBJS := $(SRCS:%.cpp=%.o)
#BINS := $(SRCS:%.c=%)

all: $(BINS)

# This default rule compiles the executable program
$(BINS): $(OBJS)
	@echo "Checking.."
#	@echo $(BINS)
#	@echo $(OBJS)
#	@echo $(SRCS)
	$(CC) $^ $(OPT) $(CFLAGS) $(LIBS) -o $@
	chmod a+x $@

# This rule compiles each module into its object file
%.o: %.cpp
	@echo "Creating object.."
	$(CC) $(OPT) $(CFLAGS) -c $< -o $@ $(LIBS)

clean:
	@echo "Cleaning up..."
	rm -rvf *.o $(BINS)
