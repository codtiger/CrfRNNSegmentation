#  -----------------------------------------------------------------------------------------------------------------------
#  Instructions:
#
#  1.  Activate your Tensorflow virtualenv before running this script. Before running this script, 'python' command in the
#      terminal should refer to the Python interpreter associated to your Tensorflow installation.
#
#  2.  Run 'make', it should produce a new file named 'high_dim_filter.so'.
#
#  3.  If this script fails, please refer to https://www.tensorflow.org/extend/adding_an_op#build_the_op_library for help.
#
#  -----------------------------------------------------------------------------------------------------------------------

# Define the compiler
# if tensorflow sourced are compiled with 
CC := clang

# Read Tensorflow paths
TF_INC := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LIB := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

# Is the Tensorflow version >= 1.4?
TF_VERSION_GTE_1_4 := $(shell expr `python -c 'import tensorflow as tf; print(tf.__version__)' | cut -f1,2 -d.` \>= 1.4)

# Flags required for all cases
CFLAGS := -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0 -shared -fPIC ${TF_INC} -O2 

# Set a special flag if we are on macOS
ifeq ($(shell uname -s), Darwin)
	CFLAGS += -undefined dynamic_lookup
endif

# Set some more flags if the Tensorflow version is >= 1.4
ifeq ($(TF_VERSION_GTE_1_4), 1)
    # CFLAGS += ${TF_INC}/external/nsync/public
	LDFLAGS := ${TF_LIB} 
else
	LDFLAGS :=
endif

# Define build targets
.PHONY: all clean

high_dim_filter.so: high_dim_filter.cc modified_permutohedral.cc
	$(CC) $(CFLAGS) -o high_dim_filter.so high_dim_filter.cc modified_permutohedral.cc $(LDFLAGS)

clean:
	$(RM) high_dim_filter.so

all: high_dim_filter.so
