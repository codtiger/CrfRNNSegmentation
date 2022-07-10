# CrfRNNSegmentation

## Attention:
If tensorflow sources are compiled with `g++` or `clang`, in `cpp/Makefile` this line should be changed:
``` MakeFile
// or clang
CC := g++ 
```

## Usage:
``` bash
cd cpp
make .
cd ..
python3 run_model.py
```