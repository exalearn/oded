#!/bin/bash

# $1 = input_file_name

export outfile='results.out'

python -c "import mocu.scripts.example_dakota; mocu.scripts.example_dakota.main( '$1' , '$outfile' )"
