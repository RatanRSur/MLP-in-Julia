#! /bin/bash
julia trainNN.jl ../sample.NNWDBC.init ../wdbc.train ../output.txt -e 100 -a 0.1
