#! /bin/bash
julia trainNN.jl sample.NNWDBC.init wdbc.train output.txt -e 1000 -a 0.1
