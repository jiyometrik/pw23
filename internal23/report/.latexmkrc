#!/usr/bin/env perl

# timezone
$ENV{'TZ'} = 'Asia/Singapore';

# compilation
$pdflatex = 'lualatex -shell-escape';
$pdf_mode = 1; # only generate the pdf
$max_repeat = 15; # maximum number of runs

# cleaning options
$bibtex_use = 1.5;
