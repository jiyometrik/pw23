#!/usr/bin/env perl

# timezone
$ENV{'TZ'} = 'Asia/Singapore';

# compilation
$pdflatex = 'lualatex';
$pdf_mode = 1; # only generate the pdf
$max_repeat = 8; # maximum number of runs

# cleaning options
$bibtex_use = 1.5;
