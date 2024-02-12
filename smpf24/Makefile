FILE=main

pdf .:
	latexmk ${FILE}.tex -silent -bibtex -xelatex -shell-escape

once 1: 
	latexmk ${FILE}.tex -bibtex -xelatex -shell-escape
	make view

view v:
	open ${FILE}.pdf

clean c:
	latexmk -c

nuke n:
	latexmk -C
