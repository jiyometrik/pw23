FILE=main

pdf .:
	latexmk ${FILE}.tex -silent -bibtex -lualatex -shell-escape

once 1: 
	latexmk ${FILE}.tex -bibtex -lualatex -shell-escape
	make view

view v:
	open ${FILE}.pdf

clean c:
	latexmk -c

nuke n:
	latexmk -C
