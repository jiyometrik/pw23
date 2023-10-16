FILE=main

pdf:
	latexmk -bibtex -xelatex ${FILE}.tex
	latexmk -c

view:
	open ${FILE}.pdf

clean:
	latexmk -c
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.auxlock

nuke:
	latexmk -C
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.txt *.auxlock
