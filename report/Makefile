FILE=main

pdf:
	latexmk ${FILE}.tex -silent

view:
	open ${FILE}.pdf

clean:
	latexmk -c
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.auxlock *.synctex.gz

nuke:
	latexmk -C
	rm -rf _minted-main .auctex-auto *.log *.run.xml *.txt *.auxlock
