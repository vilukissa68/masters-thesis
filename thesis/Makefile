render-xelatex:
	xelatex main.tex
	biber main
	xelatex main.tex
	xelatex main.tex

render-pdflatex:
	pdflatex main.tex
	biber main
	pdflatex main.tex
	pdflatex main.tex

render-no-biber:
	xelatex main.tex


clean:
	rm main.aux main.bbl main.bcf main.blg main.log main.out main.run.xml main.tex.bbl main.tex.blg main.toc texput.log
