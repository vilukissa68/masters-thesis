;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("tunithesis" "12pt" "a4paper" "finnish")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("lastpage" "") ("babel" "finnish") ("biblatex" "backend=biber" "style=numeric-comp" "citestyle=numeric-comp" "autocite=footnote" "maxbibnames=5") ("csquotes" "") ("booktabs" "") ("adjustbox" "") ("subcaption" "") ("caption" "") ("svg" "") ("acronym" "withpage") ("tikz" "") ("xcolor" "") ("amsfonts" "") ("listings" "") ("listings-rust" "") ("graphicx" "") ("float" "") ("multirow" "") ("calc" "")))
   (add-to-list 'LaTeX-verbatim-environments-local "lstlisting")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "lstinline")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "lstinline")
   (TeX-run-style-hooks
    "latex2e"
    "tunithesis"
    "tunithesis12"
    "lastpage"
    "babel"
    "biblatex"
    "csquotes"
    "booktabs"
    "adjustbox"
    "subcaption"
    "caption"
    "svg"
    "acronym"
    "tikz"
    "xcolor"
    "amsfonts"
    "listings"
    "listings-rust"
    "graphicx"
    "float"
    "multirow"
    "calc")
   (TeX-add-symbols
    '("todo" 1)
    '("fixthis" 1)
    "angs"
    "checkmark"
    "scalecheck")
   (LaTeX-add-labels
    "ch:introduction")
   (LaTeX-add-bibliographies
    "../thesis/thesis_refs"
    "../thesis/zotero")
   (LaTeX-add-listings-lstdefinestyles
    "customc"
    "customasm")
   (LaTeX-add-xcolor-definecolors
    "tunipurple"))
 :latex)

