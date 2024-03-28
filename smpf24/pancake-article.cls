\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pancake-article}[2021/09/09 Annotated article class based on Pancake's annotated book class]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass{article}

%* Headings
\RequirePackage{titlesec}

\RequirePackage{xcolor}
\PassOptionsToPackage{usenames, dvipsnames}{xcolor}
\definecolor{accent}{RGB}{0, 0, 90}

\titleformat{\section}[hang]{\normalfont\Large}{\color{accent}\thesection}{0.5em}{}
\titleformat{\subsection}[hang]{\normalfont\large}{\color{accent}\thesubsection}{0.5em}{}
\titleformat{\subsubsection}[hang]{\normalfont\normalsize}{\color{accent}\thesubsubsection}{0.5em}{\color{gray}}

%* Header/footer
\RequirePackage{emptypage}

\RequirePackage{fancyhdr}
\RequirePackage{titling}
\pagestyle{fancy}

\fancyhf{}
\renewcommand{\headrulewidth}{0.75pt}
\renewcommand{\footrulewidth}{0.75pt}
\fancyhead[C]{\@title}
\fancyfoot[L]{\scshape Singapore Mathematics Project Festival 2024}
\fancyfoot[R]{\thepage}

\fancypagestyle{plain}{
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
	\fancyfoot[LO, RE]{\thepage}
}

%* Side notes
\RequirePackage{sidenotes}
\RequirePackage{marginfix}

\setlength{\marginparwidth}{1.5in}
\setlength{\marginparsep}{0.5in}
\setlength{\marginparpush}{0.5in}

\newcommand{\smartragged}{\checkoddpage \ifoddpage\raggedright \else \raggedleft\fi}

\providecommand{\marginaliasize}{\footnotesize}
\DeclareCaptionFont{marginaliasize}{\marginaliasize}
\let\oldmarginpar\marginpar
\def\marginpar#1{\oldmarginpar{\marginaliasize\smartragged #1}}

\DeclareCaptionJustification{smartragged}{\smartragged}

\DeclareCaptionStyle{marginfigure}{font=marginaliasize,justification=smartragged}
\DeclareCaptionStyle{margintable}{font=marginaliasize,justification=smartragged}

\makeatletter
\newcommand{\ifdraft}[2][]{\ifdim\overfullrule>\z@
		\expandafter#2\else\expandafter#1\fi}
\makeatother
\newcommand{\notetoself}[1]{\sidenote{\color{gray}\itshape#1}}

%* Structure enhancements
\RequirePackage{parskip}
\RequirePackage{booktabs}
\RequirePackage{enumitem}
\RequirePackage{soul}

\newlength{\listsep}
\setlength{\listsep}{.5em}

\setlist{topsep=0em, itemsep=\dimexpr\listsep - \parskip\relax}
\setlist[itemize]{label=---}
