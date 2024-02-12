\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{pancake-article}[2021/09/09 Annotated article class based on Pancake's annotated book class]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass{article}

%* Headings
\RequirePackage{titlesec}
% \titleformat{\chapter}[hang]{\normalfont\Huge}{\chaptertitlename\ \thechapter\hspace{.5em}\vrule width .7pt}{.5em}{}
\titleformat{\section}[hang]{\bfseries\normalfont\Large}{\color{Blue}\thesection}{0.5em}{}
\titleformat{\subsection}[hang]{\normalfont\large\itshape}{\color{Blue}\thesubsection}{0.5em}{}
\titleformat{\subsubsection}[hang]{\normalfont\normalsize\itshape}{\color{Blue}\thesubsubsection}{0.5em}{\color{gray}}

\setcounter{tocdepth}{2} % don't show subsubsections in the table of contents

%* Header/footer
\RequirePackage{emptypage}

\RequirePackage{fancyhdr}
\RequirePackage{titling}
\pagestyle{fancy}

\fancyhf{}
\renewcommand{\headrulewidth}{0.8pt}
\renewcommand{\footrulewidth}{0.8pt}
\fancyhead[L]{\@title}
\fancyhead[R]{Group XX}
\fancyfoot[C]{\thepage}

\fancypagestyle{plain}{
	\fancyhf{}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
	\fancyfoot[C]{\thepage}
}

%* Side notes
\RequirePackage{sidenotes}
\RequirePackage{marginfix}

\setlength{\marginparwidth}{5em}
\setlength{\marginparsep}{1em}
\setlength{\marginparpush}{1em}

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
