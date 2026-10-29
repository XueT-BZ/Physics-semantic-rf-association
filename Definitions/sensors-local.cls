% Offline, Sensors-only compatibility class, not the unmodified publisher class.
% Layout parameters follow the MDPI legacy pdfTeX template retrieved 2026-10-15:
% https://github.com/MDPI-AG/luatex-template/tree/main/legacy-pdftex-version
% The manuscript automatically prefers official Definitions/mdpi.cls if supplied.
% Publication DOI, editorial dates and unrelated journal options are omitted.
% A logo asset supplied in the user project is retained on the preview title page.
% See TEMPLATE_NOTE.md for provenance and differences.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Definitions/sensors-local}[2026/10/15 Offline Sensors compatibility]
\DeclareOption*{}
\ProcessOptions\relax
\LoadClass[10pt,a4paper]{article}
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{mathpazo,amsmath,amssymb,amsthm,graphicx,booktabs,array,tabularx}
\RequirePackage{float,caption,multirow,enumitem,calc,changepage,ifthen,etoolbox}
\RequirePackage{titlesec,microtype,xcolor,fancyhdr,lastpage,indentfirst}
\RequirePackage[sort&compress,numbers]{natbib}
\RequirePackage[left=5.87cm,marginparwidth=4.1cm,marginparsep=.51cm,right=1.27cm,
 top=1.27cm,bottom=1.08cm,footskip=36pt,headsep=24pt,includehead,includefoot]{geometry}
\RequirePackage[unicode,breaklinks=true,hidelinks]{hyperref}
\newlength{\extralength}\setlength{\extralength}{4.61cm}
\newlength{\fulllength}\setlength{\fulllength}{18.46cm}
\linespread{1.165}
\setlength{\parindent}{.75cm}
\setlength{\parskip}{0pt}
\clubpenalty10000\widowpenalty10000\displaywidowpenalty10000
\tolerance1000\emergencystretch1em
\titleformat{\section}{\raggedright\bfseries\fontsize{12}{12}\selectfont}{\thesection.}{.4em}{}
\titleformat{\subsection}{\raggedright\itshape\fontsize{10}{12}\selectfont}{\thesubsection.}{.4em}{}
\titleformat{\subsubsection}{\raggedright\fontsize{10}{12}\selectfont}{\thesubsubsection.}{.4em}{}
\titleformat{\paragraph}[block]{\raggedright\bfseries\fontsize{10}{12}\selectfont}{}{0pt}{}
\titlespacing{\section}{0pt}{12pt}{3pt}
\titlespacing{\subsection}{0pt}{12pt}{3pt}
\titlespacing{\subsubsection}{0pt}{12pt}{3pt}
\titlespacing{\paragraph}{0pt}{8pt}{3pt}
\captionsetup{labelfont={bf,small},labelsep=period,textfont=small,singlelinecheck=false,justification=justified}
\captionsetup[figure]{position=bottom,aboveskip=6pt,belowskip=0pt}
\captionsetup[table]{position=top,aboveskip=6pt,belowskip=4pt}
\renewcommand\tabularxcolumn[1]{m{#1}}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{C}{>{\centering\arraybackslash}X}
\setlist{topsep=3pt,parsep=0pt,itemsep=1pt,leftmargin=*}
\newcommand{\Title}[1]{\gdef\@Title{#1}}
\newcommand{\Author}[1]{\gdef\@Author{#1}}
\newcommand{\AuthorNames}[1]{}
\newcommand{\address}[1]{\gdef\@address{#1}}
\newcommand{\corres}[1]{\gdef\@corres{#1}}
\renewcommand{\abstract}[1]{\gdef\@abstracttext{#1}}
\newcommand{\keyword}[1]{\gdef\@keywordtext{#1}}
\newcommand{\firstpage}[1]{\gdef\@firstpage{#1}}
\newcommand{\pubvolume}[1]{}\newcommand{\issuenum}[1]{}
\newcommand{\articlenumber}[1]{}\newcommand{\pubyear}[1]{}
\newcommand{\copyrightyear}[1]{}\newcommand{\datereceived}[1]{}
\newcommand{\daterevised}[1]{}\newcommand{\dateaccepted}[1]{}
\newcommand{\datepublished}[1]{}\newcommand{\doinum}[1]{}
\newcommand{\reftitle}[1]{\renewcommand{\refname}{#1}}
\newcommand{\backstatement}[2]{\par\vspace{6pt}\noindent{\fontsize{9}{11.2}\selectfont\textbf{#1:} #2\par}}
\newcommand{\authorcontributions}[1]{\backstatement{Author Contributions}{#1}}
\newcommand{\funding}[1]{\backstatement{Funding}{#1}}
\newcommand{\institutionalreview}[1]{\backstatement{Institutional Review Board Statement}{#1}}
\newcommand{\informedconsent}[1]{\backstatement{Informed Consent Statement}{#1}}
\newcommand{\dataavailability}[1]{\backstatement{Data Availability Statement}{#1}}
\newcommand{\acknowledgments}[1]{\backstatement{Acknowledgments}{#1}}
\newcommand{\conflictsofinterest}[1]{\backstatement{Conflicts of Interest}{#1}}
\newcommand{\supplementary}[1]{\backstatement{Supplementary Materials}{#1}}
\renewcommand{\maketitle}{%
 \thispagestyle{firstpage}%
 \begin{adjustwidth}{-\extralength}{0cm}
 \noindent\hfill\includegraphics[width=16mm]{Definitions/mdpi-logo-cropped-from-user-template.png}\par\vspace{7pt}
 \noindent\rule{\fulllength}{.45pt}\par\vspace{11pt}
 {\noindent\itshape\fontsize{10}{12}\selectfont Article\par}\vspace{6pt}
 {\noindent\fontsize{18}{21}\selectfont\bfseries\@Title\par}\vspace{11pt}
 {\noindent\bfseries\@Author\par}\vspace{9pt}
 {\noindent\fontsize{9}{11}\selectfont\@address\par\@corres\par}\vspace{13pt}
 \end{adjustwidth}
 \noindent\textbf{Abstract}\par\vspace{4pt}
 \noindent\@abstracttext\par\vspace{8pt}
 \noindent\textbf{Keywords:} \@keywordtext\par\vspace{8pt}\hrule\vspace{12pt}
}
\AtBeginDocument{\maketitle}
% Published-article-style preview: no reviewer line numbers.
\pagestyle{fancy}\fancyhf{}
\fancyheadoffset[L]{\extralength}
\fancyfootoffset[L]{\extralength}
\fancyhead[L]{\fontsize{8}{9}\selectfont Manuscript prepared for \textit{Sensors}}
\fancyhead[R]{\fontsize{8}{9}\selectfont\thepage{} of \pageref*{LastPage}}
\fancyfoot[L]{\fontsize{8}{9}\selectfont Manuscript prepared for \textit{Sensors}, 2026}
\renewcommand{\headrulewidth}{.4pt}
\fancypagestyle{firstpage}{\fancyhf{}\fancyfoot[L]{\fontsize{8}{9}\selectfont Manuscript prepared for \textit{Sensors}, 2026}\fancyfoot[R]{\fontsize{8}{9}\selectfont Research manuscript}\renewcommand{\headrulewidth}{0pt}}
\fancypagestyle{plain}{\fancyhf{}\fancyhead[L]{\fontsize{8}{9}\selectfont Manuscript prepared for \textit{Sensors}}\fancyhead[R]{\fontsize{8}{9}\selectfont\thepage{} of \pageref*{LastPage}}\renewcommand{\headrulewidth}{.4pt}}
\renewcommand{\bibfont}{\fontsize{9}{11}\selectfont}
\setlength{\bibsep}{0pt}
\renewcommand{\@biblabel}[1]{#1.}
\endinput
