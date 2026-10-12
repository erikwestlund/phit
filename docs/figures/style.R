# Shared documentation style; no statistical estimation lives in this file.
library(ggplot2)
implementation_order <- c('R','statsmodels','SciPy reference','PHP')
implementation_colors <- c(R='#0072B2',statsmodels='#D55E00',`SciPy reference`='#D55E00',PHP='#009E73')
implementation_shapes <- c(R=16,statsmodels=17,`SciPy reference`=17,PHP=15)
concept_colors <- c(reference='#0072B2',conditional='#009E73',approximation='#E69F00',observed='#64748B')
figure_theme <- function() {
 theme_minimal(base_size=12,base_family='sans') + theme(
  text=element_text(color='#253247'),panel.grid.minor=element_blank(),
  panel.grid.major=element_line(color='#E2E8F0',linewidth=.35),
  plot.title=element_text(size=18,face='bold'),plot.subtitle=element_text(size=11,color='#536174'),
  plot.caption=element_text(size=9,color='#536174',hjust=0),plot.title.position='plot',plot.caption.position='plot',
  strip.text=element_text(size=11,face='bold',hjust=0),legend.position='top',legend.title=element_blank(),
  legend.text=element_text(size=10),axis.title=element_text(size=11),
  panel.spacing=grid::unit(1.3,'lines'),plot.margin=margin(12,16,12,12))
}
number_label <- function(x) format(x,big.mark=',',scientific=FALSE,trim=TRUE)
measurement_label <- function(x) vapply(x,function(v) if(is.na(v)) 'not recorded' else if(v==0) '0*' else format(signif(v,3),trim=TRUE,scientific=FALSE),'')
