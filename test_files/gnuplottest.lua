require 'gnuplot'

gnuplot.setterm('x11')
--gnuplot.pngfigure('gnuplot.png')

LondonTemp = torch.Tensor{{9, 10, 12, 15, 18, 21, 23, 23, 20, 16, 12, 9},
                          {5,  5,  6,  7, 10, 13, 15, 15, 13, 10,  7, 5}}
gnuplot.plot({'High [°C]',LondonTemp[1]},{'Low [°C]',LondonTemp[2]})
gnuplot.raw('set xtics ("Jan" 1, "Feb" 2, "Mar" 3, "Apr" 4, "May" 5, "Jun" 6, "Jul" 7, "Aug" 8, "Sep" 9, "Oct" 10, "Nov" 11, "Dec" 12)')
gnuplot.plotflush()
gnuplot.axis{0,13,0,''}
gnuplot.grid(true)
gnuplot.title('London average temperature')