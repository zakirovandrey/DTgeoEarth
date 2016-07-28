set size square
unset key
plot [-1:31][-1:181] for [i=5:360:2] 'sensors.dat' using ($0*2.5/60.0):(sqrt(abs(column(i)))*(column(i)<0?-1:1)+0.5*i) w l lc 0
pause -1
