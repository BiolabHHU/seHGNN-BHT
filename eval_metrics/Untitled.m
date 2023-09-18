clf reset

t=0:0.1:10;
plot(t,exp(-10*t).*sin(20*pi*t))
h2=axes('Position',[0.5 0.5 0.3 0.3])
%比如插入x\in [1 2]之间的函数变化
tt=1:0.1:2;                                                       %这里在绘图前直接定义了tt的范围
plot(tt,exp(-10*tt).*sin(20*pi*tt))

