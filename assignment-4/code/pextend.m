function xut = pextend(x);
if isa(x,'structure') | isa (x,'imagedata'),
  x=getpoints(x);
end

[sx,sy]=size(x);
xut=[x;ones(1,sy)];