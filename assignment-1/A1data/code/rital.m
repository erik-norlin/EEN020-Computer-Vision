function rital(linjer,st)
% rital takes as input a 3xn matrix "linjer"
% where each column
% represents the hom. coordinates of a 2D line.
% It then plots those lines. Use "hold on" before
% rital to see all the lines.
% The optional second argument "st" controls the
% line style of the plot.
% NOTE: rital is dependent on a function pflat
% which divides each column of an mxn matrix by
% its last element and outputs the obtained
% normalized matrix.
if nargin == 1,
 st='-';
end;
if size(linjer)==0,
  slask=[];
else
  [slask,nn]=size(linjer);
  rikt=psphere([linjer(2,:);-linjer(1,:);zeros(1,nn)]);
  punkter=pflat(cross(rikt,linjer));
  for i=1:nn;
   plot([punkter(1,i)-2000*rikt(1,i) punkter(1,i)+2000*rikt(1,i)], ...
        [punkter(2,i)-2000*rikt(2,i) punkter(2,i)+2000*rikt(2,i)],st);
  end;
  slask=[];
end;
