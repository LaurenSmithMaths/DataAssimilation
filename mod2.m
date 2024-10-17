%%%%%%%%%%%
% mod function starting at third input, like mathematica
%%%%%%%%%%%%%

function y = mod2(a,b,start)

y = mod(a-start,b) + start;