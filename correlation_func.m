function C0 = correlation_func(d,r)
% Compactly supported 5th order piecewise rational function from eq 4.10 in
% Gaspari & Cohn, Construction of correlation functions in two and three
% dimensions, 1999
%   Detailed explanation goes here
dr = d/r;

if ( (0 <= d) && (d <= r))
    C0 = -0.25*dr^5 + 0.5*dr^4 + (5/8)*dr^3 - 5/3*dr^2 + 1;
elseif ( (r < d) && (d < 2*r) )
    C0 = (1/12)*dr^5 - 0.5*dr^4 + (5/8)*dr^3 + (5/3)*dr^2 - 5*dr + 4 - (2/3)*(1/dr);
else
    C0 = 0;
end
        
    
    
end

