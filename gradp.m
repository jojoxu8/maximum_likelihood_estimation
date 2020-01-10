function [grdd] = gradp(f,x0)

% Computation of stepsize (dh) for gradient

f0 = f(x0); 
n = size(f0,1); 
k = size(x0,1); 
grdd = zeros(n,k); 
ax0 = abs(x0); 

if x0 ~= 0 
    dax0 = x0./ax0; 
else
    dax0 = 1; 
end

dh = ((1e-8)*max([ax0,(1e-2)*ones(size(x0,1),1)]').*dax0')'; 
xdh = x0+dh; 
dh = xdh-x0;    % This increases precision slightly  
arg = repmat(x0,1,k);

for i = 1:size(x0,1)
    arg(i,i) = xdh(i); 
end

i = 1; 
while i <= k 
    grdd(:,i) = f(arg(:,i)); 
    i = i+1; 
end 

grdd = (grdd-repmat(f0,1,k))./(repmat(dh',n,1)); 

end

