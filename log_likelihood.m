% This function gives back the log likelihood function using the
% parameter vector and the AR(1) process as input variables. The first part
% is equal to the log likelihood contribution function

function[logli_function] = log_likelihood(par, y)
log_c=zeros(length(y)-1,1);
for i=2:1:length(y)   
    epsilon = y(i,1) - par(1,1) - par(2,1)*y(i-1,1);
    log_c(i-1,1)= log(1/sqrt(2*pi*par(3,1))) - (epsilon^2)/(2*par(3,1));
end
% Sum up all the likelihood contributions to get the log likelihood
logli_function= sum(log_c);
end