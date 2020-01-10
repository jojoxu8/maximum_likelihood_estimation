% This function gives back the log likelihood contributions using the
% parameter vector and the AR(1) process as input variables

function[log_c] = log_likelihood_contr(par, y)
% Create an output vector using a zero-vector with the lengh of y minus 1.
% We lose the first observation as it is a conditional log likelihood 
log_c=zeros(length(y)-1,1);
% Create a loop that calculates the epsilon for the current iteration step
% and hands it to the log likelihood contribution formula
for i=2:1:length(y)
    % Epsilons are calcuated using the formula: e(t) = y(t) - c - phi*y(t-1)   
    epsilon = y(i,1) - par(1,1) - par(2,1)*y(i-1,1);
    % Calculate each log likelihood contribution. The loop starts at i=2 but we
    % want the first entry to be at (1,1), so we use (i-1,1)
    log_c(i-1,1)= log(1/sqrt(2*pi*par(3,1))) - (epsilon^2)/(2*par(3,1));
end

end