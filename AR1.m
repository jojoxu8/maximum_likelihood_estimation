%Function that returns an AR(1) process according to the input parameters
%given to the function

function[y] = AR1(c, phi, y0, T)
% Innovation vector lenght T generated using random draws from the standard
% normal distribution
epsilon = randn(T, 1);
% Create an output vector using a zero-vector with lenght T
y = zeros(T, 1);
% Create the first entry based on the starting point
y(1,1) = c+phi*y0+epsilon(1,1);
% Loop over the output vector to fill in the entries for the rest of the
% AR(1) process. Starting in position 2 because 1 is already defined 
for i = 2:1:T
    y(i,1) = c +phi*y(i-1,1)+ epsilon(i,1);
end

end