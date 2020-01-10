%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     Maximum Likelihood Estimation
%                          of an AR(1) Process                     
%                              29.11.2019                                                      
%                   Advanced Time Series Analysis       
%                           Jolanda Duenner
%                            Robert Neulen
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start by clearing workspace and command window as well as setting the
% seed

clear
clc
close all

seed = 24;
rng(seed);

%% Task 1.1

% Creating an AR(1) process using the function "AR1" and the parameters given
% in the excersise ( c=5, phi=0.7,starting point in 17 and T=100)
[ar1] = AR1(5, 0.7, 17, 100);

% Calculate the theoretical mean to include in the graph
theo_mean = 5/(1-0.7);

% Plot the AR(1) process with its theoretical mean
handle=figure;
% Define the X-axis with lengh T=100
    X1 = (1:100)';
    plot(X1, ar1, "b");
    yline(theo_mean,"-.r");
    title('AR(1): y_{t} = 5 + 0.7y_{t-1} + \epsilon_{t}');
    xlabel('time');
    ylabel('y');
    legend("AR(1)", "Theoretical mean");
saveas(gcf,'AR1','jpg');

%% Task 2.1
% Creation of the log likelihood contribution function:
% log_likelihood_contr

%% Task 2.2
% Create the log likelihood function that sums up all the log likelihood
% contributions

% Evaluate the log likelihood function using the respective parameter
% vector [c phi var] and the AR(1) process from Task 1

% a.) 
par_a = [5 0.7 1]';
[log_likelihood_a] = log_likelihood(par_a, ar1);

% b.)
par_b = [-5 0.1 0.25]';
[log_likelihood_b] = log_likelihood(par_b, ar1);

%% Task 3.1

% Create the function handles for the two settings a.) and b.). As the
% optimization functions used in this excercise calculate the minimum of a
% function, we multiply our log likelihood funciton by (-1) to get the maximum
func_a = @(par_a)log_likelihood_adj(par_a, ar1);
func_b = @(par_b)log_likelihood_adj(par_b, ar1);

% Using three different optimization functions (fminunc, fminsearch und
% patternsearch), we estimate the parameters of the AR(1) process using
% the given parameters for a.) and b.) as starting values

% a.)
[max_fminunc_a]=fminunc(func_a, par_a);
[max_fminunc_b]=fminunc(func_b, par_b);
% b.)
[max_fminsearch_a]=fminsearch(func_a, par_a);
[max_fminsearch_b]=fminsearch(func_b, par_b);
% c.)
[max_patternsearch_a]=patternsearch(func_a, par_a);
[max_patternsearch_b]=patternsearch(func_b, par_b);

%% Task 3.2

% Create a vector containing the results from the log likelihood function
% with varying phi (-1 until 2) which includes the true value of 0.7 as
% well as phi's given in a.) and b.)

% Create a vector with the different values for phi (-1 to 2 in 0.1 steps).
% We expect phi to be in the range of -1 to 1, but for a better overview we
% use -1 to 2
range_phi = (-1:0.1:2)';
% Vector holdings the log likelihood for each phi (temporar vector)
temp_logli_vec_a = zeros(length(range_phi),1);
temp_logli_vec_b = zeros(length(range_phi),1);

% Create vectors that contain all estimations for c and all estimations for
% the variance in the order fminunc, fminsearch, pattersearch
c_est_a = [max_fminunc_a(1,1) max_fminsearch_a(1,1) max_patternsearch_a(1,1)]';
var_est_a = [max_fminunc_a(3,1) max_fminsearch_a(3,1) max_patternsearch_a(3,1)]';
c_est_b = [max_fminunc_b(1,1) max_fminsearch_b(1,1) max_patternsearch_b(1,1)]';
var_est_b = [max_fminunc_b(3,1) max_fminsearch_b(3,1) max_patternsearch_b(3,1)]';


% Open a matrix which is later filled with the calculated log
% likelihood values for different phi's (from -1 to 1) for the different
% estimated values c and var. The first column is filled with the values
% using the estimations from minunc, the second using fminsearch and the
% last one with the estimated values from patternsearch
matrix_a=zeros(length(range_phi),3);
matrix_b=zeros(length(range_phi),3);

% The outer loop is filling the three columns 1 to 3 for matrix_a and
% matrix_b with the vectors coming from the inner loop, containing the log
% likelihood for varying phi's. For an exact description of the varables
% and temporary variables, see variable list in the appendix
for j=1:1:3
    c_hat_a = c_est_a(j,1);
    c_hat_b = c_est_b(j,1);
    var_hat_a = var_est_a(j,1);
    var_hat_b = var_est_b(j,1);
    for i=1:1:length(range_phi)
        temp_par_a = [c_hat_a range_phi(i,1) var_hat_a]';
        temp_par_b = [c_hat_b range_phi(i,1) var_hat_b]';
        [temp_logli_scalar_a] = log_likelihood(temp_par_a,ar1);
        [temp_logli_scalar_b] = log_likelihood(temp_par_b,ar1);
        temp_logli_vec_a(i,1)= temp_logli_scalar_a;
        temp_logli_vec_b(i,1)= temp_logli_scalar_b;
    end
    matrix_a(:,j) = temp_logli_vec_a;
    matrix_b(:,j) = temp_logli_vec_b;
end

% Plot the results from matrix_a and also the estimated phi for the respective
% estimator and the true value phi=0.7
% The following figures do not have a saveas() command because of a scalling 
% problem in the resulting .jpg. Hence we saved the following two figures 
% manually

% Graph for a.)
figure;
subplot(1,3,1);
    plot(range_phi, matrix_a(:,1), "b");
    title('fminunc','FontSize',17);
    xline(max_fminunc_a(2,1),"r");
    xline(0.7,"-.k");
    xlabel('\phi','FontSize',17);
    ylabel('Log Likelihood','FontSize',17);
    legend("Log Likelihood profile", "Estimated value", "True value", "location", "southoutside",'FontSize',14);
subplot(1,3,2);
    plot(range_phi, matrix_a(:,2), "b");
    xline(max_fminsearch_a(2,1),"r");
    xline(0.7,"-.k");
    title('fminsearch','FontSize',17);
    xlabel('\phi','FontSize',17);
    legend("Log Likelihood profile", "Estimated value", "True value", "location", "southoutside",'FontSize',14);
subplot(1,3,3);
    plot(range_phi, matrix_a(:,3), "b");
    xline(max_patternsearch_a(2,1),"r");
    xline(0.7,"-.k");
    title('patternsearch','FontSize',17);
    xlabel('\phi','FontSize',17);
    legend("Log Likelihood profile", "Estimated value", "True value", "location", "southoutside",'FontSize',14);
sgt = sgtitle('Log Likelihood profile with starting value c=5, \sigma^2=1 and varying \phi');
sgt.FontSize = 16;

% Graph for b.)
figure;
subplot(1,3,1);
    plot(range_phi, matrix_b(:,1), "b");
    title('fminunc','FontSize',17);
    xline(max_fminunc_b(2,1),"r");
    xline(0.7,"-.k");
    xlabel('\phi','FontSize',17);
    ylabel('Log Likelihood','FontSize',17);
    legend("Log Likelihood profile", "Estimated value", "True value", "location", "southoutside",'FontSize',14);
subplot(1,3,2);
    plot(range_phi, matrix_b(:,2), "b");
    xline(max_fminsearch_b(2,1),"r");
    xline(0.7,"-.k");
    title('fminsearch','FontSize',17);
    xlabel('\phi','FontSize',17);
    legend("Log Likelihood profile", "Estimated value", "True value", "location", "southoutside",'FontSize',14);
subplot(1,3,3);
    plot(range_phi, matrix_b(:,3), "b");
    xline(max_patternsearch_b(2,1),"r");
    xline(0.7,"-.k");
    title('patternsearch','FontSize',17);
    xlabel('\phi','FontSize',17);
    legend("Log Likelihood profile", "Estimated value", "True value", "location", "southoutside",'FontSize',14);
sgt = sgtitle('Log Likelihood profile with starting value c=-5, \sigma^2=0.25 and varying \phi');
sgt.FontSize = 16;

%% Task 4

% Setting up the toolbox CML that is used for the rest of the assignment
options = optimset('Display','iter','TolX',10 ^(-40),'TolFun',10^(-40) ,'MaxIter',10^10 , 'MaxFunEvals', 100000);

%% Task 4.1

% Use the CML toolbox to perform the parameter estimation. We use
% fminsearch as our optimization function and the inverse of Hessian to
% calculate the covariance matrix
[cml_est_fmin_a,~,~,cml_cov_hes_a] = CML(@log_likelihood_adj, @log_likelihood_contr, ar1, par_a, 1, 1,options);
[cml_est_fmin_b,~,~,cml_cov_hes_b] = CML(@log_likelihood_adj, @log_likelihood_contr, ar1, par_b, 1, 1,options);

%% Task 4.2

% Calculate the standard errors for all the estimates by creating a new
% vector with the diagonal values from the covariance matrix and take the
% square root. We use the values from a.) as the parameters are equal to the
% true parameters
se_hessian_vec = sqrt(diag(cml_cov_hes_a));

% Test if phi = 0.7. We use 1.96 as the critical value for the 95%
% confidence interval of a two sided test (phi/2 = 0.25). As it is a two 
% sided test, we use the absolute value of the numerator
test_stat =abs((cml_est_fmin_a(2,1)-0.7))/se_hessian_vec(2,1);
if test_stat > 1.96
        text_test_42 = ['The value of the t statistic is ', num2str(test_stat), ' which is larger than 1.96'];
        text_test_42_2 = "reject H0 that phi is equal to 0.7";
else
    text_test_42 =['The value of the t statistic is ', num2str(test_stat), ' which is smaller than 1.96'];
    text_test_42_2 = "We don't reject H0 that phi is equal to 0.7";
end


%% Task 4.3
%  Compute the 95% convidence interval for phi by calculating the lower bound
%  and upper bound and display the results in the command window
conf_interval_low = (cml_est_fmin_a(2,1)-1.96*se_hessian_vec(2,1));
conf_interval_high = (cml_est_fmin_a(2,1)+1.96*se_hessian_vec(2,1));
text_conf_43 =['The confidence interval for phi is: [',num2str(conf_interval_low), ';', num2str(conf_interval_high), ']'];

%% Task 4.4

% Calculate the standard errors again but using the inverse of the
% cross-product of the first derivative and Quasi-ML covariance matrix to
% calculate the covariance matrix
[cml_est_fmin_OPG,~,~,cml_cov_OPG] = CML(@log_likelihood_adj, @log_likelihood_contr, ar1, par_a, 1, 2,options);
[cml_est_fmin_QML,~,~,cml_cov_QML] = CML(@log_likelihood_adj, @log_likelihood_contr, ar1, par_a, 1, 3,options);
se_OPG_vec = sqrt(diag(cml_cov_OPG));
se_QML_vec = sqrt(diag(cml_cov_QML));

%% Task 4.5
% We increase T from 100 to 50000 in the AR(1) process and recalculate the
% time series
[ar1_alt] = AR1(5, 0.7, 17, 50000);

% Use the toolbox again to estimate the parameters using fminsearch
[cml_est_fmin_alt,~,~,cml_cov_hes_alt] = CML(@log_likelihood_adj, @log_likelihood_contr, ar1_alt, par_a, 1, 1,options);

% Calculate the standard errors for all the estimates
se_hessian_vec_alt = sqrt(diag(cml_cov_hes_alt));

% Test if phi = 0.7
test_stat_alt =abs((cml_est_fmin_alt(2,1)-0.7))/se_hessian_vec_alt(2,1);
if test_stat_alt > 1.96
        text_test1 = ['The value of the t statistic is ', num2str(test_stat_alt), ' which is larger than 1.96'];
        text_test1_2 = "reject H0 that phi is equal to 0.7";
else
    text_test1 = ['The value of the t statistic is ', num2str(test_stat_alt), ' which is smaller than 1.96'];
    text_test1_2 = "We don't reject H0 that phi is equal to 0.7";
end

% Calculate the confidence interval for the newly estimated parameters as
% in 4.3
conf_interval_low_alt = (cml_est_fmin_alt(2,1)-1.96*se_hessian_vec_alt(2,1));
conf_interval_high_alt = (cml_est_fmin_alt(2,1)+1.96*se_hessian_vec_alt(2,1));
text_conv1 =['The confidence interval for phi is: [',num2str(conf_interval_low_alt), ';', num2str(conf_interval_high_alt), ']'];


% Calculate the standard errors again as done in 4.4
[cml_est_fmin_OPG_alt,~,~,cml_cov_OPG_alt] = CML(@log_likelihood_adj, @log_likelihood_contr, ar1_alt, par_a, 1, 2,options);
[cml_est_fmin_QML_alt,~,~,cml_cov_QML_alt] = CML(@log_likelihood_adj, @log_likelihood_contr, ar1_alt, par_a, 1, 3,options);
se_OPG_vec_alt = sqrt(diag(cml_cov_OPG_alt));
se_QML_vec_alt = sqrt(diag(cml_cov_QML_alt));

%% Task 4.6

% Now, we reduce T again to 300 but simulate 100 ensembles of the AR(1)
% process. We first prepare the matrix that will hold the t-statistic (first
% column), the lower bound of the confidence interval (second column) and
% the upper bound of the conf. interval (third column).
matrix_ensemble = zeros(100,3);

% Using a for loop, we simulate a world that is reset 100 times. The
% results are placed in the matrix "matrix_ensemble" in each iteration. The
% test statistic is calculated as in 4.2 and the confidence interval as in
% task 4.4
for i=1:1:100
    [ar1_ens] = AR1(5, 0.7, 17, 300);
    [cml_est_fmin_ens,~,~,cml_cov_hessian_ens] = CML(@log_likelihood_adj, @log_likelihood_contr, ar1_ens, par_a, 1, 1,options);
    se_hessian_ens = sqrt(cml_cov_hessian_ens(2,2));
    test_stat_ens =(cml_est_fmin_ens(2,1)-0.7)/se_hessian_ens;
    matrix_ensemble(i,1)=test_stat_ens;
    conf_interval_low_ens = (cml_est_fmin_ens(2,1)-1.96*se_hessian_ens);
    conf_interval_high_ens = (cml_est_fmin_ens(2,1)+1.96*se_hessian_ens);
    matrix_ensemble(i,2)=conf_interval_low_ens;
    matrix_ensemble(i,3)=conf_interval_high_ens;
end
 
%% Task 4.7

% Count in how many of the ensembles phi = 0.7 is in the con?dence interval.
% We use a for loop with a counter that increases by one if the condition
% that 0.7 is in the interval is true
counter = 0;
for i = 1:1:100
    if matrix_ensemble(i,2) <= 0.7 && 0.7 <= matrix_ensemble(i,3)
        counter = counter +1;
    end
end

text_conf_47=['Out of the 100 ensembles, ',num2str(counter), ' have 0.7 in the confidence interval.'];

%% Task 4.8

% Generate a kernel density estimate using the t-statisct from the 100
% ensembles (task 4.6).
[dens,xi] = ksdensity(matrix_ensemble(:,1));

% Create the probability density function (pdf) of the standard normal 
% distribution evaluated at xi
xn = normpdf(xi,0,1);

% Plot the kernel density and the standard normal density in one plot
figure;
    p = plot(xi, dens, "r", xi, xn, "-.b");
    p(1).LineWidth=1.2;
    p(2).LineWidth=0.6;
    title('Kernel Density');
    xlabel('t-statistics');
    ylabel('Density');
    legend("Kernel Density", "Standard normal density","location","northwest");
saveas(gcf,'kdensity','jpg');

%% Appendix Graphs

% This graph is used in the appendix of the paper to show the long term
% development of the AR(1) process

% Generate the ar1 from task 1 again but with T = 500
% (c=5, phi=0.7,starting point in 17 and T=500)
[ar1_500] = AR1(5, 0.7, 17, 500);

% Plot the AR(1) process with its theoretical mean
figure;
% Define the X-axis with lengh T=500
    X2 = (1:500)';
    plot(X2, ar1_500, "b");
    yline(theo_mean,"-.r");
    title('AR(1) with the true values and T = 500');
    xlabel('time');
    ylabel('y');
    legend("AR(1)", "Theoretical mean");
saveas(gcf,'AR1_500','jpg');


%% Output
clc
format short
disp("________________________________________________________________________________")
disp("----- Task 2.2 -----");
disp("Result of the conditional log likelihood function for a.)");
disp(log_likelihood_a);
disp("Result of the conditional log likelihood function for b.)");
format shortG
disp(log_likelihood_b);

disp("________________________________________________________________________________")
disp("----- Task 3.1 -----");
disp("fminunc [c phi var] for a.) and b.):");
format short
disp((max_fminunc_a)');
disp((max_fminunc_b)');
disp("fminsearch (c phi var) for a.) and b.):");
disp((max_fminsearch_a)');
disp((max_fminsearch_b)');
disp("patternsearch (c phi var) for a.) and b.):");
disp((max_patternsearch_a)');
disp((max_patternsearch_b)');

disp("________________________________________________________________________________")
disp("----- Task 4.1 -----");
disp("The estimated parameters for a.) using the toolbox [c phi var]':")
format short
disp(cml_est_fmin_a);
disp("The estimated parameters for b.) using the toolbox [c phi var]':")
disp(cml_est_fmin_b);

disp("________________________________________________________________________________")
disp("----- Task 4.2 -----");
disp("Standard error using 'Inverse of Hessian', displayed in the order [c phi var]'");
disp(se_hessian_vec);

disp(text_test_42);
disp(text_test_42_2);

disp("________________________________________________________________________________")
disp("----- Task 4.3 -----");
disp(text_conf_43);

disp("________________________________________________________________________________")
disp("----- Task 4.4 -----");
disp("Standard errors with T=100");
disp("Standard error using 'Inverse of Hessian', displayed in the order [c phi var]'");
disp(se_hessian_vec);
disp("Standard error using 'Inverse of cross-product of derivatives', displayed in the order [c phi var]'");
disp(se_OPG_vec);
disp("Standard error using 'Quasi-ML covariance matrix', displayed in the order [c phi var]'");
disp(se_QML_vec);

disp("________________________________________________________________________________")
disp("----- Task 4.5 -----");
disp("The estimated parameters for the AR(1) process with T = 50000 using the toolbox:")
disp(cml_est_fmin_alt);
disp("_ _ _ _ _ _")
disp("Test phi=0.7:")
disp(text_test1);
disp(text_test1_2);
disp("_ _ _ _ _ _")
disp("Confidence interval:")
disp(text_conv1);
disp("_ _ _ _ _ _")
disp("Standard errors with T=5000");
disp("Standard error using 'Inverse of Hessian', displayed in the order [c phi var]'");
disp(se_hessian_vec_alt);
disp("Standard error using 'Inverse of cross-product of derivatives', displayed in the order [c phi var]'");
disp(se_OPG_vec_alt);
disp("Standard error using 'Quasi-ML covariance matrix', displayed in the order [c phi var]'");
disp(se_QML_vec_alt);

disp("________________________________________________________________________________")
disp("----- Task 4.7 -----");
disp(text_conf_47);

