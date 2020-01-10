function [x,f,g,cov,retcode] = CML(fun1,fun2,dataset,x0,algorithm,covPar,options) %dalia: fun1--> lik_fun, fun2-->lik_contri
  %%
  % This function:
  
    % 1. conducts maximum likelihood estimation according to 3 algorithms:
    % fminsearch, fminunc,and patternsearch.
    
    % 2. computes the associated covariance matrix as either the inverse of the
    % Hessian matrix, the inverse of the cross-product of first
    % derivatives, or the Quasi-ML covariance matrix
    
  % INPUTS:
    % fun1:     function handle for the user-written log-likelihood function.
    % fun2:     function handle for the user-written log-likelihood
    %           contributions function.
    % dataset:  the time series for which the parameters should be estimated.
    % x0:       starting values to be used.
    % algorithm:1 - derivative-free approach, minimize the constrained
    %               multivariate function (fminsearch).
    %           2 - gradient-based approach, minimize the unconstrained
    %               multivariate function (fminunc).
    %           3 - global search (patternsearch).
    % covPar:   1 - Hessian-based covairance matrix.
    %           2 - OPG-based covariance matrix.
    %           3 - QML-based covariance matrix
    % options:  set the optimization options using optimset(...).
    
% OUTPUTS:
    % x:        column vector of estimated parameters.
    % f:        value of the ML function at which the optimum.
    % g:        value of the gradient at the estimated parameters.
    % cov:      covariance matrix according to covPar.
    % retcode:  describes the exit condition of the optimization function
    %           used.
   
  %%
  % Conduct ML estimation
  if algorithm == 1
      % Find the minimum of log likelihood function with derivative free
      % algorithm (constrained maximum likelihood CML) (Matlab help says unconstrained-dalia)
      [x,f,retcode,~] = fminsearch(@(x0)fun1(x0,dataset),x0,options);
  elseif algorithm == 2
      % Find the minimum of log likelihood function with gradient-based
      % algorithm (unconstrained maximum likelihood CML)
      [x,f,retcode,~,~,~] = fminunc(@(x0)fun1(x0,dataset),x0,options);
  elseif algorithm == 3
      % Global maximum search
      [x,f,retcode,~] = patternsearch(@(x0)fun1(x0,dataset),x0,[],[],[],[],[],[],[],options); 
  end

  % Redefine a function which only take parameter estimates as input but,
  % take data as known variable
  % In order to achieve this purpose, we provide dataset as input data
  % but define start_v as dependent variable
  fun3 = @(start_v)fun2(start_v,dataset); %dalia: basically evaluate lik_contri at starting values 

  % Call function hessp to compute the Hessian matrix numerically
  Hess_numerical = hessp(fun3,x);
  
  % Compute the Hession-based covariance matrix
  Hesscov = inv(-Hess_numerical);
  
  % Compute the stepwise gradient
  G = gradp(fun3,x);
  
  % Compute the final gradient
  g = mean(G)';
  
  % Compute the OPG-based covariance matrix
  OPGcov = inv(G'*G);
  
  % Compute the QML-based covariance matrix
  QMLcov = inv(Hess_numerical/(G'*G)*Hess_numerical);
  
  % Choose one covariance matrix as output
  if covPar == 1
      cov = Hesscov;
  elseif covPar == 2
      cov = OPGcov;
  elseif covPar == 3
      cov = QMLcov;
  end
     


end