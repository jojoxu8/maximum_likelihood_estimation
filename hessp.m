% /* Translate from GAUSS code
% ** hessp.src
% ** (C) Copyright 1988-1998 by Aptech Systems, Inc.
% ** All Rights Reserved.
% **
% **
% **
% **> hessp
% **
% **  Purpose:    Computes the matrix of second partial derivatives
% **              (Hessian matrix) of a function defined by a function.
% **
% **  Format:     h = hessp(@f,x0);
% **
% **  Inputs:    @f     pointer to a single-valued function f(x), defined
% **                    as a function, taking a single Kx1 vector
% **                    argument (f:Kx1 -> 1x1).  It is acceptable for
% **                    f(x) to have been defined in terms of global
% **                    arguments in addition to x:
% **
% **                       proc f(x);
% **                           retp( exp(x'b) );
% **                       endp;
% **
% **            x0      Kx1 vector specifying the point at which the Hessian
% **                    of f(x) is to be computed.
% **
% **  Output:   h       KxK matrix of second derivatives of f with respect
% **                    to x at x0. This matrix will be symmetric.
% **
% **  Remarks:    This procedure requires K(K+1)/2 function evaluations. Thus
% **              if K is large it may take a long time to compute the
% **              Hessian matrix.
% **
% **              No more than 3 - 4 digit accuracy should be expected from
% **              this function, though it is possible for greater accuracy
% **              to be achieved with some functions.
% **
% **              It is important that the function be properly scaled, in
% **              order to obtain greatest possible accuracy. Specifically,
% **              scale it so that the first derivatives are approximately
% **              the same size.  If these derivatives differ by more than a
% **              factor of 100 or so, the results can be meaningless.
% */

function h = hessp(f,x0)
 
  % Check for complex input
  if logical(isreal(x0)) == 0
      error('ERROR: Not implemented for complex matrices.');
      return;
  end
  
  % Initializations
  k = size(x0,1);
  hessian = zeros(k,k);
  grdd = zeros(k,1);
  eps = 6.0554544523933429e-6;
  
  % Computation of stepsize(dh)
  ax0 = abs(x0);
  if x0 ~= 0
      dax0 = x0./ax0;
  else
      dax0 = 1;
  end
  dh = eps*max([ax0,(1e-2)*ones(size(x0,1),1)]').*dax0;

  xdh = x0+dh;
  dh = xdh-x0;    % This increases precision slightly 
  ee = eye(k).*dh;
  
 % Computation of f0=f(x0) 
  f0 = sum(f(x0)); 
      
 % Compute forward step/
  i = 1;
  while i <= k
    grdd(i,1) = sum(f(x0+ee(:,i)));
    i = i+1;
  end
  
 % Compute "double" forward step 
  i = 1;
  while i <= k
      j = i;
      while j <= k
          
          hessian(i,j) = sum(f(x0+(ee(:,i)+ee(:,j))));
            if i ~= j
                hessian(j,i) = hessian(i,j);
            end

            j = j+1;
      end
        i = i+1;
  end
  
  % Return the hessian matrix
  h = ( ( (hessian - grdd) - grdd') + f0) ./ (dh.*dh');
  

end