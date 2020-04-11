function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
prediction = X*theta;
Absoluteprediction = 1./(1+exp(-prediction));
J_initial = y'*log(Absoluteprediction)+ (1-y)'*log(1-Absoluteprediction);
J_initial2 = -J_initial/m;

regularizedValueForJ_initial = 0;

thetaSquare = theta.^2;


thetaSum = sum(thetaSquare) - theta(1,1).^2;

regularizedValueForJ_initial = (lambda*thetaSum)/(2*m);


J = J_initial2 + regularizedValueForJ_initial;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

initial_gradient = zeros(size(theta));
initial_gradient = (Absoluteprediction - y).*X;
initial_gradient = sum(initial_gradient)/m;


temp= (lambda/m);

temp1= temp*theta';

initialgradient2 =  initial_gradient + temp1;

initialgradient2(1,1) = initialgradient2(1,1) - (lambda/m)*theta(1,1);



grad = initialgradient2;







% =============================================================

end
