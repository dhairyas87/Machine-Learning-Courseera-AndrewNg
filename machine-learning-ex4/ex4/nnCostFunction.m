function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

A1 = [ones(m, 1) X];
Z2 = A1 * Theta1';
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
Z3 = A2*Theta2';
H = A3 = sigmoid(Z3);


 
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :); % for classification probelems for each and every input it has to be classified into number of outputs and not just single input.
end

J_initial = Y.*log(H)+ (1-Y).*log(1-H); %element wise multiplication that is why it 

J_initial = sum(J_initial) ; %um of all the columns;


J = -sum(J_initial)/m;



%Now including the Lambda expression for the input vectors


addtionalRegularized =  sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)) - (sum(Theta1(:,1).^2) + sum(Theta2(:,1).^2)); %subtracting the first column of the theta as its values should not be included.

%above equation can also be written as 
%penalty = (lambda/(2*m))*(sum(sum(Theta1(:, 2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));
regularizedTerm = lambda*(addtionalRegularized) / (2*m);

J = J + regularizedTerm;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%%Implementing backpropogation algorithm


del3 = A3 - Y;
del2 = (del3*Theta2 .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]))(:, 2:end);


D1 = del2'*A1;
D2 = del3'*A2;


grad1 = D1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
grad2 = D2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [grad1(:) ; grad2(:)];


end
