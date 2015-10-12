clc
clear
% input is a .txt file saved under the name "input.txt"
Input_Char = textread('input.txt', '%c');

% preprocess the text and make it into a row of vectors, to figure out the
% number of unique characters
index(1)=Input_Char(1);
Char_Out(1) = 1;
for i = 2 : size(Input_Char,1)
    check = 0;
    for b = 1 : length(index)
        if Input_Char(i) == index(b)
            check = 1;
            Char_Out(i) = b;
        end
    end
    if check == 0
        index(end + 1) = Input_Char(i);
        Char_Out(i) = length(index);
    end
end

Vocab_Size = length(index); % 1 of k encoding, so input dimension is number of different characters
X_Train = Char_Out(1:end-1);
Y_Train = Char_Out(2:end);
% Parmeter input
%--------------------------------------------------------------------------
Char_Size = 25; % lenghth of character to train
Hidden_Size = 100; % number of hidden units
Learning_Rate = 0.01; % learning rate
%--------------------------------------------------------------------------

% Weight Initilization
%--------------------------------------------------------------------------
Wz = rand(Hidden_Size, Vocab_Size)-0.5;
Wi = rand(Hidden_Size, Vocab_Size)-0.5;
Wf = rand(Hidden_Size, Vocab_Size)-0.5;
Wo = rand(Hidden_Size, Vocab_Size)-0.5;
Wy = rand(Vocab_Size, Hidden_Size)-0.5;
Rz = rand(Hidden_Size, Hidden_Size)-0.5;
Ri = rand(Hidden_Size, Hidden_Size)-0.5;
Rf = rand(Hidden_Size, Hidden_Size)-0.5;
Ro = rand(Hidden_Size, Hidden_Size)-0.5;
Pi = zeros(Hidden_Size, 1);
Pf = zeros(Hidden_Size, 1);
Po = zeros(Hidden_Size, 1);
bz = zeros(Hidden_Size, 1);
bi = zeros(Hidden_Size, 1);
bf = zeros(Hidden_Size, 1);
bo = zeros(Hidden_Size, 1);
by = zeros(Vocab_Size, 1);
Z = {}; I = {}; F = {}; O = {}; DZ = {}; DI = {}; DF = {}; DO = {};
SC = {}; SC{1} = zeros(Hidden_Size, 1); DSC = {};
Hout = {}; Hout{1} = zeros(Hidden_Size,1);
Cout = {}; DCout = {};
Yout1 = {}; DYout = {};
Yout2 = {}; DYout2 = {};


%--------------------------------------------------------------------------

% Start Iteration
%--------------------------------------------------------------------------
for k = 1 : 1
    Hout = {}; Hout{1} = zeros(Hidden_Size, 1);
    X = zeros(Vocab_Size, 1);
    X(X_Train(1)) = 1;
    % Forward Pass
    % lstm cell
    Z{end + 1} = Tanh(Wz*X + Rz*Hout{1} + bz);
    I{end + 1} = Sigmoid(Wi*X + Ri*Hout{1} + Pi.*SC{1} + bi);
    F{end + 1} = Sigmoid(Wf*X + Rf*Hout{1} + Pf.*SC{1} + bf);
    SC{end + 1} = Z{1}.*I{1} + SC{1}.*F{1};
    O{end + 1} = Sigmoid(Wo*X + Ro*Hout{1} + Po.*SC{1} + bo);
    Cout{end + 1} = Tanh(SC{1}).*O{1};
    % cell output
    Yout1{end + 1} = Wy*Cout{1} + by; % compute the output, which is of vocab size
    Yout2{end + 1} = softmax(Yout1{1}); % take the softmax
    % compute log loss
    
%     Z = [Z Tanh(Wz*X + Rz*Hout(:,1) + bz)];
%     I = [I Sigmoid(Wi*X + Ri*Hout(:,1) + Pi.*SC(:,1) + bi)];
%     F = [F Sigmoid(Wf*X + Rf*Hout(:,1) + Pf.*SC(:,1) + bf)];
%     SC = [SC Z(:,1).*I(:,1)+SC(:,1).*F(:,1)];
%     O = [O Sigmoid(Wo*X + Ro*Hout(:,1) + Po.*SC(:,1) + bo)];
%     Cout = [Cout Tanh(SC(:,1)).*O(:,1)]; 
    
%     for i = 1 : length(X_Train)
%         %forward pass
%         X = zeros(Input_Dim,1); % create x input
%         X(X_Train(i))= 1; % index x vector with 1 of k encoding
%         Z = [Z Tanh(Wz*X + Rz*Hout + bz)]; % cell input
%         I = [I Sigmoid(Wi*X + Ri*Hout
%     end
end

        

