%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OM / GCED / F.-Javier Heredia https://gnom.upc.edu/heredia
% Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xtr,ytr,w_opt,L_opt,tr_acc,Xte,yte,te_acc,niter,tex] = uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)
%
% Input parameters:
%
% num_target : set of digits to be identified.
%    tr_freq : frequency of the digits target in the data set.
%    tr_seed : seed for the training set random generation.
%       tr_p : size of the training set.
%    te_seed : seed for the test set random generation.
%       te_q : size of the test set.
%         la : coefficient lambda of the decay factor.
%       epsG : optimality tolerance.
%       kmax : maximum number of iterations.
%        ils : line search (1 if exact, 2 if uo_BLS, 3 if uo_BLSNW32)
%     ialmax :  formula for the maximum step lenght (1 or 2).
%    kmaxBLS : maximum number of iterations of the uo_BLSNW32.
%      epsal : minimum progress in alpha, algorithm up_BLSNW32
%      c1,c2 : (WC) parameters.
%        isd : optimization algorithm.
%     sg_al0 : \alpha^{SG}_0.
%      sg_be : \beta^{SG}.
%      sg_ga : \gamma^{SG}.
%    sg_emax : e^{SGÇ_{max}.
%   sg_ebest : e^{SG}_{best}.
%    sg_seed : seed for the first random permutation of the SG.
%        icg : if 1 : CGM-FR; if 2, CGM-PR+      (useless in this project).
%        irc : re-starting condition for the CGM (useless in this project).
%         nu : parameter of the RC2 for the CGM  (useless in this project).
%
% Output parameters:
%
%    Xtr : X^{TR}.
%    ytr : y^{TR}.
%     wo : w^*.
%     fo : {\tilde L}^*.
% tr_acc : Accuracy^{TR}.
%    Xte : X^{TE}.
%    yte : y^{TE}.
% te_acc : Accuracy^{TE}.
%  niter : total number of iterations.
%    tex : total running time (see "tic" "toc" Matlab commands).
%


    tic;
    % Training and test data sets
 

    [Xtr, ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq); %Training
    
    [Xte, yte] = uo_nn_dataset(te_seed, te_q, num_target, 0); %Test
    
    
    % Functions
    
    sig = @(X) 1./(1+exp(-X)); %Activation function
    y = @(X,w) sig(w'*sig(X));
    
    
    L = @(w,X,Y,la) (norm(y(X,w)-Y)^2)/size(Y,2)+ (la*norm(w)^2)/2; % Loss function 
    gL = @(w,X,Y,la) (2*sig(X)*((y(X,w)-Y).*y(X,w).*(1-y(X,w)))')/size(Y,2)+la*w;
    
    w_0 = zeros(35, 1);


    hL = [];                    
    ialmin = 0; rho = 0.0; 
    
    % Find the value of w_opt minimizing L
    
    if isd == 1 % Gradient Method
        fprintf('GM Method\n');
        [wk] = uo_nn_GM(w_0, Xtr, ytr, la, L, gL, hL, epsG, kmax, ialmax, ialmin, rho, c1, c2, ils,  kmaxBLS, epsal);
        w_opt = wk(:,end);
    
    elseif isd == 3 % BFGS Method
        fprintf('BFGS Method\n');
        [wk] = uo_nn_BFGS(w_0, Xtr, ytr, la, L, gL, hL, epsG, kmax, ialmax, ialmin, rho, c1, c2, ils, kmaxBLS, epsal);
         w_opt = wk(:,end);
    
    elseif isd == 7 % Stochastic Gradient Method
        fprintf('SGM Method\n');
        [w_opt, wk, k] = uo_nn_SGM(w_0, la, L, gL, Xtr, ytr, Xte, yte, sg_al0, sg_be, sg_ga, sg_emax, sg_ebest, sg_seed);
        
    
    end
    
    L_opt = L(w_opt, Xtr, ytr, la);
    
    % Results of the model 

    % Training accuracy
    train_delta = (round(y(Xtr, w_opt)) == ytr);   
    tr_acc = 100/tr_p * sum(train_delta);
    
    % Test accuracy
    test_delta = (round(y(Xte, w_opt)) == yte);       
    te_acc = 100/te_q * sum(test_delta);
    

    niter = size(wk, 2);

    if isd == 7
        niter = k + 1;
    
    end
    tex=toc;

    % Results print
    %fprintf('L_opt = %6.2d\n', L_opt);
    %fprintf('niter = %1.0f\n', niter);
    %fprintf('Train accuracy = %1.3f\n', tr_acc);
    %fprintf('Test accuracy = %1.3f\n', te_acc);



    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
