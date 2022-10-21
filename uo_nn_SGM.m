function [w_opt, wk, k] = uo_nn_SGM(w_0, la, L, gL, Xtr, ytr, Xte, yte, sg_al0, sg_be, sg_ga, sg_emax, sg_ebest, sg_seed)
    %seeds
    if ~isempty(sg_seed), rng(sg_seed); end                 

    p = size(Xtr, 2);                                       % Training columns
    m = floor(sg_ga * p);                                   
    sg_kmax = sg_emax * ceil(p/m);                          % Max iterations
    sg_al = 0.01 * sg_al0; 
    sg_k = floor(sg_be * sg_kmax);                          % Step length parameters;
    best_Loss = inf;                                        
    w_opt = w_0; 
    wk = [w_0]; 
    w = w_0;                      
    e = 0;  s = 0; k = 0;                                

    while(e <= sg_emax && s < sg_ebest && k < sg_kmax)
        perm_Xtr = randperm(p);

        % minibatch
        for i = 0:ceil((p/m) - 1) 
            Minibatch = perm_Xtr(m*i + 1 : min(m*i + m, p));            
            Xtr_mini = Xtr(:,Minibatch); ytr_mini = ytr(Minibatch);                                              
            
            d = -gL(w, Xtr_mini, ytr_mini, la);                         
            
            if k <= sg_k                                               
                al = (1 - (k/sg_k))*sg_al0 + (k/sg_k)*sg_al;
            else
                al = sg_al;
            end
            w = w + al*d; wk = [wk, w]; k = k + 1;                     
        end

        %epoch
        e = e + 1;                                                     
        Lte_value = L(w, Xte, yte, la); 
        if Lte_value < best_Loss
            best_Loss = Lte_value;
            w_opt = w;                                                  
            s = 0;
        else
            s = s + 1;
        end
    end
end 