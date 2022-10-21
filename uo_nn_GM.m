function [wk] = uo_nn_GM(w_0, Xtr, ytr, la, L, gL, hL, epsG, kmax, ialmax, ialmin, rho, c1, c2, ils, kmaxBLS, epsal)
    k = 0;
    wk = [w_0]; 
    w = w_0; 
    iWk =[]; 
    ioutk = [];
    dk = []; 
    alk = [];

    while norm(gL(w, Xtr, ytr, la)) >= epsG & k < kmax
        d = -gL(w, Xtr, ytr, la); dk = [dk, d]; 

        if k == 0                                    
            almax = 1;
        else 
            if ialmax == 1
                almax = al*(gL(wk(:,end - 1), Xtr, ytr, la)'*dk(:,end-1))/(gL(w, Xtr, ytr, la)'*d);
            elseif ialmax == 2
                 almax = 2*(L(w, Xtr, ytr, la)-L(wk(:,end-1),Xtr, ytr, la))/(gL(w, Xtr, ytr, la)'*d);
            end
        end

        [al, iWout, iout] = s_length(w, Xtr, ytr, la, L, gL, hL, d, almax, ialmin, rho, c1, c2, ils,kmaxBLS, epsal);                     
        w = w + d*al;                                
        wk = [wk, w];  
        alk = [alk, al]; 
        iWk = [iWk, iWout]; 
        ioutk = [ioutk, iout];
        k = k + 1;
    end
end


function [al, iWout, iout] = s_length(w, Xtr, ytr, la, L, gL, hL, d, almax, ialmin, rho, c1, c2, ils,  kmaxBLS, epsal)
                
    iWout = [];
    
    if ils == 0
        al = -(gL(w)'*d)/(d'*hL(w)*d); 
        iWout = 5; 

    elseif ils == 1
        al = almax; 
        [satisfywolfe, iWout] = WOLFE(w, al, d, L, gL, c1, c2, ils);
        while ~satisfywolfe & al >= ialmin
            al = rho*al; 
            [satisfywolfe, iWout] = WOLFE(w, al, d, L, gL, c1, c2, ils);
        end

    elseif ils == 3
        [al,iout] = uo_BLSNW32(@(w) L(w, Xtr, ytr, la), @(w) gL(w, Xtr, ytr, la), w, d, almax, c1, c2, kmaxBLS, epsal);
    end
end

function [satisfywolfe, iWout] = WOLFE(x, al, d, f, g, c1, c2, iW)
    iWout = 0;
    satisfywolfe = false;
    if f(x + al*d) <= f(x) + c1*g(x)'*d*al
        iWout = 1;
        if iW == 1 
            if g(x + al*d)'*d >= c2*g(x)'*d 
                iWout = 2;
                satisfywolfe = true;
            end
        elseif iW == 2 
            if abs(g(x + al*d)'*d) <= c2*abs(g(x)'*d)
                iWout = 3;
                satisfywolfe = true;
            end
        end
    end
end
