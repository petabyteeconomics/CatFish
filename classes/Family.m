% CatFish: a Bayesian Categorical Variable Multiplicative Poisson Economic Demand Model
% (c) Ed Egan, Petabyte Economics Corp., Jan. 2024. All Rights Reserved.

classdef Family < handle
    % A Family is a prior, with a type (1, 2, or 3), that can be applied to categorical variables in one or more blocks.
    % Families of type 2 and 3 add Metropolis-with-Gibbs sampling.
    % The type of the prior determines its methods:
    %   Type 1 is a regular Gibbs prior (alpha, beta). It has no metropolis sampling.
    %   Type 2 is a one hyperparameter (z) Metropolis-with-Gibbs prior (alpha = exp(-2*z)). By default, it is metropolis but not Gibbs sampled.
    %   Type 3 is a two hyperparameter (w,z) Metropolis-with-Gibbs prior (alpha = exp(-2*z), beta = exp(-(w+2*z))). By default, it is both metropolis and Gibbs sampled.

    properties
        f   % index in top.families
        type    % 1,2,3 (note 1-indexing)
        priors = []; % Priors array
        is_gibbs   % t/f for Gibbs sampling
        is_metro   % t/f for Metropolis sampling
        sampling_defaults = logical([1 0; 0 1; 1 1]); % row = type, col = [is_gibbs, is_metro]
        metro = table; % .b, .C, .theta (if C is Gibbs), .in, .scheduled, .u.  (Note: .u is only <>0 if .in and .scheduled)
        metro_in_C2E_C = []; % C from C2E restricted to metro C and train
        metro_in_C2E_E = []; % E from C2E restricted to metro C and train
        % Metropolis-within-Gibbs settings 
        iterations = 3; % Number of iterations
        %  Metropolis-within-Gibbs diagnostics
        diagnostics = true; 
        walks = 0;
        resets = 0;
        accept_walks = 0;
        accept_resets = 0;
    end
 
    methods(Sealed)
        function alphabetas = draw_alphabetas(objs)
            % Return array with length(objs) rows of [alpha, beta] draws from underlying priors.
            alphabetas = zeros(length(objs), 2);
            for i = 1:length(objs)
                family = objs(i);
                if family.type == 1
                    % Type 1 family: alpha,beta are priors (no hierarchy) 
                    alpha = family.priors(1);
                    beta = family.priors(2);
                elseif family.type == 2
                    % Type 2 family: alpha is transformation of z, which is drawn from normal dist.
                    z = normrnd(family.priors(1), family.priors(2)); 
                    alpha = exp(-2*z);
                    beta = alpha;
                elseif family.type == 3
                    % Type 3 family: alpha, beta are transformations of z,w, which are drawn from normal dists.
                    z = normrnd(family.priors(3), family.priors(4));
                    alpha = exp(-2*z);
                    w = normrnd(family.priors(1), family.priors(2)); 
                    beta = exp(-(w+2*z));             
                end
                alphabetas(i,:) = [alpha,beta];           
            end
        end
    end

    methods
        function obj = Family(f, type, priors, sampling)
            % Family constructor. 
            % Notes:
            %   priors is array of length 2 or 4, depending on type
            %   sampling = ([is_gibbs, is_metro])
            if nargin > 0
                obj.f = f;
                obj.type = type;
                obj.priors = priors;
                if nargin < 4
                    sampling = obj.sampling_defaults;
                end
                sampling_indicators = num2cell(sampling(type, :));
                [obj.is_gibbs, obj.is_metro] = sampling_indicators{:};
            end
        end  

        function alphabeta = metropolis_sample(obj, alphabeta, theta, lambda)
            % This is the Metropolis sampling function that updates alphabeta(f,:) for a family. 
            % Note that theta and lambda are unchanged and only used to update alphabeta.
            sum_lambda = accumarray(obj.metro_in_C2E_C, lambda(obj.metro_in_C2E_E)); % metro_in_C2E_C and _E for .in and .scheduled only.
            u = obj.metro.u(obj.metro.in & obj.metro.scheduled);
            outside_thetas = obj.metro.theta(obj.metro.in & ~obj.metro.scheduled);
            numfactors = sum(obj.metro.in);
            curr_alphabeta = alphabeta(obj.f, :);
            if ~isempty(outside_thetas)
                sum_phi = [sum(log(theta(outside_thetas))), sum(theta(outside_thetas)) ];
            else
                sum_phi = [0 0];
            end
            if obj.type == 1
                % Type 1 isn't metropolis sampled. Did you get confused by the change to 1-indexing?
                warning("Family type 1 can't be metropolis sampled");
            elseif obj.type == 2
                % Type 2 samples only z (i.e., alpha). By default the Gibbs sampling is turned off. It is used to transform Poissons to negative binomials.
                [curr_value, curr_d2] = Family.posterior_kernel_z(numfactors, u, sum_lambda, sum_phi, obj.priors, curr_alphabeta);
                curr_z = -0.5*log(curr_alphabeta(1));
                curr_sigma =  1/sqrt(-curr_d2);
                for i = 1:obj.iterations
                    if curr_d2 < 0 % Current kernel is concave
                        if obj.diagnostics
                            obj.walks = obj.walks + 1;
                        end                        
                        prop_z = normrnd(curr_z, curr_sigma); % Random walk
                        prop_alphabeta = [exp(-2*prop_z), exp(-2*prop_z)];      
                        [prop_value, prop_d2] = Family.posterior_kernel_z(numfactors, u, sum_lambda, sum_phi, obj.priors, prop_alphabeta);
                        prop_sigma =  1/sqrt(-prop_d2);
                        if prop_d2 < 0 % Proposal kernal is concave
                            proposal_score = (prop_value - Family.lognormpdf(prop_z, curr_z, curr_sigma)) - ...  
                                             (curr_value - Family.lognormpdf(curr_z, prop_z, prop_sigma));
                        else
                            proposal_score = (prop_value - Family.lognormpdf(prop_z, curr_z, curr_sigma)) - ...  
                                             (curr_value - Family.lognormpdf(curr_z, obj.priors(1), obj.priors(2)));
                        end
                        if log(rand) <=  proposal_score
                            curr_alphabeta = prop_alphabeta;
                            curr_z = prop_z;
                            curr_value = prop_value;
                            curr_d2 = prop_d2; 
                            curr_sigma =  1/sqrt(-curr_d2);
                            if obj.diagnostics
                                obj.accept_walks = obj.accept_walks + 1;
                            end
                        end
                    else % Current kernal isn't concave, so draw from priors to reset.
                        if obj.diagnostics
                            obj.resets = obj.resets + 1;
                        end
                        prop_z = normrnd(obj.priors(1), obj.priors(2)); % Draw from priors
                        prop_alphabeta = [exp(-2*prop_z), exp(-2*prop_z)]; 
                        [prop_value, prop_d2] = Family.posterior_kernel_z(numfactors, u, sum_lambda, sum_phi, obj.priors, prop_alphabeta);
                        prop_sigma =  1/sqrt(-prop_d2);  
                        if prop_d2 < 0 % Proposal kernal is concave
                            proposal_score = (prop_value - Family.lognormpdf(prop_z, obj.priors(1), obj.priors(2))) - ...  
                                             (curr_value - Family.lognormpdf(curr_z, prop_z, prop_sigma));
                        else
                            proposal_score = (prop_value - Family.lognormpdf(prop_z, obj.priors(1), obj.priors(2))) - ...  
                                             (curr_value - Family.lognormpdf(curr_z, obj.priors(1), obj.priors(2)));
                        end        
                        if log(rand) <= proposal_score % Accept the proposed z (alpha)
                            curr_alphabeta = prop_alphabeta;
                            curr_z = prop_z;
                            curr_value = prop_value;
                            curr_d2 = prop_d2;    
                            curr_sigma =  1/sqrt(-curr_d2);
                            if obj.diagnostics
                                obj.accept_resets = obj.accept_resets + 1;
                            end
                        end
                    end
                end
            elseif obj.type == 3
                % Type 3 is a random walk with drift. It samples w,z (i.e., alpha and beta).
                curr_wz = [log(curr_alphabeta(1)/curr_alphabeta(2)), -0.5*log(curr_alphabeta(1))];
                [curr_value, curr_hessian] = Family.posterior_kernel_wz(numfactors, u, sum_lambda, sum_phi, obj.priors, curr_alphabeta);
                for i = 1:obj.iterations
                    if Family.isnegdef(curr_hessian)  % Current kernel is concave
                        if obj.diagnostics
                            obj.walks = obj.walks + 1;
                        end   
                        prop_wz = mvnrnd(curr_wz, -curr_hessian\eye(2));  % Random walk
                        prop_alphabeta = [exp(-2*prop_wz(2)), exp(-(prop_wz(1) + 2*prop_wz(2)))];
                        [prop_value, prop_hessian] = Family.posterior_kernel_wz(numfactors, u, sum_lambda, sum_phi, obj.priors, prop_alphabeta);               
                        if Family.isnegdef(prop_hessian)
                            proposal_score = (prop_value - Family.lognormpdf(prop_wz, curr_wz, curr_hessian)) - ...  
                                            (curr_value - Family.lognormpdf(curr_wz, prop_wz, prop_hessian));
                        else
                            proposal_score = (prop_value - Family.lognormpdf(prop_wz, curr_wz, curr_hessian)) - ...  
                                            (curr_value - (Family.lognormpdf(curr_wz(1), obj.priors(1), obj.priors(2)) + Family.lognormpdf(curr_wz(2), obj.priors(3), obj.priors(4)) ));
                        end
                        if log(rand) <= proposal_score % Accept the proposed w,z (alpha, beta)
                            curr_alphabeta = prop_alphabeta;
                            curr_wz = prop_wz;
                            curr_value = prop_value;
                            curr_hessian = prop_hessian;
                            if obj.diagnostics
                                obj.accept_walks = obj.accept_walks + 1;
                            end
                        end
                    else  % Current kernal isn't concave, so draw from priors to reset.
                        if obj.diagnostics
                            obj.resets = obj.resets + 1;
                        end
                        prop_wz = normrnd( obj.priors([1 3]), obj.priors([2 4]) );  % Draw from priors
                        prop_alphabeta = [exp(-2*prop_wz(2)), exp(-(prop_wz(1) + 2*prop_wz(2)))];            
                        [prop_value, prop_hessian] = Family.posterior_kernel_wz(numfactors, u, sum_lambda, sum_phi, obj.priors, prop_alphabeta);
                        if Family.isnegdef(prop_hessian)
                            proposal_score = (prop_value - ( Family.lognormpdf(prop_wz(1), obj.priors(1), obj.priors(2)) + Family.lognormpdf(prop_wz(2), obj.priors(3), obj.priors(4)) )) - ...  
                                            (curr_value - Family.lognormpdf(curr_wz, prop_wz, prop_hessian));
                        else
                            proposal_score = (prop_value - ( Family.lognormpdf(prop_wz(1), obj.priors(1), obj.priors(2)) + Family.lognormpdf(prop_wz(2), obj.priors(3), obj.priors(4)) )) - ...  
                                            (curr_value - (Family.lognormpdf(curr_wz(1), obj.priors(1), obj.priors(2)) + Family.lognormpdf(curr_wz(2), obj.priors(3), obj.priors(4)) ));
                        end
                        if log(rand) <= proposal_score % Accept the proposed w,z (alpha, beta)
                            curr_alphabeta = prop_alphabeta;
                            curr_wz = prop_wz;
                            curr_value = prop_value;
                            curr_hessian = prop_hessian;
                            if obj.diagnostics
                                obj.accept_resets = obj.accept_resets + 1;
                            end
                        end 
                    end
                end
            end
            alphabeta(obj.f, :) = curr_alphabeta; % Update alphabeta
        end
    end

    methods(Static)
        function tf = isnegdef(A)
            % True/false: 2 x 2 matrix A is negative definite? 
            tf = A(1)<0 && A(4)<0 && A(1)*A(4)>A(2)^2;
        end
        
        function log_p = lognormpdf(x, mu, sigma_or_hessian)
            % Get the density from a log normal kernal given a value, mean and variance/hessian.
            % Note(s): 
            %   If x is [1 1], this is equivalent to log(normpdf) with a bug fix
            %   x and mu can be [1 1] or [1 2], with sigma [1 1] or hessian [2 2], respectively.
            if length(x) == 1
                sigma = sigma_or_hessian;
                log_p = -log(sigma) - 0.5*((x - mu) / sigma)^2 - 0.918938533204673; %-0.5*log(2*pi)
            elseif length(x) == 2
                precision = -sigma_or_hessian;
                log_p = 0.5.*(log(det(precision)) - (x - mu)*precision*(x - mu)') - 1.837877066409345; %-log(2*pi)
            else
                error("Log normal function called with invalid x vector.");
            end
        end

        function [k, dk2_da2] = posterior_kernel_z(numfactors, u, sum_lambda, sum_phi, priors, alphabeta)
            % Get the value k and its second derivitive (dk2_da2) for an alpha from the type 2 posterior kernel
            alpha = alphabeta(1);
            z = -0.5*log(alpha);
            phi = sum_phi(1) + sum_phi(2);
            mu = priors(1);
            sigma = priors(2);
            % Kernel function (k = f + g)
            f = numfactors*( alpha*log(alpha) - gammaln(alpha) ) + ...
                sum( gammaln(alpha+u) - (alpha+u).*log(alpha+sum_lambda) ) + ...
                phi*alpha;
            g = -0.5 * (sigma^-2) * (z-mu)^2;
            k = f + g;
            % Partials         
            da_dz = -2*alpha;
            da2_dz2 = 4*alpha;
            df_da = numfactors*( log(alpha) + 1 - psi(alpha) ) + ...
                    sum( psi(alpha+u) - log(alpha+sum_lambda) - (alpha+u)./(alpha+sum_lambda) ) + ...
                    phi;
            df2_da2 = numfactors*( (1/alpha) - psi(1, alpha) ) + ...
                      sum( psi(1, alpha+u) - (alpha + 2*sum_lambda - u)./(alpha + sum_lambda).^2 );
            dg2_dz2 = -(sigma^-2);
            % Second derivative
            dk2_da2 = df2_da2*da_dz^2 + df_da*da2_dz2 + dg2_dz2;
        end
        
        function [k, H] = posterior_kernel_wz(numfactors, u, sum_lambda, sum_phi, priors, alphabeta)
            % Get the value k and hessian H for an alphabeta from the type 3 posterior kernel
            alpha = alphabeta(1);
            beta = alphabeta(2);
            w = log(alpha/beta);
            z = -0.5*log(alpha);
            priors = num2cell(priors);
            [nu,tau,mu,sigma] = deal(priors{:});
            log_phi = sum_phi(1);
            phi = sum_phi(2);
            % Kernel function (k = f + g)
            f = numfactors*(alpha * log(beta) - gammaln(alpha)) ...
                + sum(gammaln(alpha + u) - (alpha + u).*log(beta + sum_lambda)) ...
                + alpha*log_phi - beta*phi;
            g = -0.5 * ( ((w - nu)^2 * tau^-2) + ((z - mu)^2 * sigma^-2) );
            k = f + g;
            % Partials
            da_dz = -2*alpha;
            da2_dz2 = 4*alpha;
            db_dw = -beta;
            db_dz = -2*beta;
            db2_dw2 = beta;
            db2_dz2 = 4*beta;
            db2_dwdz = 2*beta;
            df_da = numfactors*( log(beta) - psi(alpha) ) + sum( psi(alpha+u) - log(beta+sum_lambda) ) + log_phi;
            df_db = (numfactors*alpha)/beta - sum( (alpha+u) ./ (beta+sum_lambda) ) - phi;
            df2_da2 = -numfactors*psi(1, alpha) + sum( psi(1, alpha+u) );
            df2_db2 = -numfactors*(alpha/beta^2) + sum( (alpha+u)./((beta+sum_lambda).^2) );
            df2_dadb = numfactors/beta - sum( 1./(beta+sum_lambda) );
            dg2_dw2 = -(tau^-2);
            dg2_dz2 = -(sigma^-2);
            % Hessian
            dk2_dw2 =  (df_db * db2_dw2) + (df2_db2 * db_dw^2) + dg2_dw2;
            dk2_dz2 =  (df_da * da2_dz2) + (df_db * db2_dz2) + (df2_da2 * da_dz^2) + ...
                        (df2_db2 * db_dz^2) + (2 * df2_dadb * db_dz * da_dz) + dg2_dz2;
            dk2_dwdz = (df_db * db2_dwdz) + (df2_db2 * db_dw * db_dz) + ...
                        (df2_dadb * db_dw * da_dz); 
            H = [dk2_dw2, dk2_dwdz; dk2_dwdz dk2_dz2];
        end
    end
end