% CatFish: a Bayesian Categorical Variable Multiplicative Poisson Economic Demand Model
% (c) Ed Egan, Petabyte Economics Corp., Jan. 2024. All Rights Reserved.

classdef Block < handle
    % A Block is a Gibbs sampling block: 
    % Every category C in the block is Gibbs sampled at the same time, so each C *must* cover a *disjoint* part of E.  
    % Blocks are stored in the model array and indexed by b.
    % Each block has partitions that collectively cover all D dimensions. 
    % The C2c table maps the overall category C to the partition categories c.

    properties
        b   % block index in model array
        name    % Human readable block name
        partitions = Partition.empty; % Array of partitions
        C2c = table; % Category table (.C, .c_1, ..., .c_d)
        C2f = table; % Family f mapping table (.C, .f)
        C2e = table; % C2e table (unrestricted)
        C2E = []; % Linear C2E vector (unrestricted)
        % Gibbs
        gibbs = table; % Contains .C, .theta, .f, .u and .in
        is_gibbs = false; % t/f for block is Gibbs sampled
        is_gibbs_in = false; % t/f for is Gibbs sampled & has in-sample C
        gibbs_in_C2E_C = []; % C from C2E restricted to Gibbs C and in-sample
        gibbs_in_C2E_E = []; % E from C2E restricted to Gibbs C and in-sample
        gibbs_C_idx = []; % Project gibbs.C to 1:length(gibbs.C)
        % Metro
        scheduled_metro = []; % Array of scheduled metro family indices (to model.families)
    end
 
    methods(Sealed)
        function C = C(objs, f)
            % Get C, optionally for family index(s) f, for array of blocks
            % Note that C doesn't have to be unique across blocks, so the same value could be repeated.
            C = [];
            for i = 1:length(objs)
                if nargin < 2 || isempty(f)
                    C = [C; objs.C2c.C];
                else
                    C = [C; objs(i).C2f.C(ismember(objs(i).C2f.f,f))];
                end
            end
        end
    end

    methods
        function obj = Block(b, partitions, name)
            % Block Constructor
            obj.b = b;
            if  nargin > 2
                obj.name = name;
            end
            if  nargin > 1
                obj.partitions = partitions;
            end            
        end

        function C2c = make_C2c(obj)
            % Make the C2c table for this block as the cartesian product of all partitions' c values
            c_by_dim = arrayfun(@(a) a.c, obj.partitions, 'UniformOutput', false);
            if length(c_by_dim) == 1
            	c_cartesian = c_by_dim;
            else
                % Cartesian product of c_1, ..., c_d 
                c_cartesian = cell(1, numel(c_by_dim));
                [c_cartesian{:}] = ndgrid(c_by_dim{:});
                c_cartesian = cellfun(@(x) reshape(x,[],1), c_cartesian, 'UniformOutput', false);
            end    
            C2c = array2table([c_cartesian{:}], 'VariableNames', obj.partitions.c_cols);
            % Add new C values
            C2c.C = (1:height(C2c))'; 
        end

        function modify_C2f(obj, f, C)
            % Modify the block's C2f table, setting f for each C. 
            % Note that C is a vector and f is a scalar.
            if height(obj.C2f) == 0 % C2f table needs initializing
                obj.C2f.C = obj.C; % Uses the C method with just 1 block, so C is unique.
                obj.C2f.f = zeros(height(obj.C2f.C), 1);
            end
            if nargin < 3 || isempty(C)
                C = obj.C2f.C;
            end
            [Lia, Locb] = ismember(C, obj.C2f.C);
            obj.C2f.f(Locb(Lia)) = f*ones(sum(Lia), 1);
        end

        function C2e = make_C2e(obj)
            % Make an unrestricted (i.e., for all C and e_d) C2e table
            % Note: Using innerjoin is a performance bottleneck, especially when the data gets big.
            c2e = obj.partitions.get_c2e;
            C2e = obj.C2c;
            for i = 1:length(c2e)
                C2e = innerjoin(C2e, c2e{i});
            end
            c_cols = C2e.Properties.VariableNames(contains(C2e.Properties.VariableNames,"c_"));
            C2e = removevars(C2e, c_cols);
        end   
    end

    methods(Static)
            function theta = gibbs_sample(alphabeta)
            % Gibbs sample for new thetas from alphabetas (conditional posterior is gamma)
            shape = alphabeta(:, 1);
            scale = 1./alphabeta(:, 2);
            theta = gamrnd(shape, scale);
            if any(theta == 0)
                warning("Drew a zero theta (most likely because hyperparameters weren't sane).\n");
            end
        end     
    end
end
