% CatFish: a Bayesian Categorical Variable Multiplicative Poisson Economic Demand Model
% (c) Ed Egan, Petabyte Economics Corp., Jan. 2024. All Rights Reserved.

classdef Dimension < matlab.mixin.Copyable
    % A Dimension object stores a label to e-index table for a dimension.
    % It also provides methods to create complete, degenerate, and scale l2c tables for the dimension.
    % Dimensions are stored in the model.dimensions array and indexed by d.
    
    properties
        d  % Dimension number (index in model.dimensions)
        name % Dimension name
        l2e = table; % Table of dimension labels l (string) and indices e (index_type).
    end

    methods(Sealed) % Array methods
        function indices = indices(objs, train)
            % Return the e_index vectors for array of dimension objects.
            % If more than one dimension, return them in a cell array.
            % Optionally, restrict to train or forecast.
            if nargin < 2 % Train (if specified) 0: forecast, 1: training
                train = [];
            end
            indices = cell(1, length(objs)); 
            for i = 1:length(objs)
               indices{i} = index(objs(i), train);
            end
            if length(indices) == 1
                indices = indices{:}; % Return single index
            end
        end
    end   

    methods 
        function obj = Dimension(d, name)
            % The Dimension constructor function
            obj.d = d;
            if  nargin > 1
                obj.name = name;
            end
        end

        function index = index(obj, train)
            % Get a single dimension's index 
            % Train (if specified) 0: forecast, 1: training
            index = obj.l2e.e;
            if ~isempty(train)
                index = index([obj.l2e.t == train]);
            end
        end               
    end

    methods(Static)
        function l2c = complete(dimensions, normalize)
            % Return a complete l2c table(s) for the given dimension(s) (in a cell array if more than 1)
            % Note that normalize is a string array of labels to be removed.
            if nargin < 2
                normalize = strings(length(dimensions), 1);
            end
            l2c = cell(1, length(dimensions));
            for i = 1:length(dimensions)
                c_col = "c_" + dimensions(i).d;
                l_col = "l_" + dimensions(i).d;
                tbl=table;
                tbl.(l_col) = dimensions(i).l2e.l(dimensions(i).l2e.l ~= normalize(i));
                tbl.(c_col) = dimensions(i).l2e.e(dimensions(i).l2e.l ~= normalize(i)); % Use the e-index
                %tbl.(c_col) = [1:length(l2c.l)]';
                l2c{i} = tbl;
            end
            if length(l2c) == 1 % Don't put a single result in a cell array
                l2c = l2c{1}; 
            end
        end

        function l2c = degenerate(dimensions, normalize)
            % Return a degenerate l2c table(s) for the given dimension(s) (in a cell array if more than 1)
            % Note that normalize is a string array of labels to be removed.
            if nargin < 2
                normalize = strings(length(dimensions), 1);
            end
            l2c = cell(1, length(dimensions));
            for i = 1:length(dimensions)
                c_col = "c_"+ dimensions(i).d;
                l_col = "l_" + dimensions(i).d;
                tbl=table;
                tbl.(l_col) = dimensions(i).l2e.l(dimensions(i).l2e.l ~= normalize(i));
                tbl.(c_col) = ones(length(tbl.(l_col)), 1);
                l2c{i} = tbl;
            end
            if length(l2c) == 1 % Don't put a single result in a cell array
                l2c = l2c{1}; 
            end        
        end

        function l2c = scale(dimensions, params)
            % Return a scale l2c table for a SINGLE dimension in a cell array 
            % params are [scale length, true/false for normalization of first factor]
            labels = dimensions(1).l2e.l;
            step_length = str2double(params(1));
            if length(params) > 1 && ~strcmp(params(2), "false")
                step_one = step_length;
            else
                step_one = 0;
            end
            steps = ceil((length(labels) - step_one) / step_length);
            l2c = cell(1, length(steps));
            c_col = "c_" + dimensions(1).d;
            l_col = "l_" + dimensions(1).d;
            start = step_one + 1;
            for i = 1:steps
                tbl = table;
                tbl.(l_col) = labels(start:end);
                tbl.(c_col) = ones(length(labels(start:end)),1)*i;
                l2c{i} = tbl;
                start = start + step_length;
            end
        end
    end
end

