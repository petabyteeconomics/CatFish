% CatFish: a Bayesian Categorical Variable Multiplicative Poisson Economic Demand Model
% (c) Ed Egan, Petabyte Economics Corp., Jan. 2024. All Rights Reserved.

classdef Model < matlab.mixin.Copyable
    %   The "CatFish" model object provides methods to assemble, do inference on, and analyze a ...
    %   Bayesian categorical variable, multiplicative Poisson economic model with hierarchical priors.
    %   The inference uses Gibbs, and optionally Metropolis-within-Gibbs, Markov chain Monte Carlo sampling.
    %   The model supports any number and specification of categorical variable across any number and type of dimensions.
    %
    %   Model.m requires the following other classes, and stores objects of their type in arrays:
    %       Dimension.m
    %       Partition.m
    %       Block.m - Gibbs sampling
    %       Family.m - Metropolis sampling
    %
    %   The model is run using a script with the following steps (see example1.m):
    %       1) Instantiate a model object.
    %       2) Define dimensions using vectors of labels.
    %       3) Define partitions that assign labels (from one or more dimensions) to categories.
    %       4) Create blocks of categorical variables from partitions' categories. 
    %           Note that most blocks use a single user-defined partition for one or two dimensions
    %           and automatically-created degenerate partitions for other dimensions. 
    %       5) Assign priors to categories within categorical variables using families.
    %       6) Load the training data.
    %       7) Do the in-sample inference.
    %       8) Do any out-of-sample inference (draws from possibly hierarchical priors).
    %       9) Save the inference results (thetas and alphabetas).
    %       10) Optionally create lambda vectors for forecasting.

    properties
        name % Model name 
        verbose = false; % Turn on/off status messages
        folder; % Folder for data files
        % Topology object arrays:
        dimensions = Dimension.empty; % Dimension objects
        partitions = Partition.empty; % Partition objects
        blocks = Block.empty; % Block objects
        families = Family.empty; % Family objects. F = length(obj.families)
        % Datatypes
        index_datatype = 'double';
        variable_datatype = 'double';
        flag_datatype = 'logical';
        label_datatype = 'string';
        % Linear sampler
        sz  % Array of dimension sizes (ordered)
        is_train % Logical array of train true/false
        Thetas = 0; % Number of Gibbs sampled variables
        % Factor info tables
        gibbs = table;
        metro = table;
        % Training data
        u = table; % .u, .l_1, ..., .l_d, (loaded) and .e_1, ..., .e_d (from dimensions.l2e)
        U = []; % Linear U vector (of size E)
        % Sampler parameters
        M = 8; % chains
        R = 16; % iterations
        S = 64; % inner loops
        % Autotailor settings
        autotailor = struct('on',true, 'params',"thetas", 'W_max',0.25, 'indices',[]); % Note that empty indices array doesn't initiate due to MATLAB bug
        % Inference cache and storage
        chain_alphabetas = []; % [M x F x 2 (alpha,beta)] Cached alphabetas for M chains
        chain_thetas = []; % [M x Thetas] Cached thetas for M chains
        chain_lambdas = []; % [M x E] Cached lambdas for M chains
        alphabetas = []; % Alphabetas storage: % R x M x F X 2
        thetas = []; % Thetas storage: R x M x Thetas
    end
    
    methods
        function obj = Model(name, settings)
            % Model Constructor
            if  nargin > 0
                obj.name = name;
                if nargin == 2 % Load the provided settings
                    fields = fieldnames(settings);
                    for i = 1:length(fields)
                        try 
                            obj.(fields{i}) = settings.(fields{i}); 
                        catch
                            warning("Unable to load settings %s",fields{i});
                        end
                    end
                end
            end
        end
        
        function dimension = add_dimension(obj, dimension_name)
            % Create a new dimension and add it to the Model's dimension array
            d = length(obj.dimensions) + 1;
            dimension = Dimension(d, dimension_name);
            obj.dimensions(d) = dimension;
        end

        function add_dimension_from_data(obj, dimension_name, file_info, fields)
            dimension = add_dimension(obj, dimension_name);
            % Use a datasource for the l2e table
            if file_info(2) == "parquet"
                ds = parquetDatastore(obj.folder + file_info(1));
                tbl = ds.readall;          
            else
                tbl = readtable(obj.folder + file_info(1), 'TextType','string', Delimiter=file_info(2));
            end
            % Load tbl to l2e using internal labels and data types
            dimension.l2e.l = tbl.(fields(1));
            dimension.l2e = convertvars(dimension.l2e, "l", obj.label_datatype); % Change labels to strings
            if length(unique(dimension.l2e.l)) ~= length(dimension.l2e.l)
                error("Fatal error: Dimension labels were not unique for %s.", dimension_name);
            end
            if length(fields) > 1 % There is an index field
                dimension.l2e.e =  tbl.(fields(2));
            else
                dimension.l2e.e = (1:height(tbl))'; % Make an index field
            end
            dimension.l2e = convertvars(dimension.l2e, "e", obj.index_datatype); % Change index to index_datatype
            if length(fields) > 2 % There is a training field
                tbl = convertvars(tbl, fields(3), obj.flag_datatype); % Change training flag to training_datatype
                dimension.l2e.t =  tbl.(fields(2));
            end
        end

        function add_dimension_from_vector(obj, dimension_name, labels)
            dimension = add_dimension(obj, dimension_name);
            % Use a vector for the l2e table
            tbl = array2table(labels, 'VariableNames', "l");
            tbl.e = (1:height(tbl))';
            tbl = convertvars(tbl, "l", obj.label_datatype); % Change datatypes to be safe
            tbl = convertvars(tbl, "e", obj.index_datatype); % Change index datatype
            dimension.l2e = tbl;
        end

        function set_training_markers(obj, markers)
            % Set training markers using an array of dimension marker points
            for d = 1:length(markers)
                dimension = obj.dimensions(d);
                if markers(d) == ""
                    dimension.l2e.t = ones(height(dimension.l2e.l), 1);
                else
                    cutpoint = find(dimension.l2e.l == markers(d));
                    dimension.l2e.t = [ones(cutpoint, 1); zeros(height(dimension.l2e.l)-cutpoint, 1)];
                end
                dimension.l2e = convertvars(dimension.l2e, 't', obj.flag_datatype); % Change training flag to training_datatype
            end
        end

        function partition = add_partition(obj, dimensions, p, name, l2c)
            % Create a new partition and add it to the model's partition array
            if nargin < 4
                l2c = table;
            end
            if nargin < 3
                name = "";
            end
            partition = Partition(dimensions, p, name, l2c);
            obj.partitions = [obj.partitions, partition];
        end
        
        function partitions = add_partitions_from_function(obj, d, function_info, name)
            % Add partition(s) from a function:
            %   d can be an array. 
            %   function_info(1) is function_name, rest are params. Some functions support a normalize string.
            %   name is optional.
            if nargin < 4
                name = "";
            end
            partition_dimensions = obj.dimensions(d);
            if length(function_info) > 1
                l2c = Dimension.(function_info(1))(partition_dimensions, function_info(2:end));
            else
                l2c = Dimension.(function_info(1))(partition_dimensions);
            end
            if iscell(l2c)
                partitions = cell(1, length(l2c));
                for i = 1:length(l2c)
                    l_cols = string(l2c{i}.Properties.VariableNames(startsWith(l2c{i}.Properties.VariableNames, "l")));
                    l2c{i} = convertvars(l2c{i}, l_cols, obj.label_datatype); % Change datatypes to be safe                    
                    partitions{i} = obj.add_partition(partition_dimensions, length(obj.partitions)+1, name, l2c{i});
                end
            else
                l_cols = string(l2c.Properties.VariableNames(startsWith(l2c.Properties.VariableNames, "l")));
                l2c = convertvars(l2c, l_cols, obj.label_datatype); % Change datatypes to be safe                    
                partitions = obj.add_partition(partition_dimensions, length(obj.partitions)+1, name, l2c);
            end
        end   

        function partition = add_partition_from_data(obj, d, file_info, fields, name, normalize)
            % Add partition from file:
            % Notes: 
            %   d can be array.
            % file_info is [file_name, file_type], fields is [s_field, l_fields...], normalize is optional category number array.
            if nargin < 6
                normalize = [];
            end
            if nargin < 5
                name = "";
            end
            % Prep variables
            partition_dimensions = obj.dimensions(d);
            p = length(obj.partitions)+1;
            l2c = table;
            % Use a datasource for the partition's l2c table
            if file_info(2) == "parquet"
                ds = parquetDatastore(obj.folder + file_info(1));
                tbl = ds.readall;          
            else
                tbl = readtable(obj.folder + file_info(1), 'TextType','string', Delimiter=file_info(2));
            end
            % Change the tbl to use internal labels and data types
            tbl = convertvars(tbl, fields(1), obj.variable_datatype); % Change variable datatype
            tbl = convertvars(tbl, fields(2:end), obj.label_datatype); % Change labels to strings
            c_col = "c_" + join(string([partition_dimensions.d]), "x"); % c.f. Partition.c_cols
            l2c.(c_col) = tbl.(fields(1));
            for i = 1:length(d)
                l_col = "l_" + d(i);
                l2c.(l_col) = tbl.(fields(i+1)); % Dimension fields must be listed in correct order              
            end
            % Normalize l2c if requested
            if ~isempty(normalize)
                l2c = l2c(~ismember(l2c.(c_col), normalize), :);
            end
            % Add the partition
            partition = add_partition(obj, partition_dimensions, p, name, l2c);
        end

        function partitions = append_partitions(~, partitions, append_partitions)
            % Append an array of partitions to each cell of a cell array of partitions
            if iscell(partitions)
                existing_partitions = [partitions{:}];
                for i = 1:length(partitions)
                    partitions{i} = [partitions{i}, append_partitions];
                end
            else 
                existing_partitions = partitions;
                partitions = [partitions, append_partitions];
            end
            % Error check
            existing_c_cols = [existing_partitions.c_cols];
            new_c_cols = [append_partitions.c_cols];
            if (any(ismember(new_c_cols, existing_c_cols)))
                    error("Attempted to append partitions covering same dimensions");
            end
        end
    
        function partitions = name_partitions(obj, names)
            % Name the partitions!
            partitions = obj.partitions(1:length(names));
            names = num2cell(names);
            [partitions.name] = names{:};   
        end

        function partitions = get_partitions(obj, names)
            % Get an array of partition objects by name
            matches = ismember([obj.partitions.name], names);
            partitions = obj.partitions(matches);
        end

        function blocks = add_blocks_from_partitions(obj, partitions)
            % Assign each cell of 'partitions' to a new block
            if iscell(partitions)
                blocks = Block.empty(length(partitions), 0);
                for i = 1:length(partitions)
                    blocks(i) = obj.add_block(partitions{i});
                end
            else 
                blocks = obj.add_block(partitions);
            end            
        end

        function block = add_block(obj, partitions, name)
            % Create a new block and add it to the Model's blocks array
            if nargin < 3
                name = "";
            end
            if nargin < 2
                partitions = Partition.empty;
            end
            b = length(obj.blocks)+1;
            block = Block(b, partitions, name);
            obj.blocks(b) = block;
        end

        function blocks = name_blocks(obj, names)
            % Name the blocks!
            blocks = obj.blocks(1:length(names));
            names = num2cell(names);
            [blocks.name] = names{:};   
        end

        function blocks = modify_blocks_add_partitions(~, blocks, partitions)
            % Adds array (not cell) of partitions to each of an array of blocks
            for i = 1:length(blocks)
                blocks(i).partitions = [blocks(i).partitions, partitions];
            end
        end

        function blocks = complete_blocks(obj, blocks)
            % Determine which blocks are missing degenerate partitions and create and assign them.
            if nargin < 2 || isempty(blocks)
                blocks = obj.blocks;
            end
            % Make a full set of degenerate partitions anew
            all_d = [obj.dimensions.d];
            degenerates = Partition.empty;
            for i = 1:length(all_d)
                degenerates(i) = add_partitions_from_function(obj, all_d(i), "degenerate", "auto_dg_" + all_d(i));
            end
            % Iterate through the blocks adding degenerate partition when needed
            for i = 1:length(blocks)
                covered_dimensions = [blocks(i).partitions.dimensions];
                covered_d = [covered_dimensions.d];
                [~, idx] = setdiff(all_d, covered_d);
                obj.modify_blocks_add_partitions(blocks(i), degenerates(idx));
            end
        end

        function build_blocks_C2c(obj)
            % Build the C2c tables for the blocks
            for b = 1:length(obj.blocks)
                % Sort the block's partition array
                [~,ind] = sort([obj.blocks(b).partitions.min_d]);
                obj.blocks(b).partitions = obj.blocks(b).partitions(ind);
                % Create the C2c
                C2c = obj.blocks(b).make_C2c;
                C2c = convertvars(C2c, 'C', obj.variable_datatype);
                obj.blocks(b).C2c = C2c;
            end
        end

        function add_families(obj, families_spec)
            % Adds families from a spec array with row [type param1 ... param4]
            % Note: only param1, param2 required unless spec includes type=2
            f = length(obj.families); % This is the last existing f, which may be 0
            for g = 1:height(families_spec)
                obj.families(f+g) = Family(f+g, families_spec(g,1), families_spec(g, 2:end));
            end
        end
    
        function assign_family(obj, f, b, C)
            % Assigns a family f to block(s) b and category(s) C
            if nargin < 4
                C = [];
            end
            if nargin < 3 || isempty(b)
                b = 1:length(obj.blocks); % Assign to all blocks
            end
            for i = 1:length(b)
                obj.blocks(b(i)).modify_C2f(f,C); 
            end
        end

        function families = families_by_type(obj, type)
            % Return families by type
            families = obj.families(ismember([obj.families.type], type));
        end

        function print_blocks_spec(~)
            % Print the blocks' spec!
            % To be written... 
            %block_partitions = {model.blocks.partitions}
            %block_partitions = [model.blocks.partitions]
            %[block_partitions.get_c_cols]
        end
      
        function configure_sampler(obj)
            % Build sz for full lambda, train indicator
            obj.sz = Model.indices_size(obj.dimensions.indices);
            train_idx = Model.linear_index(obj.dimensions.indices(1), obj.sz);
            obj.is_train = false(prod(obj.sz), 1);
            obj.is_train(train_idx) = true(length(train_idx), 1);
            metro_bf = [];
            % Build the C2e tables for the partitions
            obj.partitions.make_c2e;
            % Set up the blocks 
            for b = 1:length(obj.blocks)
                block = obj.blocks(b);
                block_f = unique(block.C2f.f);
                % Make C2e and C2E
                block.C2e = block.make_C2e;
                e_d = cell(1, length(obj.dimensions));
                for d = 1:length(obj.dimensions)
                    e_d{d} = block.C2e.("e_"+d);
                end
                idx = sub2ind(obj.sz, e_d{:}); 
                block.C2E = zeros(prod(obj.sz), 1);
                block.C2E(idx) = block.C2e.C;       
                % Configure the Gibbs sampling
                block.is_gibbs = any(obj.families(block_f).is_gibbs);
                if block.is_gibbs
                    % Build block.gibbs table (.C, .theta, .f, and later .in and .u too)
                    gibbs_f = block_f([obj.families(block_f).is_gibbs]);
                    block.gibbs = block.C2f(ismember(block.C2f.f, gibbs_f), :); % Assigns .C and .f
                    block.gibbs.theta = ([obj.Thetas + 1 : obj. Thetas + height(block.gibbs)]'); % Assigns theta
                    obj.Thetas = max(block.gibbs.theta); % model.Thetas stores the (total) length of the thetas vector(s) (number of thetas).
                    % Build the gibbs_in_C2E vectors (reduce C2E to relevant C and train)
                    gibbs_C = ismember(block.C2E, block.gibbs.C); % logical array, excludes possibility of s = 0.
                    block.gibbs_in_C2E_C =  block.C2E(gibbs_C & obj.is_train);
                    lambda_idx = (1:prod(obj.sz))';
                    block.gibbs_in_C2E_E =  lambda_idx(gibbs_C & obj.is_train);
                    % Add the .in column to block.gibbs
                    gibbs_in_C = unique(block.gibbs_in_C2E_C);
                    block.gibbs.in = ismember(block.gibbs.C, gibbs_in_C);
                    if ~isempty(gibbs_in_C)
                        block.is_gibbs_in = true;
                    end
                    % Build gibbs_C_idx (works for Gibbs C in and out-of-sample)
                    block.gibbs_C_idx = zeros(max(block.gibbs.C), 1);
                    block.gibbs_C_idx(block.gibbs.C) = [1:length(block.gibbs.C)]';
                end
                % Append [b f] for metro families to master list
                metro_f = block_f(obj.families(block_f).is_metro);
                metro_bf = [metro_bf; b*ones(length(metro_f), 1) metro_f];
            end
            % Set up the families
            if ~isempty(metro_bf)
                [f, ia] = unique(metro_bf(:, 2)); % Get indices of unique families
                b = metro_bf(ia, 1); % This is an array of the (to be) scheduled block index
                for i = 1:length(f) % Configure the Metropolis sampling
                    family = obj.families(f(i)); % Note that family.is_metro = true is already set.
                    block = obj.blocks(b(i)); % Scheduled block
                    other_blocks = obj.blocks(metro_bf(metro_bf(:, 2) == f(i) & metro_bf(:, 1) ~= b(i), 1)); % These are the non-scheduled blocks that contain C from the same family
                    % Build family.metro table (.b, .C, .theta, .in, .scheduled and later .u).
                    % Start with the scheduled block
                    family.metro.C = block.C2f.C(ismember(block.C2f.f, f(i))); % Assigns .C
                    family.metro.b = block.b*ones(size(family.metro.C));
                    family.metro.theta = zeros(size(family.metro.C));
                    if ismember("C", block.gibbs.Properties.VariableNames) % if block.gibbs.C exists, then block.gibbs.theta should be assigned by now
                        family.metro.theta = block.gibbs.theta(ismember(block.gibbs.C, family.metro.C));
                    end
                    family.metro.scheduled = true(size(family.metro.C));
                    % Build the metro_in_C2E vectors (reduce from block's C2E)
                    C2E_metro_C = ismember(block.C2E, family.metro.C); % logical array.
                    family.metro_in_C2E_C =  block.C2E(C2E_metro_C & obj.is_train);
                    lambda_idx = (1:prod(obj.sz))';
                    family.metro_in_C2E_E =  lambda_idx(C2E_metro_C & obj.is_train);
                    % Add the .in column to family.metro
                    metro_in_C = unique(family.metro_in_C2E_C);
                    family.metro.in = false(size(family.metro.C));
                    family.metro.in = ismember(family.metro.C, metro_in_C);
                    % Add the info for the relevant C from the other blocks
                    for j = 1:length(other_blocks)
                        this_block = other_blocks(j);
                        this_metro = table;
                        this_metro.C = this_block.C2f.C(ismember(this_block.C2f.f, f(i))); % Assigns .C
                        this_metro.b = this_block.b*ones(size(this_metro.C));
                        this_metro.theta = zeros(size(this_metro.C));
                        this_metro.theta = this_block.gibbs.theta(ismember(this_block.gibbs.C, this_metro.C));
                        this_metro.scheduled = false(size(this_metro.C));
                        % Calculate the in-sample t/f indicator for other block(s)
                        this_C2E_metro_in_C =  this_block.C2E(ismember(this_block.C2E, this_metro.C) & obj.is_train);
                        this_metro_in_C = unique(this_C2E_metro_in_C);
                        this_metro.in = false(size(this_metro.C));
                        this_metro.in = ismember(this_metro.C, this_metro_in_C);
                        family.metro = [family.metro; this_metro];
                    end
                    % Schedule this (metro) family in the block (if it has in-sample in its scheduled block)
                    if any(family.metro.in & family.metro.scheduled)
                        block.scheduled_metro = [block.scheduled_metro, f(i)];
                    end
                end
            end
            % Make the model-level info tables from block- and family-level ones
            for f = 1:length(obj.families)
                family_info = obj.families(f).metro;
                family_info.f = f*ones(height(family_info), 1);
                if isempty(obj.metro)
                    obj.metro = family_info;
                else
                    obj.metro = [obj.metro; family_info];
                end
            end
            for b = 1:length(obj.blocks)
                block_info = obj.blocks(b).gibbs;
                if isempty(block_info)
                        continue; % Empty Gibbs table so ignore (wrong number of cols)
                end
                block_info.b =  b*ones(height(block_info), 1);
                if isempty(obj.gibbs)
                    obj.gibbs = block_info;
                else
                    obj.gibbs = [obj.gibbs; block_info];
                end                
            end
        end

        function load_training_data(obj, file_info, fields)
            % Load u from data and populate usum in block.gibbs.u (block.is_gibbs=true) and family.metro.u (family.is_metro=true)
            % Notes: 
            %       The data better be small enough to EASILY fit into memory! Don't do this with big parquet files; they need to be read one page at a time.
            %       fields(1) is the u field, fields(2:d+1) are the dimension fields.
            if file_info(2) == "parquet"
                ds = parquetDatastore(obj.folder + file_info(1));
                tbl = ds.readall;          
            else
                tbl = readtable(obj.folder + file_info(1), 'TextType','string', Delimiter=file_info(2));
            end
            % Load tbl to u using internal label and data types
            obj.u.u = tbl.(fields(1));
            l_d = strings(1, length(obj.dimensions)); % Array of labels by dimension
            for d = 1:length(obj.dimensions)
                l_d(d) = "l_"+ d;
                obj.u.(l_d(d)) = tbl.(fields(d+1));
            end
            obj.u = convertvars(obj.u, l_d, obj.label_datatype); % Change labels to strings
            obj.u = convertvars(obj.u, "u", obj.variable_datatype); % Change u to variable type
            % Convert labels to indices, discarding data that is out of range
            include_rows = true(height(obj.u), 1);
            for d = 1:length(obj.dimensions)
                dimension = obj.dimensions(d);
                [exists, dimension_idx] = ismember(obj.u.(l_d(d)), dimension.l2e.l); % dimension.l2e.l may not contain all u labels.
                obj.u.("e_" + d) = zeros(height(obj.u), 1);
                obj.u.("e_" + d)(exists) = dimension.l2e.e(dimension_idx(exists == 1));
                include_rows = include_rows & exists;
            end
            obj.u = obj.u(include_rows, :); % Exclude rows with unmatched labels
            % Build the obj.U linear vector
            e_d = cell(1, length(obj.dimensions));
            for d = 1:length(obj.dimensions)
                e_d{d} = obj.u.("e_" + d);
            end
            obj.U = zeros(prod(obj.sz), 1);
            idx = sub2ind(obj.sz, e_d{:});
            obj.U(idx) = obj.u.u;
            % obj.U(~obj.is_train) = zeros(length(~obj.is_train), 1); % Set non-train U to zero. Not needed as C2E_E vectors are for train only. 
            % Compute u sum for Gibbs blocks
            for b = 1:length(obj.blocks)
                block = obj.blocks(b);
                if block.is_gibbs == true
                    block.gibbs.u = zeros(height(block.gibbs), 1); 
                    if any(block.gibbs.in)
                        u_total = accumarray(block.gibbs_in_C2E_C, obj.U(block.gibbs_in_C2E_E));
                        block.gibbs.u(block.gibbs.in) = u_total;
                    end
                end
            end
            % Compute u sum for metropolis families (C in-sample AND from scheduled block ONLY)
            for f = 1:length(obj.families)
                family = obj.families(f);
                if family.is_metro == true
                    family.metro.u = zeros(height(family.metro), 1); 
                    if any(family.metro.in & family.metro.scheduled) % Note: Could be revised to calc .u for unscheduled metro C
                        u_total = accumarray(family.metro_in_C2E_C, obj.U(family.metro_in_C2E_E));
                        family.metro.u(family.metro.in & family.metro.scheduled) = u_total;
                    end
                end
            end            
        end

        function initialize_storage(obj)
            % Initialize the chain-level cache and storage
            obj.chain_alphabetas = NaN(obj.M, length(obj.families), 2, obj.variable_datatype); 
            obj.chain_thetas = ones(obj.M, obj.Thetas, obj.variable_datatype); % Note:  out-of-sample thetas are initiated at one in storage.
            obj.chain_lambdas = ones(obj.M, prod(obj.sz), obj.variable_datatype); % Caution: this can get big fast!           
            obj.alphabetas = NaN(obj.R, obj.M, length(obj.families), 2, obj.variable_datatype); 
            obj.thetas = ones(obj.R, obj.M, obj.Thetas, obj.variable_datatype); % Note: out-of-sample thetas are initiated at one in the cache, though they aren't used.
        end        

        function initialize(obj)
            % Initialize the alphabetas, thetas, and lambas for each chain and put them in the cache.
            obj.initialize_storage;
            for m = 1:obj.M 
                obj.chain_alphabetas(m, [obj.families.f], :) = obj.families.draw_alphabetas; % Draw the initial alphabetas
                for b = 1:length(obj.blocks)
            	    block = obj.blocks(b);
                    if block.is_gibbs_in == true
                        gibbs_in = block.gibbs(block.gibbs.in, :); % Restrict the block's Gibbs table to in-sample only
                        block_alphabeta =  reshape(obj.chain_alphabetas(m, gibbs_in.f ,:), [], 2); % Using gibbs.f does projection into theta space.
                        block_theta = block.gibbs_sample( block_alphabeta ); % Draw the initial thetas
                        chain_lambda = reshape(obj.chain_lambdas(m, block.gibbs_in_C2E_E), [], 1);
                        chain_lambda = chain_lambda.* block_theta(block.gibbs_C_idx(block.gibbs_in_C2E_C));
                        % Cache the results
                        obj.chain_thetas(m, gibbs_in.theta, :) = block_theta';
                        obj.chain_lambdas(m, block.gibbs_in_C2E_E) = chain_lambda';
                    end
                end
            end
        end

        function train(obj)
            % Infer the parameters of the model, using settings from the autotailor (which may be off)
            obj.initialize;
            obj.iterate(obj.R); % Do R iterations
            if obj.autotailor.on == true
                w = obj.autotailor_compute_warmup;
                fprintf('Autotailor needs %d more iterations to replace warm-up\n', w)
                obj.iterate(w, 1); % Do w iterations, recording them starting at r = 1
            end
        end

        function iterate(obj, R, start_record)
            % Iterate the sampler R times, using the autotailor if autotailor.on = true
            if nargin < 3 
            	start_record = 1;
            end
            if nargin < 2 
            	R = obj.R; % Use default R if not specified
            end
            for r = 1:R % Outer loops
                iteration_timer = tic;
                if obj.autotailor.on == true
                    params_before = obj.autotailor_params;
                end
                for s = 1:obj.S % Inner loops
                    for m = 1:obj.M % Chains
                        % Get the params for chain m from cache
                        alphabeta = reshape(obj.chain_alphabetas(m,:,:), [], 2);
                        theta = reshape(obj.chain_thetas(m,:), [], 1); % Contains out-of-sample thetas
                        lambda = reshape(obj.chain_lambdas(m,:), [], 1);
                        % Sample every parameter!
                        [alphabeta, theta, lambda] = obj.sample(alphabeta, theta, lambda); 
                        % Write the results to cache for chain m
                        obj.chain_alphabetas(m,:,:) = alphabeta;
                        obj.chain_thetas(m,:) = theta';
                        obj.chain_lambdas(m,:) = lambda';
                    end
                    if obj.autotailor.on == true % Use the autotailor to determine convergence
                        params_after = obj.autotailor_params; % M x Params
                        test_stat = mean(log(0.5 * var(params_after - params_before) ./ var(params_before) )); % Var across chains (1 x Params)
                        if test_stat > 0
                            break;
                        end
                    end
                end
                % Optionally print the status message
                if obj.verbose == true
                    fprintf("\t r = %d, s = %d, test = %10.4e, time = %.1fs\n", r, s, test_stat, toc(iteration_timer)) 
                end
                % Write the cache to storage
                obj.alphabetas(r + (start_record - 1),:,:,:) = obj.chain_alphabetas; 
                obj.thetas(r + (start_record - 1),:,:) = obj.chain_thetas;
            end
        end

        function [alphabeta, theta, lambda] = sample(obj, alphabeta, theta, lambda)
            % Sample every parameter, block by block. 
            % Notes:
            %   alphabeta (F x 4), theta (Thetas x 1), lambda (E x 1), sum_lambda, block_alphabeta, block_theta are all column vectors
            %   Things will get weird fast if C2E_E vectors have repeated indices...
            for b = 1:length(obj.blocks)
            	block = obj.blocks(b);
                % Metropolis sampling
                if ~isempty(block.scheduled_metro)
                    for i = 1:length(block.scheduled_metro) 
                        family = obj.families(block.scheduled_metro(i));
                        alphabeta = family.metropolis_sample(alphabeta, theta, lambda);
                    end
                end
                % Gibbs sampling
                if block.is_gibbs_in == true
                    gibbs_in = block.gibbs(block.gibbs.in, :); % Restrict the block's Gibbs table to in-sample only
                    block_alphabeta = alphabeta(gibbs_in.f, :); % Note that alphabeta is projected into theta space
                    sum_lambda = accumarray(block.gibbs_in_C2E_C, lambda(block.gibbs_in_C2E_E)); % gibbs_in_C2E_C and _E for train only.
                    block_theta = block.gibbs_sample( [block_alphabeta(:,1) + gibbs_in.u, block_alphabeta(:,2) + (sum_lambda ./ theta(gibbs_in.theta))]); % Update thetas.
                    % Update theta and lambda
                    theta_update = block_theta ./ theta(gibbs_in.theta);
                    theta(gibbs_in.theta) = block_theta;
                    lambda(block.gibbs_in_C2E_E) = lambda(block.gibbs_in_C2E_E) .* theta_update(block.gibbs_C_idx(block.gibbs_in_C2E_C)); % Update lambda.
                end
            end
        end

        function w = autotailor_compute_warmup(obj)
            % Get warmup iterations using autotailor
            across_chain_variance  = squeeze( var(obj.autotailor_params([]), 0, 2) ); % autotailor_params([]) returns thetas [R x M x Thetas(in)], so across_chain_variance is [R x Thetas(in)]
            max_W = min( ceil(obj.R * obj.autotailor.W_max), obj.R - 1 );
            for w = 1:max_W
                test_stat = sum( across_chain_variance(w, :) ./ mean(across_chain_variance(w+1:end, :)) ); % interation's across_chain_variance [1 X Thetas(in)] / mean for remaining iterations [1 x Thetas(in)]
                if test_stat <= size(across_chain_variance, 2) % size(across_chain_variance, 2) = Thetas(in)
                    break;
                end
            end         
        end

        function params = autotailor_params(obj, r)
            % Retrieve the autotailor params (either thetas or lambdas) for thinning and warmup
            % Note(s):  r is a vector and can be [1:obj.R]
            if nargin < 2
                if obj.autotailor.params == "thetas"
                    if ~isfield(obj.autotailor, 'indices') || isempty(obj.autotailor.indices) % Note isfield is needed as empty array doesn't initialize due to MATLAB bug
                        params = obj.chain_thetas(:, obj.gibbs.theta(obj.gibbs.in));
                    else
                        params = obj.chain_thetas(:, obj.autotailor.indices);                  
                    end
                elseif obj.autotailor.params == "lambdas"
                    if ~isfield(obj.autotailor, 'indices') || isempty(obj.autotailor.indices)
                        params = obj.chain_lambdas;
                    else
                        params = obj.chain_lambdas(:, obj.autotailor.indices);
                    end                
                end 
            % Only thetas are available in storage    
            elseif isempty(r) 
                if ~isfield(obj.autotailor, 'indices') || isempty(obj.autotailor.indices)
                    params = obj.thetas(:, :, obj.gibbs.theta(obj.gibbs.in));
                else
                    params = obj.thetas(:, :, obj.autotailor.indices);                  
                end
            else 
                if ~isfield(obj.autotailor, 'indices') || isempty(obj.autotailor.indices)
                    params = obj.thetas(r, :, obj.gibbs.theta(obj.gibbs.in));
                else
                    params = obj.thetas(r, :, obj.autotailor.indices);                  
                end               
            end
        end

        function draw_out_of_sample(obj)
            % Draw any out-of-sample thetas. (Users shouldn't have set up any out-of-sample alphabetas...).
            gibbs_out = obj.gibbs(~obj.gibbs.in, :);
            alphabeta = obj.alphabetas(1:obj.R, 1:obj.M, gibbs_out.f, :);
            linear_alphabeta = reshape(reshape(alphabeta, 1, 1, [], 2), [], 2); 
            linear_theta = Block.gibbs_sample(linear_alphabeta);
            theta = reshape(linear_theta, obj.R, obj.M, []);
            obj.thetas(1:obj.R, 1:obj.M, gibbs_out.theta) = theta;           
        end

        function [posteriors_mean, posteriors_sd, priors_mean, priors_sd] = summarize(obj)
            % Compute the summary results (mean and sd) for priors and posteriors and store them in the gibbs table
            priors = reshape(mean(obj.alphabetas, [1 2]), [], 2); % F x 2 (alpha, beta)
            priors_mean = priors(:, 1) ./ priors(:, 2);  % F x 1 
            priors_sd = priors(:, 1) ./ ( priors(:, 2) .^ 2 ); % F x 1 
            posteriors_mean = reshape(mean(obj.thetas, [1 2]), [], 1); % Thetas x 1
            posteriors_sd = reshape(std(obj.thetas, 0, [1 2]), [], 1); % Thetas x 1
            obj.gibbs.prior_mean = priors_mean(obj.gibbs.f);
            obj.gibbs.prior_sd =  priors_sd(obj.gibbs.f);
            obj.gibbs.post_mean = posteriors_mean(obj.gibbs.theta);
            obj.gibbs.post_sd = posteriors_sd(obj.gibbs.theta);
        end

        function lambdas = lambda(obj, R, M )
            % Make and return R x M lambda vectors
            if nargin <2
                R = obj.R;
                M = obj.M;
            end
            lambdas = ones(R, M, prod(obj.sz));
            for b = 1:length(obj.blocks)
                block = obj.blocks(b);
                if block.is_gibbs == true
                    % Test this!
                    lambdas(1:R, 1:M, block.C2E ~= 0) = lambdas(1:R,1:M, block.C2E ~= 0).* obj.thetas(1:R, 1:M, block.gibbs.theta( block.gibbs_C_idx( block.C2E(block.C2E ~= 0) )));
                end
            end
        end

        function plot_traces_by_family_type(obj, type)
            % If there is are one or more families of the specified type, plot w,z traces for them
            selected_families = obj.families_by_type(type);
            if ~isempty(selected_families)
                hold off;
                title("Type " + num2str(type) + " family traces");
                for i = 1:length(selected_families)
                    alpha = squeeze(obj.alphabetas(:, :, selected_families(i).f, 1));
                    beta = squeeze(obj.alphabetas(:, :, selected_families(i).f, 2));
                    w = log(alpha ./ beta);
                    z = -0.5*log(alpha);
                    subplot(length(selected_families), 2, i);
                    plot(w);
                    ylabel('w');
                    subplot(length(selected_families), 2, i*2);
                    plot(z);       
                    ylabel('z');
                end
            end
        end

        function plot_traces_by_theta(obj, selection)
            % Plot traces for selected thetas. 
            if isempty(selection)
                selection = randsample(obj.Thetas, min(10,obj.Thetas)); % Randomly draw up to 10 thetas
            end
            hold off;
            title("Trace for selected thetas");
            for i = 1:length(selection)
                subplot(length(selection), 1, i)
                selected_theta = squeeze(obj.thetas(:,:,selection(i)));
                plot(selected_theta)
            end
        end       
    end

    methods (Static)
        function [linear_index, sz] = linear_index(indices, sz)
            % Notes:
            %   indices must be col vector(s) (e_d x 1) in *dimension ordered* cell array (1 x D)
            %   sz (optional) is array of max(e_d) (1 x D)
            if nargin < 2
                if iscell(indices)
                    sz = cellfun(@max, indices);
                else
                    sz = length(indices);
                end
            end
            if size(indices, 2) == 1 % No projected needed
                linear_index = indices;
            else
                % Use ndgrid for cartesian product and sub2ind for linear projection
                index_grid = cell(1, numel(indices));
                [index_grid{:}] = ndgrid(indices{:});
                index_grid = cellfun(@(x) reshape(x,[],1), index_grid, 'UniformOutput', false);
                linear_index = sub2ind(sz, index_grid{:});                
            end   
        end

        function sz = indices_size(indices)
            % Return the indices size (fast method)
            if iscell(indices)
                sz = cellfun(@max, indices);
            else
                sz = length(indices);
            end
        end        
    end
end
