% CatFish: a Bayesian Categorical Variable Multiplicative Poisson Economic Demand Model
% (c) Ed Egan, Petabyte Economics Corp., Jan. 2024. All Rights Reserved.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 1
%
% This is the "Startup Cities" example, using "CatFish" a Bayesian categorical variable
% multiplicative Poisson model. The example infers the number of first "growth" venture
% capital financings of startups in 198 US towns and cities each year for three industries
% from 1981 to 2025. 
%
% The model has:
%   3 dimensions (198 places x 3 industries x 45 years)
%   16 user-defined partitions (2 are degenerate)
%   13 blocks, corresponding to 6 categorical variables, including:
%       3 x 1-d global factors (blocks 1, 2, 3)
%       1 x 2-d global interaction factor (block 4)
%       1 x 2-d local factor (block 5)
%       1 x 1-d scale factor (blocks 6-13)
%   5 families:
%       3 x type 1 (Gibbs only)
%       1 x type 2 (Metropolis only)
%       1 x type 3 (Gibbs + Metropolis)
%
% Required data files:
%   sc_dim_place.txt - Dimension file for place, with place -> state partition.
%   sc_dim_year.txt - Dimension file for year (but not used as such), with year -> decade partition.
%   sc_anchorfund_cat.txt - Partition file, with anchorcat -> place, year ("Local factor")
%   sc_growthdeals.txt - Training data with growthdeals -> placestate,industry,year

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preamble
clear all;
addpath('classes/');
time_start = tic;
%warning('off','all');

settings = struct();
settings.folder = "data/example1/";
settings.verbose = true;
settings.M = 8; % Number of chains
settings.R = 16; % Outer iterations of the sampler
settings.S = 32; % Inner iterations of the sampler. Note: This isn't enough to get convergence.
settings.autotailor = struct; 
settings.autotailor.on = true; % Use the autotailor, stopping inner iterations if convergence is reached and removing warmup outer iterations
settings.autotailor.params = "thetas"; % Use thetas (rather than lambdas) to determine convergence.
settings.autotailor.W_max = 0.25; % Maximum fraction of outer iterations to use for warmup.
%settings.autotailor.indices = [1:419]; % Optionally specify parameter indices for the autotailor. Default is to use all in-sample. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build the model
tic
model = Model("StartupCitiesExample1", settings); % Call the Model constructor, passing in the new defaults
fprintf("Building the topology for %s...\n", model.name);

% Add the dimensions in order 
model.add_dimension_from_data("place", ["sc_dim_place.txt", "tab"], ["placestate","idx","train"]);                              %d=1, place dimension (198: Addison, TX, ..., Woburn, MA)
model.add_dimension_from_vector("industry", ["Information Technology","Medical/Health/Life Science","Non-High Technology"]');   %d=2, industry dimension (3: IT, Life Science, Non-HT)
model.add_dimension_from_vector("year", string([1981:2025])');                                                                  %d=3, year dimension (45: 1981 to 2025)
fprintf("\t %d dimensions: %s\n", length(model.dimensions), join([model.dimensions.name], ","));

% Set the dimensions' training markers
model.set_training_markers(["","","2020"]); % Overwrites any existing training flags loaded with the dimensions

% Add the partitions, which divide up (a subset) one or more dimensions' labels into categories c
model.add_partitions_from_function(1, "complete");                                          % p=1: place (d = 1)
model.add_partitions_from_function(2, ["complete", "Non-High Technology"]);                 % p=2: industry (normalized l="Non-High Technology") (d = 2)
model.add_partitions_from_function(3, "complete");                                          % p=3: year (d = 3)
model.add_partition_from_data(1, ["sc_dim_place.txt", "tab"], ["statenum","placestate"]);   % p=4: state (d = 1)
model.add_partition_from_data(3, ["sc_dim_year.txt", "tab"], ["decade","year"], [], 1);     % p=5: decade (no name, normalized c=1) (d = 3)
model.add_partition_from_data([1 3], ["sc_anchorfund_cat.txt", "tab"], ...                  % p=6: anchorcat (d = 1,3)
    ["anchorcat","placestate", "year"]);
model.add_partitions_from_function(1, "degenerate");                                        % p=7: degenerate (d = 1)
model.add_partitions_from_function(2, "degenerate");                                        % p=8: degenerate (d = 2)
scale_partitions = model.add_partitions_from_function(3, ["scale", 5, true]);               % p=9-16: Scale function creates multiple partitions, returned in a cell array (d = 3)
scale_interactions = model.append_partitions(scale_partitions, [model.partitions(7)]); % Interacts with a degenerate partition as an example of the function
model.name_partitions(["place","ind","year","state","decade","anchorfund","dg_1","dg_2"]);
fprintf("\t %d partitions\n", length(model.partitions));

% Provide the blocks' specification, so that each block has partitions that cover all three dimension.
model.add_blocks_from_partitions([{...
    [model.get_partitions(["place","dg_2"])], ...   % b=1: place
    [model.partitions(2)], ...                      % b=2: industry (normalized)
    [model.partitions(3)], ...                      % b=3: year (type 2, "vibrations" - not Gibbs sampled)
    [model.partitions(4:5)], ...                    % b=4: state-decade interaction
    [model.partitions(6)], ...                      % b=5: anchorfund (local factor)
    }, scale_interactions]);                        % b=6-13: 5yr scale factors (type 3, "random walk with drift")
model.name_blocks(["place","ind","year","state-decade", "anchorfund", ...
    "scale86", "scale91", "scale96", "scale01", "scale06", "scale11", "scale16", "scale21"]); 
model.modify_blocks_add_partitions([model.blocks(3)],[model.partitions(7), model.partitions(8)]); % Manually add two degenerate partitions to complete block 3
model.complete_blocks; % Automatically add degenerate partitions where needed to complete blocks
model.build_blocks_C2c;
fprintf("\t %d blocks\n", length(model.blocks));

% Add families
model.add_families([ 1 0.6 0.6; ...         % f=1: type 1 (Started with [1 1])
                     1 1.25 1.25; ...       % f=2: type 1 (Started with [1.5 1.5])
                     1 5 5 ]);              % f=3: type 1 (Started with [2 2])
model.add_families([ 2 -1.7 0.5]);          % f=4: type 2 (Started with [-3 0.5])
model.add_families([ 3 0.15 0.1 -0.7 0.5]); % f=5, type 3 (Started with [0.1 0.1 -3 0.5])
% and assign them to blocks' sets 
model.assign_family(1, []); % Assign family 1 as the default priors everywhere
model.assign_family(2, 2, [2 3]); % Assign categories 2&3 (Life Science and Non-high tech) from block 2 to family 2. 
model.assign_family(3, 2, [1]); % Assign category 1 (IT) from block 2 to family 3. 
model.assign_family(4, 3); % Assign block 3 categories to family 4 (type 2, "vibrations" - not Gibbs sampled)
model.assign_family(5, [6:13]); % Assign 5-year scale blocks to family 5 (type 3, "random walk with drift").
fprintf("\t %d families\n", length(model.families));

% End of build
fprintf("Model built in...\t %.1fs\n", toc);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configure the sampler and infer the model's parameters
tic;
fprintf("Configuring the model for inference...\t");
model.configure_sampler; % Creates the structures and vectors to needed to compute u sum and do the inference. 
model.load_training_data(["sc_growthdeals.txt", "tab"],["growthdeals","placestate","industry","year"])
fprintf("%.1fs\n", toc);

% Estimate the thetas and alphabetas!
tic;
fprintf("In-sample inference (autotailor: %s):\n", string(model.autotailor.on));
model.train; 
fprintf("Out-of-sample inference\n");
model.draw_out_of_sample;
fprintf("Inference completed in...\t %.1fs\n", toc);

% Compute and save the Gibbs inferences
model.summarize; % Stores the means and sd of the posteriors in the model.gibbs table.
writetable(model.gibbs, settings.folder + "GibbsInferences.txt", 'Delimiter','\t');

% Construct lambda
tic;
fprintf("Build full Lambda (in and out-of-sample)...\n");
lambdas = model.lambda(); % prod(model.sz) must be small to build all lambda for each iteration (R x M)
fprintf("Lambda built in...\t %.1fs\n", toc);

%%%%%%%%%%%%%%%%%%%%%%%
% Diagnostics

figure(1);
model.plot_traces_by_family_type(2); % If there is are one or more type 2 families, plot traces for them
figure(2);
model.plot_traces_by_family_type(3); % If there is are one or more type 3 families, plot traces for them
figure(3);
[~,ia] = unique(model.gibbs.b);
first_thetas = model.gibbs.theta(ia); % Get the first theta from each block
model.plot_traces_by_theta(first_thetas); % Plot traces for the first theta from each block

%%%%%%%%%%%%%%%%%%%%%%%
% Analysis

% Do the time series plot, inferred versus actual
figure(4);
hold on;
lambdas_3d = reshape(lambdas, [model.R*model.M, model.sz]); % draws x (place x industry x year)
inferred_time_total = squeeze(sum(lambdas_3d, [2 3]))'; % year x draws;
inferred_mean = mean(inferred_time_total, 2);
observed_time_total = squeeze(sum(reshape(model.U, model.sz), [1 2])); % year x 1
time_labels = categorical(model.dimensions(3).l2e.l); %year x 1
training = model.dimensions(3).l2e.t; 
observed_time_total(~training) = nan(sum(~training), 1);
time_labels_training = time_labels(training);
plot(time_labels(training), inferred_mean(training, :), 'Color' ,"#850101"); % Deep Red
plot(time_labels(training), observed_time_total(training, :), 'Color' ,"#056608"); % Deep green
xlabel("year");
xticks(time_labels([1:2:length(time_labels(training))]));
xlim([time_labels(1) time_labels_training(end)]);
set(gca, 'XTickLabelRotation', 45);
ylabel("Inferred vs. Actual New Startup Financings");
title("Number of New Startup Financings 1981 to 2020");
legend("Inferred","Actual")

% Do the bubble plots for the scale factors
figure(5);
f_type3 = [model.families([model.families.type] == 3).f];
thetas_type3 = model.gibbs.theta(ismember(model.gibbs.f, f_type3));
draws_type3 = reshape(model.thetas(:,:,thetas_type3), [], length(thetas_type3));
%scatter((1:8)',draws_type3');
bubblechart((1:8)',mean(draws_type3)',std(draws_type3)');
ylabel("128 draws of theta (size is std.dev.)");
xlabel("5-year offset scale factors");
xticklabels({"1986->","1991->","1996->","2001->","2006->","2011->","2016->","2021->"});
title("Individual (Not Compounded) Scale Factor Effects");

% Do the shotgun plots for the scale factors
figure(6);
years = (1981:2025);
paths = ones(128, length(years)-1);
for step = 1:8
    first = step*5;
    last = ((step+1)*5);
    paths(:,(first:last-1)) = paths(:,(first:last-1)) .* repmat(draws_type3(:,step), [1,(last-first)]);
end
plot(paths');
ylabel("128 Scale Factor Theta Paths");
xlabel("Year");
xticklabels(years(1:5:45));
xline(39);
title("Cummulative Scale Factor Effects by Year");

% Look at last two thetas' distros (and simulate the future one)
theta_sz  = size(model.thetas);
thetas = reshape(model.thetas, [], theta_sz(end));
figure(7);
%clf(7);
hold on;
histogram(draws_type3(:,7),'facealpha',.7,'edgecolor','none');
hold on;
histogram(draws_type3(:,8),'facealpha',.3,'edgecolor','none');
scale8_mean = mean(draws_type3(:,8));
scale8_sd = std(draws_type3(:,8));
pd = makedist('Normal','mu',scale8_mean,'sigma',scale8_sd);
t = truncate(pd,0,inf);
r = random(t, 128, 1);
histogram(r,'facealpha',.5,'edgecolor','none');
legend("2016-> (Inferred)","2021-> (Inferred)","2021-> (Simulated)");
title("Histograms of scale factor distributions");
