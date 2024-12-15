% New thresholds based on observations
red_threshold_low = 0.2;  % Lower bound for red pixels
red_threshold_high = 0.6; % Upper bound to avoid false reds
green_threshold_low = 0.4; % Avocado should have stronger green
green_threshold_high = 0.9; % Upper bound for strong green pixels
blue_threshold_low = 0.0;  % Keep blue low, as it is not dominant for most fruits

% Normalizing RGB channels
red_channel_norm = double(red_channel) / 255;
green_channel_norm = double(green_channel) / 255;
blue_channel_norm = double(blue_channel) / 255;

% Recounting pixel colors using adjusted thresholds
red_pixel_count = sum(red_channel_norm(:) > red_threshold_low & red_channel_norm(:) < red_threshold_high & green_channel_norm(:) < 0.4 & blue_channel_norm(:) < 0.4);
green_pixel_count = sum(green_channel_norm(:) > green_threshold_low & green_channel_norm(:) < green_threshold_high & red_channel_norm(:) < 0.4 & blue_channel_norm(:) < 0.4);
blue_pixel_count = sum(blue_channel_norm(:) > blue_threshold_low & blue_channel_norm(:) < 0.4 & red_channel_norm(:) < 0.4 & green_channel_norm(:) < 0.4);

% Recheck the pixel counts
fprintf('Red Pixel Count: %d\n', red_pixel_count);
fprintf('Green Pixel Count: %d\n', green_pixel_count);
fprintf('Blue Pixel Count: %d\n', blue_pixel_count);

% Test data for current image
test_data = [red_pixel_count, green_pixel_count, blue_pixel_count];

% KNN classification (k=3)
k = 3;
distances = sqrt(sum((train_data - test_data).^2, 2));
[~, sorted_indices] = sort(distances);
nearest_neighbors = train_labels(sorted_indices(1:k));

% Finding the mode of the k-nearest neighbors
nearest_neighbors_cat = categorical(nearest_neighbors);
fruit_type_knn = mode(nearest_neighbors_cat);

% Display the determined fruit type
fprintf('Fruit Type (KNN): %s\n', char(fruit_type_knn));
