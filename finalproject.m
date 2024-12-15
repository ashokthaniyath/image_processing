% Training data with more sample values for Red Apple, Green Apple, Cherry, Blueberry, and Grape
train_data = [
    % Red Apples (dominantly red with lower green and blue values)
    4600, 350, 120;
    4800, 290, 80;
    4700, 320, 100;
    4500, 300, 110;
    4900, 250, 90;
    
    
    % Bananas (high red and green values, very low blue)
    2200, 2200, 50;
    2300, 2100, 40;
    2400, 2000, 30;
    2100, 2300, 60;
    2250, 2250, 45;
    
    % Blueberries (dominantly blue with low red and green values)
    200, 300, 3200;
    220, 280, 3150;
    210, 310, 3250;
    190, 290, 3300;
    205, 275, 3180;
    
    % Grapes (moderate red, green, and blue values)
    1200, 2200, 1500;
    1300, 2100, 1600;
    1250, 2250, 1550;
    1180, 2150, 1520;
    1220, 2180, 1480;
];

% Corresponding labels for each entry in the training data
train_labels = {
    'Red Apple';
    'Red Apple';
    'Red Apple';
    'Red Apple';
    'Red Apple';
       
    
    'Banana';
    'Banana';
    'Banana';
    'Banana';
    'Banana';

    'Blueberry';
    'Blueberry';
    'Blueberry';
    'Blueberry';
    'Blueberry';
    
    'Grape';
    'Grape';
    'Grape';
    'Grape';
    'Grape';
};
% Load the image from the specified path
image = imread('Grapes.jpeg');

% Display the loaded image on the screen
imshow(image);

% Get the dimensions of the image
[image_height, image_width, ~] = size(image);

% Define highlight threshold and cap pixel intensities to reduce bright areas
highlight_threshold = 200;
image(image > highlight_threshold) = highlight_threshold;

% Separate the color channels of the image
red_channel = image(:, :, 1);
green_channel = image(:, :, 2);
blue_channel = image(:, :, 3);

% Set color thresholds to accurately capture dominant colors in each channel
color_threshold = 100;

% Count red pixels (predominantly red, low in green and blue)
red_pixel_count = sum(red_channel(:) > color_threshold & ...
                      green_channel(:) < color_threshold & ...
                      blue_channel(:) < color_threshold);

% Count green pixels (predominantly green, low in red and blue)
green_pixel_count = sum(red_channel(:) < color_threshold & ...
                        green_channel(:) > color_threshold & ...
                        blue_channel(:) < color_threshold);

% Count blue pixels (predominantly blue, low in red and green)
blue_pixel_count = sum(red_channel(:) < color_threshold & ...
                       green_channel(:) < color_threshold & ...
                       blue_channel(:) > color_threshold);

% Display the pixel counts for red, green, and blue
fprintf('Red Pixel Count: %d\n', red_pixel_count);
fprintf('Green Pixel Count: %d\n', green_pixel_count);
fprintf('Blue Pixel Count: %d\n', blue_pixel_count);

% Features of the current image (Red, Green, Blue pixel counts)
test_data = [red_pixel_count, green_pixel_count, blue_pixel_count];

% Set k for k-NN
k = 3;

% Calculate Euclidean distance between test data and each training sample
distances = sqrt(sum((train_data - test_data).^2, 2));

% Find the k-nearest neighbors based on smallest distances
[~, sorted_indices] = sort(distances);
nearest_neighbors = train_labels(sorted_indices(1:k));

% Convert nearest_neighbors to a categorical array to compute mode
nearest_neighbors_cat = categorical(nearest_neighbors);

% Find the most common fruit label among the nearest neighbors
fruit_type_knn = mode(nearest_neighbors_cat);

% Display the identified fruit type using k-NN
fprintf('Fruit Type (KNN): %s\n', char(fruit_type_knn));