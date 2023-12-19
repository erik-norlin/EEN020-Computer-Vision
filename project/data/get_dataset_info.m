% -*- Octave -*-

% Datasets 1-5 are relatively easy, since they have little lens distortion and
% no dominant scene plane. You should get reasonably good reconstructions for
% these datasets using RANSAC to estimate E only (CE2 in assignment 4).

% Datasets 6-9 are much more difficult, since the scene is dominated by a near
% planar structure. Also the lens distortion is larger for the used camera,
% hence feel free to adjust the pixel_threshold if necessary.

function [K, img_names, init_pair, pixel_threshold] = get_dataset_info(dataset)
  if dataset == 1
    img_names = {"data/1/kronan1.JPG", "data/1/kronan2.JPG"};
    im_width = 1936; im_height = 1296;
    focal_length_35mm = 45.0; % from the EXIF data
    init_pair = [1, 2];
    pixel_threshold = 1.0;
  elseif dataset == 2
    % Corner of a courtyard
    img_names = {"data/2/DSC_0025.JPG", "data/2/DSC_0026.JPG", "data/2/DSC_0027.JPG", "data/2/DSC_0028.JPG", "data/2/DSC_0029.JPG", ...
                 "data/2/DSC_0030.JPG", "data/2/DSC_0031.JPG", "data/2/DSC_0032.JPG", "data/2/DSC_0033.JPG"};
    im_width = 1936; im_height = 1296;
    focal_length_35mm = 43.0; % from the EXIF data
    init_pair = [1, 9];
    pixel_threshold = 1.0;
  elseif dataset == 3
    % Smaller gate of a cathetral
    img_names = {"data/3/DSC_0001.JPG", "data/3/DSC_0002.JPG", "data/3/DSC_0003.JPG", "data/3/DSC_0004.JPG", "data/3/DSC_0005.JPG", ...
                 "data/3/DSC_0006.JPG", "data/3/DSC_0007.JPG", "data/3/DSC_0008.JPG", "data/3/DSC_0009.JPG", "data/3/DSC_0010.JPG", ...
                 "data/3/DSC_0011.JPG", "data/3/DSC_0012.JPG"};
    im_width = 1936; im_height = 1296;
    focal_length_35mm = 43.0; % from the EXIF data
    init_pair = [5, 8];
    pixel_threshold = 1.0;
  elseif dataset == 4
    % Fountain
    img_names = {"data/4/DSC_0480.JPG", "data/4/DSC_0481.JPG", "data/4/DSC_0482.JPG", "data/4/DSC_0483.JPG", "data/4/DSC_0484.JPG", ...
                 "data/4/DSC_0485.JPG", "data/4/DSC_0486.JPG", "data/4/DSC_0487.JPG", "data/4/DSC_0488.JPG", "data/4/DSC_0489.JPG", ...
                 "data/4/DSC_0490.JPG", "data/4/DSC_0491.JPG", "data/4/DSC_0492.JPG", "data/4/DSC_0493.JPG"};
    im_width = 1936; im_height = 1296;
    focal_length_35mm = 43.0; % from the EXIF data
    init_pair = [5, 10];
    pixel_threshold = 1.0;
  elseif dataset == 5
    % Golden statue
    img_names = {"data/5/DSC_0336.JPG", "data/5/DSC_0337.JPG", "data/5/DSC_0338.JPG", "data/5/DSC_0339.JPG", "data/5/DSC_0340.JPG", ...
                 "data/5/DSC_0341.JPG", "data/5/DSC_0342.JPG", "data/5/DSC_0343.JPG", "data/5/DSC_0344.JPG", "data/5/DSC_0345.JPG"};
    im_width = 1936; im_height = 1296;
    focal_length_35mm = 45.0; % from the EXIF data
    init_pair = [3, 7];
    pixel_threshold = 1.0;
  elseif dataset == 6
    % Detail of the Landhaus in Graz.
    img_names = {"data/6/DSCN2115.JPG", "data/6/DSCN2116.JPG", "data/6/DSCN2117.JPG", "data/6/DSCN2118.JPG", "data/6/DSCN2119.JPG", ...
                 "data/6/DSCN2120.JPG", "data/6/DSCN2121.JPG", "data/6/DSCN2122.JPG"};
    im_width = 2272; im_height = 1704;
    focal_length_35mm = 38.0; % from the EXIF data
    init_pair = [2, 4];
    pixel_threshold = 1.0;
  elseif dataset == 7
    % Building in Heidelberg.
    img_names = {"data/7/DSCN7409.JPG", "data/7/DSCN7410.JPG", "data/7/DSCN7411.JPG", ...
                 "data/7/DSCN7412.JPG", "data/7/DSCN7413.JPG", "data/7/DSCN7414.JPG", "data/7/DSCN7415.JPG"};
    im_width = 2272; im_height = 1704;
    focal_length_35mm = 38.0; % from the EXIF data
    init_pair = [1, 7];
    pixel_threshold = 1.0;
  elseif dataset == -8
    % Roman building in Trier
    img_names = {"data/8/DSCN7225.JPG", "data/8/DSCN7226.JPG", "data/8/DSCN7227.JPG", "data/8/DSCN7228.JPG", "data/8/DSCN7229.JPG", ...
                 "data/8/DSCN7230.JPG", "data/8/DSCN7231.JPG", "data/8/DSCN7232.JPG", "data/8/DSCN7233.JPG", "data/8/DSCN7234.JPG"};
    im_width = 2272; im_height = 1704;
    focal_length_35mm = 38.0; % from the EXIF data
    init_pair = [4, 5];
    pixel_threshold = 1.0;
  elseif dataset == 8
    % Relief
    img_names = {"data/8/DSCN5540.JPG", "data/8/DSCN5541.JPG", "data/8/DSCN5542.JPG", "data/8/DSCN5543.JPG", "data/8/DSCN5544.JPG", "data/8/DSCN5545.JPG",...
                 "data/8/DSCN5546.JPG", "data/8/DSCN5547.JPG", "data/8/DSCN5548.JPG", "data/8/DSCN5549.JPG", "data/8/DSCN5550.JPG"};
    im_width = 2272; im_height = 1704;
    focal_length_35mm = 38.0; % from the EXIF data
    init_pair = [4, 7];
    pixel_threshold = 1.0;
  elseif dataset == 9
    % Triceratops model on a poster.
    img_names = {"data/9/DSCN5184.JPG", "data/9/DSCN5185.JPG", "data/9/DSCN5186.JPG", "data/9/DSCN5187.JPG", "data/9/DSCN5188.JPG", ...
                 "data/9/DSCN5189.JPG", "data/9/DSCN5191.JPG", "data/9/DSCN5192.JPG", "data/9/DSCN5193.JPG"};
    im_width = 2272; im_height = 1704;
    focal_length_35mm = 38.0; % from the EXIF data
    init_pair = [4, 6];
    pixel_threshold = 1.0;
  elseif dataset == 10
    % Add your optional datasets...
  end

  focal_length = max(im_width, im_height) * focal_length_35mm / 35.0
  K = [focal_length 0 im_width/2; 0 focal_length im_height/2; 0 0 1]
end
